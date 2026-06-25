use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use wordpipe_protocol::{
    is_backend, is_model_profile, BACKENDS, BUS_NAME, DEFAULT_BACKEND, DEFAULT_MODEL_PROFILE,
    DEFAULT_NUM_THREADS, DEFAULT_SAMPLE_RATE, DEFAULT_SHORTCUT, MODEL_PROFILES, OBJECT_PATH,
};
use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedValue, Value};
use zbus::{connection, interface};

type VariantMap = HashMap<String, OwnedValue>;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    replace: bool,
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ServiceConfig {
    backend: String,
    model_profile: String,
    input_device: String,
    shortcut: String,
    model_root: String,
    worker_path: String,
    model_installer_path: String,
    sample_rate: u32,
    num_threads: u32,
    spoken_punctuation: bool,
    insert_partials: bool,
    stream_insert_delay_ms: u32,
    show_overlay: bool,
}

#[derive(Default, Deserialize, Serialize)]
struct PersistedConfig {
    backend: Option<String>,
    model_profile: Option<String>,
    input_device: Option<String>,
    shortcut: Option<String>,
    model_root: Option<String>,
    worker_path: Option<String>,
    model_installer_path: Option<String>,
    sample_rate: Option<u32>,
    num_threads: Option<u32>,
    spoken_punctuation: Option<bool>,
    insert_partials: Option<bool>,
    stream_insert_delay_ms: Option<u32>,
    show_overlay: Option<bool>,
}

impl From<&ServiceConfig> for PersistedConfig {
    fn from(config: &ServiceConfig) -> Self {
        Self {
            backend: Some(config.backend.clone()),
            model_profile: Some(config.model_profile.clone()),
            input_device: Some(config.input_device.clone()),
            shortcut: Some(config.shortcut.clone()),
            model_root: Some(config.model_root.clone()),
            worker_path: Some(config.worker_path.clone()),
            model_installer_path: Some(config.model_installer_path.clone()),
            sample_rate: Some(config.sample_rate),
            num_threads: Some(config.num_threads),
            spoken_punctuation: Some(config.spoken_punctuation),
            insert_partials: Some(config.insert_partials),
            stream_insert_delay_ms: Some(config.stream_insert_delay_ms),
            show_overlay: Some(config.show_overlay),
        }
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            backend: DEFAULT_BACKEND.to_string(),
            model_profile: DEFAULT_MODEL_PROFILE.to_string(),
            input_device: String::new(),
            shortcut: DEFAULT_SHORTCUT.to_string(),
            model_root: default_model_root(),
            worker_path: default_worker_path(),
            model_installer_path: default_model_installer_path(),
            sample_rate: DEFAULT_SAMPLE_RATE,
            num_threads: DEFAULT_NUM_THREADS,
            spoken_punctuation: true,
            insert_partials: true,
            stream_insert_delay_ms: 0,
            show_overlay: true,
        }
    }
}

struct ServiceData {
    config: ServiceConfig,
    listening: bool,
    installing: bool,
    loading_model: bool,
    model_loaded: bool,
    session_id: u64,
    seq: u64,
    partial_text: String,
    last_error: String,
    worker: Option<WorkerProcess>,
}

impl Default for ServiceData {
    fn default() -> Self {
        Self {
            config: ServiceConfig::default(),
            listening: false,
            installing: false,
            loading_model: false,
            model_loaded: false,
            session_id: 0,
            seq: 0,
            partial_text: String::new(),
            last_error: String::new(),
            worker: None,
        }
    }
}

struct WorkerProcess {
    stdin: Arc<Mutex<ChildStdin>>,
    child: Child,
}

#[derive(Clone, Default)]
struct WordpipeService {
    data: Arc<Mutex<ServiceData>>,
    emitter: Arc<Mutex<Option<SignalEmitter<'static>>>>,
    config_path: PathBuf,
}

impl WordpipeService {
    fn new(config_path: PathBuf, config: ServiceConfig) -> Self {
        Self {
            data: Arc::new(Mutex::new(ServiceData {
                config,
                ..ServiceData::default()
            })),
            emitter: Arc::default(),
            config_path,
        }
    }
}

#[interface(interface = "dev.wordpipe.Service1")]
impl WordpipeService {
    async fn start(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        self.ensure_worker(emitter.to_owned()).await?;
        let (state, session_id, stdin) = {
            let mut data = self.lock_data()?;
            if !data.listening {
                data.listening = true;
                data.session_id = next_session_id(data.session_id);
                data.seq = 0;
                data.partial_text.clear();
            }
            let stdin = data
                .worker
                .as_ref()
                .map(|worker| Arc::clone(&worker.stdin))
                .ok_or_else(|| zbus::fdo::Error::Failed("ASR worker is not running".to_string()))?;
            (state_map(&data), data.session_id, stdin)
        };
        send_worker_command(&stdin, "start").map_err(fdo_failed)?;
        Self::session_started(&emitter, session_id).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn stop(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (state, stopped_session, stdin) = {
            let mut data = self.lock_data()?;
            let stopped_session = data.session_id;
            data.listening = false;
            data.seq = data.seq.saturating_add(1);
            let stdin = data.worker.as_ref().map(|worker| Arc::clone(&worker.stdin));
            (state_map(&data), stopped_session, stdin)
        };
        if let Some(stdin) = stdin {
            send_worker_command(&stdin, "stop").map_err(fdo_failed)?;
        }
        Self::session_stopped(&emitter, stopped_session).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn toggle(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        if self.lock_data()?.listening {
            self.stop(emitter).await
        } else {
            self.start(emitter).await
        }
    }

    fn get_state(&self) -> zbus::fdo::Result<VariantMap> {
        let data = self.lock_data()?;
        Ok(state_map(&data))
    }

    fn get_config(&self) -> zbus::fdo::Result<VariantMap> {
        let data = self.lock_data()?;
        Ok(config_map(&data.config))
    }

    fn list_backends(&self) -> Vec<VariantMap> {
        BACKENDS
            .iter()
            .map(|backend| {
                let mut item = VariantMap::new();
                insert_str(&mut item, "id", backend.id);
                insert_str(&mut item, "title", backend.title);
                insert_str(&mut item, "description", backend.description);
                item
            })
            .collect()
    }

    fn list_model_profiles(&self) -> Vec<VariantMap> {
        let model_root = self
            .data
            .lock()
            .map(|data| data.config.model_root.clone())
            .unwrap_or_default();
        MODEL_PROFILES
            .iter()
            .map(|profile| {
                let runtime_dir =
                    profile_runtime_dir(&model_root, profile.output_name, profile.ort_format);
                let mut item = VariantMap::new();
                insert_str(&mut item, "id", profile.id);
                insert_str(&mut item, "title", profile.title);
                insert_str(&mut item, "description", profile.description);
                insert_str(&mut item, "build_profile", profile.build_profile);
                insert_str(&mut item, "output_name", profile.output_name);
                insert_str(&mut item, "prebuilt_filename", profile.prebuilt_filename);
                insert_bool(&mut item, "ort_format", profile.ort_format);
                insert_str(&mut item, "runtime_dir", &runtime_dir);
                insert_bool(&mut item, "installed", profile_installed(&runtime_dir));
                item
            })
            .collect()
    }

    fn list_input_devices(&self) -> zbus::fdo::Result<Vec<VariantMap>> {
        enumerate_input_devices().map_err(fdo_failed)
    }

    async fn set_backend(
        &self,
        backend: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        if !is_backend(backend) {
            return Err(zbus::fdo::Error::InvalidArgs(format!(
                "unknown backend: {backend}"
            )));
        }
        let (config_data, config) = {
            let mut data = self.lock_data()?;
            data.config.backend = backend.to_string();
            shutdown_worker(&mut data);
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            (config_data, config)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn set_model_profile(
        &self,
        profile: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        if !is_model_profile(profile) {
            return Err(zbus::fdo::Error::InvalidArgs(format!(
                "unknown model profile: {profile}"
            )));
        }
        let (config_data, config) = {
            let mut data = self.lock_data()?;
            if data.config.model_profile != profile {
                data.config.model_profile = profile.to_string();
                shutdown_worker(&mut data);
            }
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            (config_data, config)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn set_input_device(
        &self,
        selector: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (config_data, config) = {
            let mut data = self.lock_data()?;
            if data.config.input_device != selector {
                data.config.input_device = selector.to_string();
                shutdown_worker(&mut data);
            }
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            (config_data, config)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn set_shortcut(
        &self,
        accelerator: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (config_data, config) = {
            let mut data = self.lock_data()?;
            data.config.shortcut = accelerator.to_string();
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            (config_data, config)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn set_insertion_options(
        &self,
        options: VariantMap,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (config_data, config) = {
            let mut data = self.lock_data()?;
            if let Some(value) = get_bool(&options, "spoken_punctuation") {
                data.config.spoken_punctuation = value;
            }
            if let Some(value) = get_bool(&options, "insert_partials") {
                data.config.insert_partials = value;
            }
            if let Some(value) = get_u32(&options, "stream_insert_delay_ms") {
                data.config.stream_insert_delay_ms = value;
            }
            if let Some(value) = get_bool(&options, "show_overlay") {
                data.config.show_overlay = value;
            }
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            (config_data, config)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn set_runtime_options(
        &self,
        options: VariantMap,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (config_data, config) = {
            let mut data = self.lock_data()?;
            let mut restart_worker = false;
            if let Some(value) = get_string(&options, "model_root") {
                restart_worker |= data.config.model_root != value;
                data.config.model_root = value;
            }
            if let Some(value) = get_string(&options, "worker_path") {
                restart_worker |= data.config.worker_path != value;
                data.config.worker_path = value;
            }
            if let Some(value) = get_string(&options, "model_installer_path") {
                data.config.model_installer_path = value;
            }
            if let Some(value) = get_u32(&options, "sample_rate") {
                if value == 0 {
                    return Err(zbus::fdo::Error::InvalidArgs(
                        "sample_rate must be positive".to_string(),
                    ));
                }
                restart_worker |= data.config.sample_rate != value;
                data.config.sample_rate = value;
            }
            if let Some(value) = get_u32(&options, "num_threads") {
                if value == 0 {
                    return Err(zbus::fdo::Error::InvalidArgs(
                        "num_threads must be positive".to_string(),
                    ));
                }
                restart_worker |= data.config.num_threads != value;
                data.config.num_threads = value;
            }
            if restart_worker {
                shutdown_worker(&mut data);
            }
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            (config_data, config)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn install_model(
        &self,
        profile: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        if !is_model_profile(profile) {
            return Err(zbus::fdo::Error::InvalidArgs(format!(
                "unknown model profile: {profile}"
            )));
        }
        let (installer_path, model_root) = {
            let mut data = self.lock_data()?;
            if data.installing {
                return Err(zbus::fdo::Error::Failed(
                    "model installation is already running".to_string(),
                ));
            }
            data.installing = true;
            (
                data.config.model_installer_path.clone(),
                data.config.model_root.clone(),
            )
        };
        let mut progress = VariantMap::new();
        insert_str(&mut progress, "phase", "starting");
        insert_str(&mut progress, "message", "starting model installer");
        insert_f64(&mut progress, "fraction", 0.0);
        Self::install_progress(&emitter, profile, progress).await?;

        let service = self.clone();
        let profile = profile.to_string();
        let emitter = emitter.to_owned();
        std::thread::Builder::new()
            .name("wordpipe-model-install".to_string())
            .spawn(move || {
                service.run_model_installer(installer_path, model_root, profile, emitter);
            })
            .map_err(|err| zbus::fdo::Error::Failed(err.to_string()))?;
        Ok(())
    }

    async fn shutdown(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let state = {
            let mut data = self.lock_data()?;
            data.listening = false;
            shutdown_worker(&mut data);
            state_map(&data)
        };
        Self::state_changed(&emitter, state).await?;
        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            std::process::exit(0);
        });
        Ok(())
    }

    #[zbus(signal)]
    async fn state_changed(emitter: &SignalEmitter<'_>, state: VariantMap) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn config_changed(emitter: &SignalEmitter<'_>, config: VariantMap) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn session_started(emitter: &SignalEmitter<'_>, session_id: u64) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn text_delta(
        emitter: &SignalEmitter<'_>,
        session_id: u64,
        seq: u64,
        text: &str,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn partial(
        emitter: &SignalEmitter<'_>,
        session_id: u64,
        seq: u64,
        full_text: &str,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn commit(
        emitter: &SignalEmitter<'_>,
        session_id: u64,
        seq: u64,
        text: &str,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn session_stopped(emitter: &SignalEmitter<'_>, session_id: u64) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn install_progress(
        emitter: &SignalEmitter<'_>,
        profile: &str,
        progress: VariantMap,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn metrics(emitter: &SignalEmitter<'_>, metrics: VariantMap) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn error(emitter: &SignalEmitter<'_>, message: &str) -> zbus::Result<()>;
}

impl WordpipeService {
    fn lock_data(&self) -> zbus::fdo::Result<std::sync::MutexGuard<'_, ServiceData>> {
        self.data
            .lock()
            .map_err(|_| zbus::fdo::Error::Failed("service state lock poisoned".to_string()))
    }

    fn persist_config(&self, config: &ServiceConfig) -> zbus::fdo::Result<()> {
        save_service_config(&self.config_path, config).map_err(fdo_failed)
    }

    async fn ensure_worker(&self, emitter: SignalEmitter<'static>) -> zbus::fdo::Result<()> {
        let spawn = {
            let data = self.lock_data()?;
            data.worker.is_none()
        };
        if !spawn {
            return Ok(());
        }

        let (worker, stdout) = {
            let mut data = self.lock_data()?;
            let config = data.config.clone();
            let runtime_dir = selected_runtime_dir(&config);
            if !profile_installed(&runtime_dir) {
                return Err(zbus::fdo::Error::Failed(format!(
                    "model profile '{}' is not installed at {runtime_dir}",
                    config.model_profile
                )));
            }
            data.loading_model = true;
            data.model_loaded = false;
            data.last_error.clear();
            spawn_worker(&config, &runtime_dir).map_err(fdo_failed)?
        };

        {
            let mut data = self.lock_data()?;
            data.worker = Some(worker);
        }

        let service = self.clone();
        std::thread::Builder::new()
            .name("wordpipe-asr-events".to_string())
            .spawn(move || service.read_worker_events(stdout, emitter))
            .map_err(|err| zbus::fdo::Error::Failed(err.to_string()))?;
        Ok(())
    }

    fn read_worker_events(
        &self,
        stdout: std::process::ChildStdout,
        emitter: SignalEmitter<'static>,
    ) {
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<JsonValue>(&line) {
                Ok(value) => self.handle_worker_event(value, &emitter),
                Err(err) => {
                    let _ = zbus::block_on(Self::error(
                        &emitter,
                        &format!("invalid ASR worker event: {err}"),
                    ));
                }
            }
        }
        let state = {
            let mut data = match self.data.lock() {
                Ok(data) => data,
                Err(_) => return,
            };
            data.listening = false;
            data.loading_model = false;
            data.model_loaded = false;
            data.worker = None;
            state_map(&data)
        };
        let _ = zbus::block_on(Self::state_changed(&emitter, state));
    }

    fn handle_worker_event(&self, value: JsonValue, emitter: &SignalEmitter<'static>) {
        let event = value
            .get("event")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        match event {
            "loading_model" => {
                let state = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.loading_model = true;
                    data.model_loaded = false;
                    state_map(&data)
                };
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            "model_loaded" => {
                let (state, metrics) = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.loading_model = false;
                    data.model_loaded = true;
                    let metrics = value
                        .get("data")
                        .map(json_to_variant_map)
                        .unwrap_or_default();
                    (state_map(&data), metrics)
                };
                let _ = zbus::block_on(Self::state_changed(emitter, state));
                let _ = zbus::block_on(Self::metrics(emitter, metrics));
            }
            "ready" => {
                let state = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.loading_model = false;
                    data.model_loaded = true;
                    state_map(&data)
                };
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            "listening" => {
                let state = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.listening = true;
                    state_map(&data)
                };
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            "partial" => self.forward_partial(value, emitter),
            "commit" => self.forward_commit(value, emitter),
            "stats" => {
                let metrics = value
                    .get("data")
                    .map(json_to_variant_map)
                    .unwrap_or_default();
                let _ = zbus::block_on(Self::metrics(emitter, metrics));
            }
            "stopped" => {
                let (state, session_id) = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.listening = false;
                    (state_map(&data), data.session_id)
                };
                let _ = zbus::block_on(Self::session_stopped(emitter, session_id));
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            "error" => {
                let message = value
                    .get("message")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("ASR worker error")
                    .to_string();
                let state = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.last_error = message.clone();
                    state_map(&data)
                };
                let _ = zbus::block_on(Self::error(emitter, &message));
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            _ => {}
        }
    }

    fn forward_partial(&self, value: JsonValue, emitter: &SignalEmitter<'static>) {
        let text = value
            .get("text")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string();
        let metrics = value
            .get("data")
            .map(json_to_variant_map)
            .unwrap_or_default();
        let (session_id, seq, delta, text) = {
            let mut data = match self.data.lock() {
                Ok(data) => data,
                Err(_) => return,
            };
            let text = if data.config.spoken_punctuation {
                normalize_spoken_punctuation_partial(&text)
            } else {
                text
            };
            data.seq = data.seq.saturating_add(1);
            let delta = text
                .strip_prefix(&data.partial_text)
                .unwrap_or(&text)
                .to_string();
            data.partial_text = text.clone();
            (data.session_id, data.seq, delta, text)
        };
        let _ = zbus::block_on(Self::partial(emitter, session_id, seq, &text));
        if !delta.is_empty() {
            let _ = zbus::block_on(Self::text_delta(emitter, session_id, seq, &delta));
        }
        let _ = zbus::block_on(Self::metrics(emitter, metrics));
    }

    fn forward_commit(&self, value: JsonValue, emitter: &SignalEmitter<'static>) {
        let text = value
            .get("text")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string();
        let metrics = value
            .get("data")
            .map(json_to_variant_map)
            .unwrap_or_default();
        let (session_id, seq, text) = {
            let mut data = match self.data.lock() {
                Ok(data) => data,
                Err(_) => return,
            };
            let text = if data.config.spoken_punctuation {
                normalize_spoken_punctuation(&text)
            } else {
                text
            };
            data.seq = data.seq.saturating_add(1);
            (data.session_id, data.seq, text)
        };
        if !text.is_empty() {
            let _ = zbus::block_on(Self::commit(emitter, session_id, seq, &text));
        }
        let _ = zbus::block_on(Self::metrics(emitter, metrics));
    }

    fn run_model_installer(
        &self,
        installer_path: String,
        model_root: String,
        profile: String,
        emitter: SignalEmitter<'static>,
    ) {
        let mut command = Command::new(&installer_path);
        command
            .arg("--profile")
            .arg(&profile)
            .arg("--model-root")
            .arg(&model_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = run_progress_command(command, &profile, &emitter);
        let state = {
            let mut data = match self.data.lock() {
                Ok(data) => data,
                Err(_) => return,
            };
            data.installing = false;
            match result {
                Ok(()) => {
                    data.last_error.clear();
                    let mut progress = VariantMap::new();
                    insert_str(&mut progress, "phase", "complete");
                    insert_str(&mut progress, "message", "model profile installed");
                    insert_f64(&mut progress, "fraction", 1.0);
                    let _ = zbus::block_on(Self::install_progress(&emitter, &profile, progress));
                }
                Err(err) => {
                    data.last_error = err.to_string();
                    let mut progress = VariantMap::new();
                    insert_str(&mut progress, "phase", "error");
                    insert_str(&mut progress, "message", &err.to_string());
                    insert_f64(&mut progress, "fraction", 0.0);
                    let _ = zbus::block_on(Self::install_progress(&emitter, &profile, progress));
                    let _ = zbus::block_on(Self::error(&emitter, &err.to_string()));
                }
            }
            state_map(&data)
        };
        let _ = zbus::block_on(Self::state_changed(&emitter, state));
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    zbus::block_on(run(args))
}

async fn run(args: Args) -> Result<()> {
    if args.replace {
        eprintln!("wordpipe-service: --replace requested; D-Bus will replace an existing name if the bus allows it");
    }
    let config_path = args.config.unwrap_or_else(default_config_path);
    let config = load_service_config(&config_path)
        .with_context(|| format!("failed to load service config {}", config_path.display()))?;
    let service = WordpipeService::new(config_path, config);
    let service_handle = service.clone();
    let _connection = connection::Builder::session()?
        .serve_at(OBJECT_PATH, service)?
        .name(BUS_NAME)?
        .build()
        .await
        .with_context(|| format!("failed to own D-Bus name {BUS_NAME}"))?;
    let emitter = SignalEmitter::new(&_connection, OBJECT_PATH)?.to_owned();
    *service_handle
        .emitter
        .lock()
        .expect("service emitter lock should not be poisoned") = Some(emitter);
    eprintln!("wordpipe-service: listening on {BUS_NAME} {OBJECT_PATH}");
    std::future::pending::<()>().await;
    Ok(())
}

fn state_map(data: &ServiceData) -> VariantMap {
    let mut map = VariantMap::new();
    insert_bool(&mut map, "listening", data.listening);
    insert_bool(&mut map, "installing", data.installing);
    insert_bool(&mut map, "loading_model", data.loading_model);
    insert_bool(&mut map, "model_loaded", data.model_loaded);
    insert_u64(&mut map, "session_id", data.session_id);
    insert_u64(&mut map, "seq", data.seq);
    insert_str(&mut map, "backend", &data.config.backend);
    insert_str(&mut map, "model_profile", &data.config.model_profile);
    insert_str(&mut map, "input_device", &data.config.input_device);
    insert_str(&mut map, "last_error", &data.last_error);
    map
}

fn config_map(config: &ServiceConfig) -> VariantMap {
    let mut map = VariantMap::new();
    insert_str(&mut map, "backend", &config.backend);
    insert_str(&mut map, "model_profile", &config.model_profile);
    insert_str(&mut map, "input_device", &config.input_device);
    insert_str(&mut map, "shortcut", &config.shortcut);
    insert_str(&mut map, "model_root", &config.model_root);
    insert_str(&mut map, "worker_path", &config.worker_path);
    insert_str(
        &mut map,
        "model_installer_path",
        &config.model_installer_path,
    );
    insert_u32(&mut map, "sample_rate", config.sample_rate);
    insert_u32(&mut map, "num_threads", config.num_threads);
    insert_bool(&mut map, "spoken_punctuation", config.spoken_punctuation);
    insert_bool(&mut map, "insert_partials", config.insert_partials);
    insert_u32(
        &mut map,
        "stream_insert_delay_ms",
        config.stream_insert_delay_ms,
    );
    insert_bool(&mut map, "show_overlay", config.show_overlay);
    map
}

fn default_config_path() -> PathBuf {
    if let Some(value) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(value).join("wordpipe").join("service.json")
    } else if let Some(value) = std::env::var_os("HOME") {
        PathBuf::from(value)
            .join(".config")
            .join("wordpipe")
            .join("service.json")
    } else {
        PathBuf::from("wordpipe-service.json")
    }
}

fn load_service_config(path: &Path) -> Result<ServiceConfig> {
    if !path.exists() {
        return Ok(ServiceConfig::default());
    }
    let persisted: PersistedConfig = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    apply_persisted_config(ServiceConfig::default(), persisted)
}

fn apply_persisted_config(
    mut config: ServiceConfig,
    persisted: PersistedConfig,
) -> Result<ServiceConfig> {
    if let Some(value) = persisted.backend {
        if !is_backend(&value) {
            return Err(anyhow!("unknown backend in service config: {value}"));
        }
        config.backend = value;
    }
    if let Some(value) = persisted.model_profile {
        if !is_model_profile(&value) {
            return Err(anyhow!("unknown model profile in service config: {value}"));
        }
        config.model_profile = value;
    }
    if let Some(value) = persisted.input_device {
        config.input_device = value;
    }
    if let Some(value) = persisted.shortcut {
        config.shortcut = value;
    }
    if let Some(value) = persisted.model_root {
        config.model_root = value;
    }
    if let Some(value) = persisted.worker_path {
        config.worker_path = value;
    }
    if let Some(value) = persisted.model_installer_path {
        config.model_installer_path = value;
    }
    if let Some(value) = persisted.sample_rate {
        if value == 0 {
            return Err(anyhow!("sample_rate must be positive"));
        }
        config.sample_rate = value;
    }
    if let Some(value) = persisted.num_threads {
        if value == 0 {
            return Err(anyhow!("num_threads must be positive"));
        }
        config.num_threads = value;
    }
    if let Some(value) = persisted.spoken_punctuation {
        config.spoken_punctuation = value;
    }
    if let Some(value) = persisted.insert_partials {
        config.insert_partials = value;
    }
    if let Some(value) = persisted.stream_insert_delay_ms {
        config.stream_insert_delay_ms = value;
    }
    if let Some(value) = persisted.show_overlay {
        config.show_overlay = value;
    }
    Ok(config)
}

fn save_service_config(path: &Path, config: &ServiceConfig) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("config path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    let temporary = path.with_file_name(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("service.json"),
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(&PersistedConfig::from(config))?;
    fs::write(&temporary, bytes)
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| {
        format!(
            "failed to replace {} with {}",
            path.display(),
            temporary.display()
        )
    })?;
    Ok(())
}

fn enumerate_input_devices() -> Result<Vec<VariantMap>> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let mut devices = Vec::new();
    for (index, device) in host
        .input_devices()
        .context("failed to enumerate input devices")?
        .enumerate()
    {
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        let mut item = VariantMap::new();
        insert_u32(&mut item, "index", index as u32);
        insert_str(&mut item, "name", &name);
        insert_str(&mut item, "selector", &name);
        insert_bool(
            &mut item,
            "is_default",
            default_name.as_ref() == Some(&name),
        );
        devices.push(item);
    }
    Ok(devices)
}

fn default_model_root() -> String {
    if let Some(value) = std::env::var_os("XDG_DATA_HOME") {
        format!("{}/wordpipe/models", value.to_string_lossy())
    } else if let Some(value) = std::env::var_os("HOME") {
        format!("{}/.local/share/wordpipe/models", value.to_string_lossy())
    } else {
        "wordpipe/models".to_string()
    }
}

fn profile_runtime_dir(model_root: &str, output_name: &str, ort_format: bool) -> String {
    if ort_format {
        format!("{model_root}/{output_name}-ort-format")
    } else {
        format!("{model_root}/{output_name}")
    }
}

fn selected_runtime_dir(config: &ServiceConfig) -> String {
    let Some(profile) = MODEL_PROFILES
        .iter()
        .find(|profile| profile.id == config.model_profile)
    else {
        return config.model_root.clone();
    };
    profile_runtime_dir(&config.model_root, profile.output_name, profile.ort_format)
}

fn profile_installed(runtime_dir: &str) -> bool {
    let path = std::path::Path::new(runtime_dir);
    path.join("encoder.onnx").exists()
        || path.join("encoder.ort").exists()
        || path.join("encoder.encoder.onnx").exists()
        || path.join("encoder.encoder.ort").exists()
}

fn next_session_id(current: u64) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_micros() as u64)
        .unwrap_or(0);
    now.max(current.saturating_add(1))
}

fn insert_str(map: &mut VariantMap, key: &str, value: &str) {
    map.insert(key.to_string(), owned(Value::from(value.to_string())));
}

fn insert_bool(map: &mut VariantMap, key: &str, value: bool) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_u32(map: &mut VariantMap, key: &str, value: u32) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_u64(map: &mut VariantMap, key: &str, value: u64) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_f64(map: &mut VariantMap, key: &str, value: f64) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_i64(map: &mut VariantMap, key: &str, value: i64) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn owned(value: Value<'_>) -> OwnedValue {
    OwnedValue::try_from(value).expect("primitive D-Bus variant value should be valid")
}

fn get_bool(map: &VariantMap, key: &str) -> Option<bool> {
    map.get(key)
        .and_then(|value| bool::try_from(value.clone()).ok())
}

fn get_u32(map: &VariantMap, key: &str) -> Option<u32> {
    map.get(key)
        .and_then(|value| u32::try_from(value.clone()).ok())
}

fn get_string(map: &VariantMap, key: &str) -> Option<String> {
    map.get(key)
        .and_then(|value| String::try_from(value.clone()).ok())
}

fn fdo_failed(err: anyhow::Error) -> zbus::fdo::Error {
    zbus::fdo::Error::Failed(err.to_string())
}

fn default_worker_path() -> String {
    if let Ok(value) = std::env::var("WORDPIPE_WORKER") {
        return value;
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            let sibling = parent.join("wordpipe-parakeet-worker");
            if sibling.exists() {
                return sibling.to_string_lossy().to_string();
            }
        }
    }
    "wordpipe-parakeet-worker".to_string()
}

fn default_model_installer_path() -> String {
    if let Ok(value) = std::env::var("WORDPIPE_MODEL_INSTALLER") {
        return value;
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            let sibling = parent.join("wordpipe-model-install");
            if sibling.exists() {
                return sibling.to_string_lossy().to_string();
            }
        }
    }
    "wordpipe-model-install".to_string()
}

fn default_ort_dylib_path() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("ORT_DYLIB_PATH") {
        return Some(PathBuf::from(value));
    }

    for path in [
        PathBuf::from("/app/lib/libonnxruntime.so"),
        PathBuf::from("/app/lib/onnxruntime/libonnxruntime.so"),
    ] {
        if path.exists() {
            return Some(path);
        }
    }

    for root in candidate_repo_roots() {
        for venv in [".venv", ".venv-nemo-export"] {
            if let Some(path) = find_ort_in_venv(&root.join(venv)) {
                return Some(path);
            }
        }
    }
    None
}

fn candidate_repo_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(current_dir) = std::env::current_dir() {
        roots.push(current_dir);
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(target_dir) = current_exe.parent().and_then(Path::parent) {
            if target_dir.file_name().and_then(|value| value.to_str()) == Some("target") {
                if let Some(repo) = target_dir.parent() {
                    roots.push(repo.to_path_buf());
                }
            }
        }
    }
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        let crate_dir = PathBuf::from(manifest_dir);
        if let Some(repo) = crate_dir.parent().and_then(Path::parent) {
            roots.push(repo.to_path_buf());
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

fn find_ort_in_venv(venv: &Path) -> Option<PathBuf> {
    let python_dirs = std::fs::read_dir(venv.join("lib")).ok()?;
    for python_dir in python_dirs.flatten() {
        let capi_dir = python_dir
            .path()
            .join("site-packages")
            .join("onnxruntime")
            .join("capi");
        let files = std::fs::read_dir(capi_dir).ok()?;
        for file in files.flatten() {
            let path = file.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with("libonnxruntime.so") {
                return Some(path);
            }
        }
    }
    None
}

fn spawn_worker(
    config: &ServiceConfig,
    runtime_dir: &str,
) -> Result<(WorkerProcess, std::process::ChildStdout)> {
    let mut command = Command::new(&config.worker_path);
    command
        .arg("--model-dir")
        .arg(runtime_dir)
        .arg("--num-threads")
        .arg(config.num_threads.to_string())
        .arg("--sample-rate")
        .arg(config.sample_rate.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if !config.input_device.is_empty() {
        command.arg("--input-device").arg(&config.input_device);
    }
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Some(path) = default_ort_dylib_path() {
            command.env("ORT_DYLIB_PATH", path);
        }
    }
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start ASR worker '{}'; build/install wordpipe-parakeet-worker or set WORDPIPE_WORKER",
            config.worker_path
        )
    })?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("ASR worker stdin was not piped"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("ASR worker stdout was not piped"))?;
    Ok((
        WorkerProcess {
            stdin: Arc::new(Mutex::new(stdin)),
            child,
        },
        stdout,
    ))
}

fn send_worker_command(stdin: &Arc<Mutex<ChildStdin>>, command: &str) -> Result<()> {
    let mut stdin = stdin
        .lock()
        .map_err(|_| anyhow!("ASR worker stdin lock poisoned"))?;
    serde_json::to_writer(&mut *stdin, &json!({ "command": command }))?;
    writeln!(stdin)?;
    stdin.flush()?;
    Ok(())
}

fn shutdown_worker(data: &mut ServiceData) {
    if let Some(mut worker) = data.worker.take() {
        let _ = send_worker_command(&worker.stdin, "shutdown");
        let _ = worker.child.kill();
        let _ = worker.child.wait();
    }
    data.listening = false;
    data.loading_model = false;
    data.model_loaded = false;
    data.partial_text.clear();
}

fn normalize_spoken_punctuation(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    normalize_spoken_words(&words)
}

fn normalize_spoken_punctuation_partial(text: &str) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    if words
        .last()
        .is_some_and(|word| is_incomplete_command_prefix(word))
    {
        words.pop();
    }
    normalize_spoken_words(&words)
}

fn normalize_spoken_words(words: &[&str]) -> String {
    let mut output = String::new();
    let mut index = 0;
    while index < words.len() {
        if let Some((command, consumed)) = match_spoken_command(&words, index) {
            if matches!(command, "," | "." | "?" | "!" | ":" | ";") {
                while output.ends_with(' ') {
                    output.pop();
                }
                output.push_str(command);
                output.push(' ');
            } else {
                while output.ends_with(' ') {
                    output.pop();
                }
                output.push_str(command);
            }
            index += consumed;
            continue;
        }
        append_spoken_word(&mut output, words[index]);
        index += 1;
    }
    output.trim_end_matches(' ').to_string()
}

fn is_incomplete_command_prefix(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "full" | "question" | "exclamation" | "new"
    )
}

fn match_spoken_command(words: &[&str], index: usize) -> Option<(&'static str, usize)> {
    let remaining = words.len().saturating_sub(index);
    for size in (1..=remaining.min(2)).rev() {
        let first = words[index].to_ascii_lowercase();
        let second = if size == 2 {
            Some(words[index + 1].to_ascii_lowercase())
        } else {
            None
        };
        let command = match (first.as_str(), second.as_deref()) {
            ("comma", None) => Some(","),
            ("period", None) => Some("."),
            ("full", Some("stop")) => Some("."),
            ("question", Some("mark")) => Some("?"),
            ("exclamation", Some("point")) => Some("!"),
            ("exclamation", Some("mark")) => Some("!"),
            ("colon", None) => Some(":"),
            ("semicolon", None) => Some(";"),
            ("new", Some("line")) => Some("\n"),
            ("newline", None) => Some("\n"),
            ("new", Some("paragraph")) => Some("\n\n"),
            _ => None,
        };
        if let Some(command) = command {
            return Some((command, size));
        }
    }
    None
}

fn append_spoken_word(output: &mut String, word: &str) {
    if output.is_empty() || output.ends_with('\n') || output.ends_with(' ') {
        output.push_str(word);
        output.push(' ');
    } else {
        output.push(' ');
        output.push_str(word);
        output.push(' ');
    }
}

fn run_progress_command(
    mut command: Command,
    profile: &str,
    emitter: &SignalEmitter<'static>,
) -> Result<()> {
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start model installer {:?}", command))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("model installer stdout was not piped"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("model installer stderr was not piped"))?;

    let stdout_profile = profile.to_string();
    let stdout_emitter = emitter.clone();
    let stdout_thread =
        std::thread::spawn(move || stream_progress_lines(stdout, &stdout_profile, &stdout_emitter));

    let stderr_profile = profile.to_string();
    let stderr_emitter = emitter.clone();
    let stderr_thread =
        std::thread::spawn(move || stream_progress_lines(stderr, &stderr_profile, &stderr_emitter));

    let status = child.wait()?;
    let _ = stdout_thread.join();
    let _ = stderr_thread.join();
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("model installer exited with {status}"))
    }
}

fn stream_progress_lines<R>(reader: R, profile: &str, emitter: &SignalEmitter<'static>)
where
    R: std::io::Read,
{
    let reader = std::io::BufReader::new(reader);
    for line in reader.lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let mut progress = VariantMap::new();
        insert_str(&mut progress, "phase", "running");
        insert_str(&mut progress, "message", line.trim());
        insert_f64(&mut progress, "fraction", 0.0);
        let _ = zbus::block_on(WordpipeService::install_progress(
            emitter, profile, progress,
        ));
    }
}

fn json_to_variant_map(value: &JsonValue) -> VariantMap {
    let mut map = VariantMap::new();
    let Some(object) = value.as_object() else {
        return map;
    };
    for (key, value) in object {
        match value {
            JsonValue::Bool(value) => insert_bool(&mut map, key, *value),
            JsonValue::Number(number) => {
                if let Some(value) = number.as_u64() {
                    insert_u64(&mut map, key, value);
                } else if let Some(value) = number.as_i64() {
                    insert_i64(&mut map, key, value);
                } else if let Some(value) = number.as_f64() {
                    insert_f64(&mut map, key, value);
                }
            }
            JsonValue::String(value) => insert_str(&mut map, key, value),
            _ => {}
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_config_overrides_defaults() {
        let config = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                model_profile: Some("compact".to_string()),
                input_device: Some("pipewire".to_string()),
                num_threads: Some(4),
                sample_rate: Some(16_000),
                show_overlay: Some(false),
                ..PersistedConfig::default()
            },
        )
        .unwrap();

        assert_eq!(config.model_profile, "compact");
        assert_eq!(config.input_device, "pipewire");
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.sample_rate, 16_000);
        assert!(!config.show_overlay);
        assert_eq!(config.backend, DEFAULT_BACKEND);
    }

    #[test]
    fn persisted_config_rejects_invalid_profile() {
        let err = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                model_profile: Some("tiny".to_string()),
                ..PersistedConfig::default()
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("unknown model profile"));
    }

    #[test]
    fn normalizes_spoken_punctuation_commands() {
        assert_eq!(
            normalize_spoken_punctuation(
                "hello comma world period new line second paragraph question mark"
            ),
            "hello, world.\nsecond paragraph?"
        );
        assert_eq!(
            normalize_spoken_punctuation("wait full stop no exclamation point"),
            "wait. no!"
        );
    }

    #[test]
    fn normalizes_spoken_punctuation_without_touching_unknown_words() {
        assert_eq!(
            normalize_spoken_punctuation("new deal comma newline done"),
            "new deal,\ndone"
        );
    }

    #[test]
    fn partial_normalization_holds_incomplete_commands() {
        assert_eq!(normalize_spoken_punctuation_partial("new"), "");
        assert_eq!(
            normalize_spoken_punctuation_partial("hello question"),
            "hello"
        );
        assert_eq!(
            normalize_spoken_punctuation_partial("hello question mark"),
            "hello?"
        );
        assert_eq!(normalize_spoken_punctuation_partial("new deal"), "new deal");
    }
}
