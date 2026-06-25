use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};
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

#[derive(Clone, Debug)]
struct ServiceConfig {
    backend: String,
    model_profile: String,
    input_device: String,
    shortcut: String,
    model_root: String,
    sample_rate: u32,
    num_threads: u32,
    spoken_punctuation: bool,
    insert_partials: bool,
    stream_insert_delay_ms: u32,
    show_overlay: bool,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            backend: DEFAULT_BACKEND.to_string(),
            model_profile: DEFAULT_MODEL_PROFILE.to_string(),
            input_device: String::new(),
            shortcut: DEFAULT_SHORTCUT.to_string(),
            model_root: default_model_root(),
            sample_rate: DEFAULT_SAMPLE_RATE,
            num_threads: DEFAULT_NUM_THREADS,
            spoken_punctuation: true,
            insert_partials: true,
            stream_insert_delay_ms: 0,
            show_overlay: true,
        }
    }
}

#[derive(Clone, Debug)]
struct ServiceData {
    config: ServiceConfig,
    listening: bool,
    installing: bool,
    session_id: u64,
    seq: u64,
    last_error: String,
}

impl Default for ServiceData {
    fn default() -> Self {
        Self {
            config: ServiceConfig::default(),
            listening: false,
            installing: false,
            session_id: 0,
            seq: 0,
            last_error: String::new(),
        }
    }
}

#[derive(Clone, Default)]
struct WordpipeService {
    data: Arc<Mutex<ServiceData>>,
}

#[interface(interface = "dev.wordpipe.Service1")]
impl WordpipeService {
    async fn start(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (state, session_id) = {
            let mut data = self.lock_data()?;
            if !data.listening {
                data.listening = true;
                data.session_id = next_session_id(data.session_id);
                data.seq = 0;
            }
            (state_map(&data), data.session_id)
        };
        Self::session_started(&emitter, session_id).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn stop(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (state, stopped_session) = {
            let mut data = self.lock_data()?;
            let stopped_session = data.session_id;
            data.listening = false;
            data.seq = data.seq.saturating_add(1);
            (state_map(&data), stopped_session)
        };
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
        let config = {
            let mut data = self.lock_data()?;
            data.config.backend = backend.to_string();
            config_map(&data.config)
        };
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
        let config = {
            let mut data = self.lock_data()?;
            data.config.model_profile = profile.to_string();
            config_map(&data.config)
        };
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn set_input_device(
        &self,
        selector: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let config = {
            let mut data = self.lock_data()?;
            data.config.input_device = selector.to_string();
            config_map(&data.config)
        };
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn set_shortcut(
        &self,
        accelerator: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let config = {
            let mut data = self.lock_data()?;
            data.config.shortcut = accelerator.to_string();
            config_map(&data.config)
        };
        Self::config_changed(&emitter, config).await?;
        Ok(())
    }

    async fn set_insertion_options(
        &self,
        options: VariantMap,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let config = {
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
            config_map(&data.config)
        };
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
        {
            let mut data = self.lock_data()?;
            data.installing = true;
        }
        let mut progress = VariantMap::new();
        insert_str(&mut progress, "phase", "queued");
        insert_str(
            &mut progress,
            "message",
            "model export is not yet wired into the Rust service on this branch",
        );
        insert_f64(&mut progress, "fraction", 0.0);
        Self::install_progress(&emitter, profile, progress).await?;

        let state = {
            let mut data = self.lock_data()?;
            data.installing = false;
            state_map(&data)
        };
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn shutdown(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let state = {
            let mut data = self.lock_data()?;
            data.listening = false;
            state_map(&data)
        };
        Self::state_changed(&emitter, state).await?;
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
}

fn main() -> Result<()> {
    let args = Args::parse();
    zbus::block_on(run(args))
}

async fn run(args: Args) -> Result<()> {
    if args.replace {
        eprintln!("wordpipe-service: --replace requested; D-Bus will replace an existing name if the bus allows it");
    }
    let service = WordpipeService::default();
    let _connection = connection::Builder::session()?
        .serve_at(OBJECT_PATH, service)?
        .name(BUS_NAME)?
        .build()
        .await
        .with_context(|| format!("failed to own D-Bus name {BUS_NAME}"))?;
    eprintln!("wordpipe-service: listening on {BUS_NAME} {OBJECT_PATH}");
    std::future::pending::<()>().await;
    Ok(())
}

fn state_map(data: &ServiceData) -> VariantMap {
    let mut map = VariantMap::new();
    insert_bool(&mut map, "listening", data.listening);
    insert_bool(&mut map, "installing", data.installing);
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

fn profile_installed(runtime_dir: &str) -> bool {
    let path = std::path::Path::new(runtime_dir);
    path.join("encoder.onnx").exists() || path.join("encoder.ort").exists()
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

fn fdo_failed(err: anyhow::Error) -> zbus::fdo::Error {
    zbus::fdo::Error::Failed(err.to_string())
}
