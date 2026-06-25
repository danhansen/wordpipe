pub const BUS_NAME: &str = "dev.wordpipe.Service";
pub const OBJECT_PATH: &str = "/dev/wordpipe/Service";
pub const INTERFACE_NAME: &str = "dev.wordpipe.Service1";

pub const DEFAULT_BACKEND: &str = "parakeet";
pub const DEFAULT_MODEL_PROFILE: &str = "fast";
pub const DEFAULT_SHORTCUT: &str = "<Control><Alt>space";
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_NUM_THREADS: u32 = 2;

pub const BACKENDS: &[BackendSpec] = &[BackendSpec {
    id: "parakeet",
    title: "Parakeet",
    description: "Rust parakeet-rs Nemotron streaming backend",
}];

pub const MODEL_PROFILES: &[ModelProfileSpec] = &[
    ModelProfileSpec {
        id: "fast",
        title: "Fast",
        description: "FP32 projected-cache model; fastest validated profile, largest footprint.",
        build_profile: "fp32-projected",
        output_name: "nemotron-wordpipe-fast-fp32-projected",
        prebuilt_filename: "wordpipe-nemotron-fast-fp32-projected.tar.gz",
        ort_format: false,
    },
    ModelProfileSpec {
        id: "compact",
        title: "Compact",
        description: "Dynamic-int8 projected-cache model with fixed shapes and ORT-format startup.",
        build_profile: "compact-fixed-shape",
        output_name: "nemotron-wordpipe-compact-fixed-shape",
        prebuilt_filename: "wordpipe-nemotron-compact-fixed-shape.tar.gz",
        ort_format: true,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendSpec {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelProfileSpec {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub build_profile: &'static str,
    pub output_name: &'static str,
    pub prebuilt_filename: &'static str,
    pub ort_format: bool,
}

pub fn is_backend(value: &str) -> bool {
    BACKENDS.iter().any(|backend| backend.id == value)
}

pub fn is_model_profile(value: &str) -> bool {
    MODEL_PROFILES.iter().any(|profile| profile.id == value)
}

pub const INTROSPECTION_XML: &str = r#"
<node>
  <interface name="dev.wordpipe.Service1">
    <method name="Start"/>
    <method name="Stop"/>
    <method name="Toggle"/>
    <method name="Shutdown"/>
    <method name="GetState">
      <arg name="state" type="a{sv}" direction="out"/>
    </method>
    <method name="GetConfig">
      <arg name="config" type="a{sv}" direction="out"/>
    </method>
    <method name="ListBackends">
      <arg name="backends" type="aa{sv}" direction="out"/>
    </method>
    <method name="ListModelProfiles">
      <arg name="profiles" type="aa{sv}" direction="out"/>
    </method>
    <method name="ListInputDevices">
      <arg name="devices" type="aa{sv}" direction="out"/>
    </method>
    <method name="SetBackend">
      <arg name="backend" type="s" direction="in"/>
    </method>
    <method name="SetModelProfile">
      <arg name="profile" type="s" direction="in"/>
    </method>
    <method name="SetInputDevice">
      <arg name="selector" type="s" direction="in"/>
    </method>
    <method name="SetShortcut">
      <arg name="accelerator" type="s" direction="in"/>
    </method>
    <method name="SetInsertionOptions">
      <arg name="options" type="a{sv}" direction="in"/>
    </method>
    <method name="InstallModel">
      <arg name="profile" type="s" direction="in"/>
    </method>
    <signal name="StateChanged">
      <arg name="state" type="a{sv}"/>
    </signal>
    <signal name="ConfigChanged">
      <arg name="config" type="a{sv}"/>
    </signal>
    <signal name="SessionStarted">
      <arg name="session_id" type="t"/>
    </signal>
    <signal name="TextDelta">
      <arg name="session_id" type="t"/>
      <arg name="seq" type="t"/>
      <arg name="text" type="s"/>
    </signal>
    <signal name="Partial">
      <arg name="session_id" type="t"/>
      <arg name="seq" type="t"/>
      <arg name="full_text" type="s"/>
    </signal>
    <signal name="Commit">
      <arg name="session_id" type="t"/>
      <arg name="seq" type="t"/>
      <arg name="text" type="s"/>
    </signal>
    <signal name="SessionStopped">
      <arg name="session_id" type="t"/>
    </signal>
    <signal name="InstallProgress">
      <arg name="profile" type="s"/>
      <arg name="progress" type="a{sv}"/>
    </signal>
    <signal name="Metrics">
      <arg name="metrics" type="a{sv}"/>
    </signal>
    <signal name="Error">
      <arg name="message" type="s"/>
    </signal>
  </interface>
</node>
"#;
