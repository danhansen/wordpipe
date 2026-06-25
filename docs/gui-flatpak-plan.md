# GUI Flatpak Plan

Wordpipe should feel like a GNOME utility app, with the Flatpak GUI as the
primary entry point and the command line kept as a diagnostic/backend layer.
Use the GNOME Human Interface Guidelines, GTK 4, and libadwaita as the design
baseline:

- GNOME HIG: https://developer.gnome.org/hig/
- GNOME UI components: https://developer.gnome.org/documentation/tutorials/beginners/components.html
- libadwaita API: https://gnome.pages.gitlab.gnome.org/libadwaita/doc/main/

## Design Direction

- Use `Adw.ApplicationWindow`, `Adw.HeaderBar`, `Adw.ToastOverlay`, `Adw.Banner`,
  `Adw.Clamp`, `Adw.PreferencesGroup`, `Adw.ActionRow`, `Adw.ComboRow`,
  `Adw.SwitchRow`, `Adw.SpinRow`, and `Adw.StatusPage` where they match the job.
- Keep the main window quiet and status-oriented: dictation state, selected
  model, live transcript, last committed text, and current errors.
- Put configuration in preferences-style rows rather than custom dashboard
  cards: model profile, microphone, shortcut, insertion behavior, spoken
  punctuation, stream pacing, and metrics/logging.
- Use GNOME feedback patterns: banners for persistent actionable problems,
  toasts for completed actions, spinners/progress rows for long-running model
  work, and dialogs only for decisions that need confirmation.
- Keep labels short and action-focused. Avoid verbose instructional text in the
  main workflow.
- Preserve adaptive layout behavior for narrow windows.

## Implementation Sequence

1. Main app shell
   - Convert the main view to libadwaita groups and rows.
   - Keep GTK fallback usable for environments without libadwaita.
   - Add native error banner and toast hooks.

2. First-run readiness
   - Show Wayland, portal, microphone, model, shortcut, and daemon readiness in
     one status page.
   - Make missing model and missing shortcut states actionable.

3. Model management
   - Keep `model-install` as the backend.
   - Show install phase, progress text, retry, and logs.
   - Allow installing and switching both `compact` and `fast`.

4. Microphone settings
   - Add a Parakeet/CPAL input-device selector.
   - Add a microphone test with level meter and permission errors.

5. Voice keyboard service
   - Move daemon start/stop/restart/status/log access into reusable app code.
   - Surface stale pid and worker crash recovery in the GUI.

6. GNOME shortcut setup
   - Detect the host custom shortcut.
   - Add install/repair controls for the Flatpak launcher command.
   - Show the active keybinding.

7. Dictation safety
   - Warn that portal keyboard insertion follows current focus.
   - Add a safe mode that buffers/stops insertion on suspected focus loss.
   - Leave anchored original-field insertion to a future GNOME Shell extension.

8. Distribution hardening
   - Replace network-enabled local Flatpak sources with generated Cargo and pip
     source manifests.
   - Keep AppStream metadata, icon, screenshots, and validation gates current.

9. Validation
   - Test GNOME Text Editor, Terminal, Firefox text fields, LibreOffice Writer,
     Flatpak text fields, microphone switching, shortcut repair, model install,
     daemon crash/stale pid recovery, and upgrades with existing config/model
     caches.
