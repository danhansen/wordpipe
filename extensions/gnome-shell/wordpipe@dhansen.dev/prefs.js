import Adw from 'gi://Adw';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import Gtk from 'gi://Gtk';

import {ExtensionPreferences, gettext as _} from 'resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js';

const BUS_NAME = 'dev.wordpipe.Service';
const OBJECT_PATH = '/dev/wordpipe/Service';

const SERVICE_XML = `
<node>
  <interface name="dev.wordpipe.Service1">
    <method name="Start"/>
    <method name="Stop"/>
    <method name="Toggle"/>
    <method name="Shutdown"/>
    <method name="GetState"><arg name="state" type="a{sv}" direction="out"/></method>
    <method name="GetConfig"><arg name="config" type="a{sv}" direction="out"/></method>
    <method name="ListBackends"><arg name="backends" type="aa{sv}" direction="out"/></method>
    <method name="ListModelProfiles"><arg name="profiles" type="aa{sv}" direction="out"/></method>
    <method name="ListInputDevices"><arg name="devices" type="aa{sv}" direction="out"/></method>
    <method name="SetBackend"><arg name="backend" type="s" direction="in"/></method>
    <method name="SetModelProfile"><arg name="profile" type="s" direction="in"/></method>
    <method name="SetInputDevice"><arg name="selector" type="s" direction="in"/></method>
    <method name="SetShortcut"><arg name="accelerator" type="s" direction="in"/></method>
    <method name="SetInsertionOptions"><arg name="options" type="a{sv}" direction="in"/></method>
    <method name="SetRuntimeOptions"><arg name="options" type="a{sv}" direction="in"/></method>
    <method name="InstallModel"><arg name="profile" type="s" direction="in"/></method>
    <signal name="StateChanged"><arg name="state" type="a{sv}"/></signal>
    <signal name="ConfigChanged"><arg name="config" type="a{sv}"/></signal>
    <signal name="SessionStarted"><arg name="session_id" type="t"/></signal>
    <signal name="TextDelta"><arg name="session_id" type="t"/><arg name="seq" type="t"/><arg name="text" type="s"/></signal>
    <signal name="Partial"><arg name="session_id" type="t"/><arg name="seq" type="t"/><arg name="full_text" type="s"/></signal>
    <signal name="Commit"><arg name="session_id" type="t"/><arg name="seq" type="t"/><arg name="text" type="s"/></signal>
    <signal name="SessionStopped"><arg name="session_id" type="t"/></signal>
    <signal name="InstallProgress"><arg name="profile" type="s"/><arg name="progress" type="a{sv}"/></signal>
    <signal name="Metrics"><arg name="metrics" type="a{sv}"/></signal>
    <signal name="Error"><arg name="message" type="s"/></signal>
  </interface>
</node>`;

const WordpipeProxy = Gio.DBusProxy.makeProxyWrapper(SERVICE_XML);

const PROFILES = [
    ['fast', _('Fast'), _('FP32 projected-cache model; fastest profile, largest footprint.')],
    ['compact', _('Compact'), _('Dynamic-int8 fixed-shape profile with ORT-format startup.')],
];

const BACKENDS = [
    ['parakeet', _('Parakeet')],
];

const WordpipePage = GObject.registerClass(
class WordpipePage extends Adw.PreferencesPage {
    constructor(settings) {
        super({
            title: _('Wordpipe'),
            icon_name: 'audio-input-microphone-symbolic',
        });
        this._settings = settings;
        this._proxy = null;
        this._signalIds = [];
        this._syncingSettings = false;
        this._backends = BACKENDS.map(([id, title]) => ({id, title, description: ''}));
        this._profiles = PROFILES.map(([id, title, description]) => ({
            id,
            title,
            description,
            installed: false,
            runtime_dir: '',
        }));
        this._deviceSelectors = [''];
        this._installRows = new Map();
        this._installButtons = new Map();
        this._installing = false;
        this._installingProfile = '';
        this._buildModelGroup();
        this._buildInputGroup();
        this._buildBehaviorGroup();
        this._buildAdvancedGroup();
        this._buildServiceGroup();
        this._connectProxy();
    }

    vfunc_unroot() {
        if (this._proxy) {
            for (const id of this._signalIds)
                this._proxy.disconnectSignal(id);
            this._signalIds = [];
            this._proxy = null;
        }
        super.vfunc_unroot();
    }

    _buildModelGroup() {
        this._modelGroup = new Adw.PreferencesGroup({
            title: _('Model'),
        });
        this.add(this._modelGroup);

        this._backendModel = new Gtk.StringList();
        this._backends.forEach(backend => this._backendModel.append(backend.title));
        this._backendRow = new Adw.ComboRow({
            title: _('Backend'),
            model: this._backendModel,
        });
        this._backendRow.selected = this._selectedIndex(
            this._backends, this._settings.get_string('backend'));
        this._backendRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const backend = this._backends[row.selected]?.id;
            if (!backend)
                return;
            this._settings.set_string('backend', backend);
            this._callRemote('SetBackend', backend);
        });
        this._modelGroup.add(this._backendRow);

        this._profileModel = new Gtk.StringList();
        this._profiles.forEach(profile => this._profileModel.append(profile.title));
        this._profileRow = new Adw.ComboRow({
            title: _('Model Profile'),
            subtitle: _('Choose the speed, memory, and disk footprint tradeoff.'),
            model: this._profileModel,
        });
        this._profileRow.selected = this._selectedIndex(
            this._profiles, this._settings.get_string('model-profile'));
        this._profileRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const profile = this._profiles[row.selected]?.id;
            if (!profile)
                return;
            this._settings.set_string('model-profile', profile);
            this._callRemote('SetModelProfile', profile);
        });
        this._modelGroup.add(this._profileRow);

        this._rebuildProfileRows();
    }

    _buildInputGroup() {
        const group = new Adw.PreferencesGroup({
            title: _('Input'),
        });
        this.add(group);

        this._deviceModel = new Gtk.StringList();
        this._deviceModel.append(_('System Default'));
        this._deviceRow = new Adw.ComboRow({
            title: _('Microphone'),
            subtitle: _('The service uses the system default when no device is selected.'),
            model: this._deviceModel,
        });
        this._deviceRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const selector = this._deviceSelectors[row.selected] ?? '';
            this._settings.set_string('input-device', selector);
            this._callRemote('SetInputDevice', selector);
        });
        group.add(this._deviceRow);

        const refreshRow = new Adw.ActionRow({
            title: _('Refresh Microphones'),
        });
        const button = new Gtk.Button({
            icon_name: 'view-refresh-symbolic',
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Refresh input devices from the Wordpipe service'),
        });
        button.connect('clicked', () => this._refreshInputDevices());
        refreshRow.add_suffix(button);
        group.add(refreshRow);
    }

    _buildBehaviorGroup() {
        const group = new Adw.PreferencesGroup({
            title: _('Behavior'),
        });
        this.add(group);

        let row = new Adw.SwitchRow({
            title: _('Spoken Punctuation'),
            active: this._settings.get_boolean('spoken-punctuation'),
        });
        row.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('spoken-punctuation', widget.active);
            this._pushInsertionOptions();
        });
        this._spokenPunctuationRow = row;
        group.add(row);

        row = new Adw.SwitchRow({
            title: _('Stream Text Immediately'),
            active: this._settings.get_boolean('insert-partials'),
        });
        row.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('insert-partials', widget.active);
            this._pushInsertionOptions();
        });
        this._insertPartialsRow = row;
        group.add(row);

        row = new Adw.SwitchRow({
            title: _('Show Overlay'),
            active: this._settings.get_boolean('show-overlay'),
        });
        row.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('show-overlay', widget.active);
            this._pushInsertionOptions();
        });
        this._showOverlayRow = row;
        group.add(row);

        this._delayRow = Adw.SpinRow.new_with_range(0, 1000, 25);
        this._delayRow.title = _('Insertion Delay');
        this._delayRow.subtitle = _('Additional delay in milliseconds before inserting streamed text.');
        this._delayRow.value = this._settings.get_uint('stream-insert-delay-ms');
        this._delayRow.connect('notify::value', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_uint('stream-insert-delay-ms', Math.round(widget.value));
            this._pushInsertionOptions();
        });
        group.add(this._delayRow);

        this._shortcutRow = new Adw.EntryRow({
            title: _('Shortcut'),
            text: this._settings.get_strv('toggle-shortcut')[0] ?? '',
        });
        this._shortcutRow.connect('changed', row => {
            if (this._syncingSettings)
                return;
            const accelerator = row.text.trim();
            this._settings.set_strv('toggle-shortcut', accelerator ? [accelerator] : []);
            this._callRemote('SetShortcut', accelerator);
        });
        group.add(this._shortcutRow);
    }

    _buildAdvancedGroup() {
        const group = new Adw.PreferencesGroup({
            title: _('Advanced'),
        });
        this.add(group);

        this._modelRootRow = new Adw.EntryRow({
            title: _('Model Directory'),
            text: this._settings.get_string('model-root'),
        });
        this._modelRootRow.connect('changed', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_string('model-root', row.text.trim());
            this._pushRuntimeOptions();
        });
        group.add(this._modelRootRow);

        this._threadsRow = Adw.SpinRow.new_with_range(1, 16, 1);
        this._threadsRow.title = _('Worker Threads');
        this._threadsRow.value = this._settings.get_uint('num-threads');
        this._threadsRow.connect('notify::value', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_uint('num-threads', Math.max(1, Math.round(row.value)));
            this._pushRuntimeOptions();
        });
        group.add(this._threadsRow);

        this._sampleRateRow = Adw.SpinRow.new_with_range(8000, 48000, 1000);
        this._sampleRateRow.title = _('Sample Rate');
        this._sampleRateRow.value = this._settings.get_uint('sample-rate');
        this._sampleRateRow.connect('notify::value', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_uint('sample-rate', Math.max(1, Math.round(row.value)));
            this._pushRuntimeOptions();
        });
        group.add(this._sampleRateRow);
    }


    _buildServiceGroup() {
        const group = new Adw.PreferencesGroup({
            title: _('Service'),
        });
        this.add(group);

        this._statusRow = new Adw.ActionRow({
            title: _('Status'),
            subtitle: _('Connecting'),
        });
        group.add(this._statusRow);

        this._progressRow = new Adw.ActionRow({
            title: _('Model Setup'),
            subtitle: _('Idle'),
        });
        group.add(this._progressRow);

        this._metricsRow = new Adw.ActionRow({
            title: _('Runtime Metrics'),
            subtitle: _('No metrics yet'),
        });
        group.add(this._metricsRow);

        const actions = new Adw.ActionRow({
            title: _('Dictation'),
        });
        this._startButton = new Gtk.Button({
            icon_name: 'media-record-symbolic',
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Start dictation'),
        });
        this._startButton.connect('clicked', () => this._callRemote('Start'));
        this._stopButton = new Gtk.Button({
            icon_name: 'media-playback-stop-symbolic',
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Stop dictation'),
        });
        this._stopButton.connect('clicked', () => this._callRemote('Stop'));
        actions.add_suffix(this._startButton);
        actions.add_suffix(this._stopButton);
        group.add(actions);
    }

    _connectProxy() {
        this._proxy = new WordpipeProxy(
            Gio.DBus.session,
            BUS_NAME,
            OBJECT_PATH,
            (_proxy, error) => {
                if (error) {
                    this._statusRow.subtitle = _('Service unavailable');
                    return;
                }
                this._statusRow.subtitle = _('Connected');
                this._subscribeSignals();
                this._refreshBackends();
                this._refreshModelProfiles();
                this._refreshConfig();
                this._refreshState();
                this._refreshInputDevices();
            });
    }

    _subscribeSignals() {
        if (this._signalIds.length > 0)
            return;
        this._signalIds.push(this._proxy.connectSignal('StateChanged',
            (_proxy, _sender, [state]) => this._handleState(deepUnpackMap(state))));
        this._signalIds.push(this._proxy.connectSignal('ConfigChanged',
            (_proxy, _sender, [config]) => this._syncFromConfig(deepUnpackMap(config))));
        this._signalIds.push(this._proxy.connectSignal('InstallProgress',
            (_proxy, _sender, [profile, progress]) => {
                this._handleInstallProgress(profile, deepUnpackMap(progress));
            }));
        this._signalIds.push(this._proxy.connectSignal('Metrics',
            (_proxy, _sender, [metrics]) => {
                const summary = formatMetrics(deepUnpackMap(metrics));
                if (summary)
                    this._metricsRow.subtitle = summary;
            }));
        this._signalIds.push(this._proxy.connectSignal('Error',
            (_proxy, _sender, [message]) => {
                this._statusRow.subtitle = message;
                this._progressRow.subtitle = message;
            }));
    }

    _refreshState() {
        this._callRemote('GetState', state => {
            this._handleState(deepUnpackMap(state));
        });
    }

    _refreshConfig() {
        this._callRemote('GetConfig', config => {
            this._syncFromConfig(deepUnpackMap(config));
        });
    }

    _refreshBackends() {
        this._callRemote('ListBackends', backends => {
            const parsed = backends.map(item => deepUnpackMap(item))
                .filter(item => typeof item.id === 'string');
            if (parsed.length === 0)
                return;
            this._backends = parsed.map(item => ({
                id: item.id,
                title: item.title ?? item.id,
                description: item.description ?? '',
            }));
            clearStringList(this._backendModel);
            this._backends.forEach(backend => this._backendModel.append(backend.title));
            this._syncComboSelections();
        });
    }

    _refreshModelProfiles() {
        this._callRemote('ListModelProfiles', profiles => {
            const parsed = profiles.map(item => deepUnpackMap(item))
                .filter(item => typeof item.id === 'string');
            if (parsed.length === 0)
                return;
            this._profiles = parsed.map(item => ({
                id: item.id,
                title: item.title ?? item.id,
                description: item.description ?? '',
                installed: Boolean(item.installed),
                runtime_dir: item.runtime_dir ?? '',
            }));
            clearStringList(this._profileModel);
            this._profiles.forEach(profile => this._profileModel.append(profile.title));
            this._rebuildProfileRows();
            this._syncComboSelections();
        });
    }

    _refreshInputDevices() {
        this._callRemote('ListInputDevices', devices => {
            clearStringList(this._deviceModel);
            this._deviceSelectors = [''];
            this._deviceModel.append(_('System Default'));

            for (const device of devices) {
                const values = deepUnpackMap(device);
                const selector = values.selector ?? values.name ?? '';
                this._deviceSelectors.push(selector);
                this._deviceModel.append(values.is_default
                    ? _(`${values.name} (default)`)
                    : values.name);
            }

            const configured = this._settings.get_string('input-device');
            const selected = Math.max(0, this._deviceSelectors.indexOf(configured));
            this._withSyncing(() => {
                this._deviceRow.selected = selected;
            });
        });
    }

    _syncFromConfig(values) {
        this._syncingSettings = true;
        try {
            if (typeof values.backend === 'string')
                this._settings.set_string('backend', values.backend);
            if (typeof values.model_profile === 'string')
                this._settings.set_string('model-profile', values.model_profile);
            if (typeof values.input_device === 'string')
                this._settings.set_string('input-device', values.input_device);
            if (typeof values.model_root === 'string')
                this._settings.set_string('model-root', values.model_root);
            if (typeof values.shortcut === 'string')
                this._settings.set_strv('toggle-shortcut', values.shortcut ? [values.shortcut] : []);
            if (typeof values.num_threads === 'number')
                this._settings.set_uint('num-threads', values.num_threads);
            if (typeof values.sample_rate === 'number')
                this._settings.set_uint('sample-rate', values.sample_rate);
            if (typeof values.spoken_punctuation === 'boolean')
                this._settings.set_boolean('spoken-punctuation', values.spoken_punctuation);
            if (typeof values.insert_partials === 'boolean')
                this._settings.set_boolean('insert-partials', values.insert_partials);
            if (typeof values.stream_insert_delay_ms === 'number')
                this._settings.set_uint('stream-insert-delay-ms', values.stream_insert_delay_ms);
            if (typeof values.show_overlay === 'boolean')
                this._settings.set_boolean('show-overlay', values.show_overlay);

            this._syncComboSelections();
            this._syncDeviceSelection();
            this._syncControlValues();
        } finally {
            this._syncingSettings = false;
        }
    }

    _syncComboSelections() {
        if (!this._backendRow || !this._profileRow)
            return;
        const backend = this._settings.get_string('backend');
        const profile = this._settings.get_string('model-profile');
        this._withSyncing(() => {
            this._backendRow.selected = this._selectedIndex(this._backends, backend);
            this._profileRow.selected = this._selectedIndex(this._profiles, profile);
        });
    }

    _syncDeviceSelection() {
        if (!this._deviceRow)
            return;
        const configured = this._settings.get_string('input-device');
        const selected = Math.max(0, this._deviceSelectors.indexOf(configured));
        this._withSyncing(() => {
            this._deviceRow.selected = selected;
        });
    }

    _syncControlValues() {
        this._spokenPunctuationRow.active = this._settings.get_boolean('spoken-punctuation');
        this._insertPartialsRow.active = this._settings.get_boolean('insert-partials');
        this._showOverlayRow.active = this._settings.get_boolean('show-overlay');
        this._delayRow.value = this._settings.get_uint('stream-insert-delay-ms');
        this._shortcutRow.text = this._settings.get_strv('toggle-shortcut')[0] ?? '';
        this._modelRootRow.text = this._settings.get_string('model-root');
        this._threadsRow.value = this._settings.get_uint('num-threads');
        this._sampleRateRow.value = this._settings.get_uint('sample-rate');
    }

    _rebuildProfileRows() {
        for (const row of this._installRows.values())
            this._modelGroup.remove(row);
        this._installRows.clear();
        this._installButtons.clear();

        for (const profile of this._profiles) {
            const installing = this._installing && this._installingProfile === profile.id;
            const row = new Adw.ActionRow({
                title: installing
                    ? _(`Installing ${profile.title}`)
                    : profile.installed
                    ? profile.title
                    : _(`Install ${profile.title}`),
                subtitle: this._profileSubtitle(profile),
            });
            const button = new Gtk.Button({
                icon_name: profile.installed
                    ? 'emblem-ok-symbolic'
                    : installing
                        ? 'emblem-synchronizing-symbolic'
                        : 'folder-download-symbolic',
                valign: Gtk.Align.CENTER,
                sensitive: !profile.installed && !this._installing,
                tooltip_text: profile.installed
                    ? _(`${profile.title} is installed`)
                    : installing
                        ? _(`${profile.title} is installing`)
                    : _(`Download and prepare the ${profile.title} model`),
            });
            button.connect('clicked', () => {
                if (profile.installed || this._installing)
                    return;
                this._progressRow.subtitle = _(`Starting ${profile.title}`);
                this._callRemote('InstallModel', profile.id);
            });
            row.add_suffix(button);
            this._installRows.set(profile.id, row);
            this._installButtons.set(profile.id, button);
            this._modelGroup.add(row);
        }
    }

    _profileSubtitle(profile) {
        if (this._installing && this._installingProfile === profile.id)
            return _('Installing');
        const status = profile.installed ? _('Installed') : _('Not installed');
        const detail = profile.description || profile.runtime_dir;
        return detail ? `${status} - ${detail}` : status;
    }

    _handleState(values) {
        const previousInstalling = this._installing;
        const previousInstallingProfile = this._installingProfile;
        this._installing = Boolean(values.installing);
        this._installingProfile = values.installing_profile ?? '';
        const selectedModelInstalled = values.selected_model_installed !== false;
        if (values.loading_model)
            this._statusRow.subtitle = _('Loading model');
        else if (values.listening)
            this._statusRow.subtitle = _('Listening');
        else if (values.stopping)
            this._statusRow.subtitle = _('Stopping');
        else if (values.installing)
            this._statusRow.subtitle = _('Installing model');
        else if (!selectedModelInstalled)
            this._statusRow.subtitle = _('Model missing');
        else if (values.last_error)
            this._statusRow.subtitle = values.last_error;
        else
            this._statusRow.subtitle = _('Ready');
        if (this._startButton) {
            this._startButton.sensitive = !values.loading_model &&
                !values.installing &&
                !values.listening &&
                !values.stopping &&
                selectedModelInstalled;
        }
        if (this._stopButton)
            this._stopButton.sensitive = Boolean(values.listening || values.stopping);
        this._syncInstallButtons();
        if (
            previousInstalling !== this._installing ||
            previousInstallingProfile !== this._installingProfile
        )
            this._rebuildProfileRows();
    }

    _handleInstallProgress(profile, progress) {
        const message = progress.message ?? progress.phase ?? '';
        this._progressRow.subtitle = message ? `${profile}: ${message}` : profile;
        if (progress.phase === 'complete' || progress.phase === 'error')
            this._refreshModelProfiles();
    }

    _syncInstallButtons() {
        for (const [profileId, button] of this._installButtons.entries()) {
            const profile = this._profiles.find(item => item.id === profileId);
            button.sensitive = Boolean(profile && !profile.installed && !this._installing);
        }
    }

    _selectedIndex(items, selectedId) {
        return Math.max(0, items.findIndex(item => item.id === selectedId));
    }

    _withSyncing(callback) {
        const wasSyncing = this._syncingSettings;
        this._syncingSettings = true;
        try {
            callback();
        } finally {
            this._syncingSettings = wasSyncing;
        }
    }

    _pushInsertionOptions() {
        this._callRemote('SetInsertionOptions', {
            spoken_punctuation: new GLib.Variant('b',
                this._settings.get_boolean('spoken-punctuation')),
            insert_partials: new GLib.Variant('b',
                this._settings.get_boolean('insert-partials')),
            stream_insert_delay_ms: new GLib.Variant('u',
                this._settings.get_uint('stream-insert-delay-ms')),
            show_overlay: new GLib.Variant('b',
                this._settings.get_boolean('show-overlay')),
        });
    }

    _pushRuntimeOptions() {
        this._callRemote('SetRuntimeOptions', {
            model_root: new GLib.Variant('s', this._settings.get_string('model-root')),
            num_threads: new GLib.Variant('u', this._settings.get_uint('num-threads')),
            sample_rate: new GLib.Variant('u', this._settings.get_uint('sample-rate')),
        });
    }

    _callRemote(method, ...args) {
        const callback = typeof args.at(-1) === 'function' ? args.pop() : null;
        const remote = this._proxy?.[`${method}Remote`];
        if (!remote)
            return;
        remote.call(this._proxy, ...args, (result, error) => {
            if (error) {
                this._statusRow.subtitle = _('Service unavailable');
                logError(error, `Wordpipe ${method} failed`);
                return;
            }
            if (callback)
                callback(...result);
        });
    }
});

export default class WordpipePreferences extends ExtensionPreferences {
    fillPreferencesWindow(window) {
        window.add(new WordpipePage(this.getSettings()));
    }
}

function deepUnpackMap(value) {
    if (!value)
        return {};
    const unpacked = value.deep_unpack ? value.deep_unpack() : value;
    const result = {};
    for (const [key, variant] of Object.entries(unpacked))
        result[key] = variant?.deep_unpack ? variant.deep_unpack() : variant;
    return result;
}

function clearStringList(model) {
    while (model.get_n_items() > 0)
        model.remove(0);
}

function formatMetrics(metrics) {
    const rtf = numberValue(metrics.real_audio_real_time_factor ?? metrics.real_time_factor);
    const audioSeconds = numberValue(metrics.audio_seconds);
    const decodeSeconds = numberValue(metrics.decode_seconds);
    const droppedChunks = numberValue(metrics.dropped_audio_chunks);
    if (
        rtf === null &&
        audioSeconds === null &&
        decodeSeconds === null &&
        droppedChunks === null
    )
        return '';

    const parts = [];
    if (rtf !== null)
        parts.push(`RTF ${rtf.toFixed(3)}`);
    if (audioSeconds !== null)
        parts.push(`${audioSeconds.toFixed(1)}s audio`);
    if (decodeSeconds !== null)
        parts.push(`${decodeSeconds.toFixed(1)}s decode`);
    if (droppedChunks)
        parts.push(`${droppedChunks} ${_('dropped')}`);
    return parts.join(' - ');
}

function numberValue(value) {
    if (typeof value === 'number' && Number.isFinite(value))
        return value;
    return null;
}
