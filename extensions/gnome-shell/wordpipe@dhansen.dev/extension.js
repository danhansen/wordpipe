import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import Meta from 'gi://Meta';
import Shell from 'gi://Shell';
import St from 'gi://St';

import {Extension, gettext as _} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';

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

const Indicator = GObject.registerClass(
class Indicator extends PanelMenu.Button {
    constructor(extension) {
        super(0.0, _('Wordpipe'));
        this._extension = extension;
        this._listening = false;

        this._box = new St.BoxLayout({
            style_class: 'wordpipe-panel-status',
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._icon = new St.Icon({
            icon_name: 'audio-input-microphone-symbolic',
            style_class: 'system-status-icon wordpipe-panel-icon',
        });
        this._icon.set_pivot_point(0.5, 0.5);
        this._box.add_child(this._icon);

        this._levelBars = [];
        this._levelBox = new St.BoxLayout({
            style_class: 'wordpipe-level-bars',
            y_align: Clutter.ActorAlign.CENTER,
            opacity: 0,
        });
        for (let i = 0; i < 4; i++) {
            const bar = new St.Widget({
                style_class: 'wordpipe-level-bar',
                y_align: Clutter.ActorAlign.CENTER,
            });
            bar.set_pivot_point(0.5, 1.0);
            bar.scale_y = 0.2;
            this._levelBox.add_child(bar);
            this._levelBars.push(bar);
        }
        this._box.add_child(this._levelBox);
        this.add_child(this._box);

        this._statusItem = new PopupMenu.PopupMenuItem(_('Service unavailable'), {
            reactive: false,
        });
        this.menu.addMenuItem(this._statusItem);

        this._toggleItem = new PopupMenu.PopupMenuItem(_('Start Dictation'));
        this._toggleItem.connect('activate', () => this._extension.toggleDictation());
        this.menu.addMenuItem(this._toggleItem);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        this._profileStatusItem = new PopupMenu.PopupMenuItem(_('Model unavailable'), {
            reactive: false,
        });
        this.menu.addMenuItem(this._profileStatusItem);

        this._profileItems = [];
        this._installProfileItems = new Map();
        this._installing = false;
        this._installingProfile = '';
        this._profileSection = new PopupMenu.PopupMenuSection();
        this.menu.addMenuItem(this._profileSection);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        const prefsItem = new PopupMenu.PopupMenuItem(_('Preferences'));
        prefsItem.connect('activate', () => this._extension.openPreferences());
        this.menu.addMenuItem(prefsItem);
    }

    setState(state, available) {
        const listening = available && Boolean(state?.listening);
        const stopping = Boolean(state?.stopping);
        const installing = Boolean(state?.installing);
        const loading = Boolean(state?.loading_model);
        const selectedModelInstalled = state?.selected_model_installed !== false;
        this._installing = installing;
        this._installingProfile = state?.installing_profile ?? '';
        this._setListening(listening);
        this._toggleItem.label.text = listening || stopping
            ? _('Stop Dictation')
            : _('Start Dictation');
        this._toggleItem.setSensitive(
            available && !installing && !loading &&
            (listening || stopping || selectedModelInstalled));
        this._statusItem.label.text = available
            ? statusText(state, selectedModelInstalled)
            : _('Service unavailable');
        this._syncInstallActions();
    }

    setProfiles(profiles, selectedProfile) {
        for (const item of this._profileItems)
            item.destroy();
        this._profileItems = [];
        this._installProfileItems.clear();

        if (!profiles.length) {
            this._profileStatusItem.label.text = _('No model profiles');
            return;
        }

        const selected = profiles.find(profile => profile.id === selectedProfile);
        this._profileStatusItem.label.text = selected
            ? `${_('Model')}: ${selected.title}${selected.installed ? '' : ` (${_('not installed')})`}`
            : `${_('Model')}: ${selectedProfile || _('Unknown')}`;

        for (const profile of profiles) {
            const isSelected = profile.id === selectedProfile;
            if (profile.installed) {
                const selectItem = new PopupMenu.PopupMenuItem(
                    isSelected
                        ? `${profile.title} ${_('selected')}`
                        : `${_('Use')} ${profile.title}`,
                    {reactive: !isSelected});
                if (!isSelected) {
                    selectItem.connect('activate',
                        () => this._extension.selectModelProfile(profile.id));
                }
                this._profileSection.addMenuItem(selectItem);
                this._profileItems.push(selectItem);
            } else {
                const installing = this._installingProfile === profile.id;
                const missingItem = new PopupMenu.PopupMenuItem(
                    installing
                        ? `${profile.title} ${_('installing')}`
                        : `${profile.title} ${_('not installed')}`,
                    {reactive: false});
                this._profileSection.addMenuItem(missingItem);
                this._profileItems.push(missingItem);

                const installItem = new PopupMenu.PopupMenuItem(
                    `${_('Install')} ${profile.title}`);
                installItem.setSensitive(!this._installing);
                installItem.connect('activate', () => this._extension.installModel(profile.id));
                this._profileSection.addMenuItem(installItem);
                this._profileItems.push(installItem);
                this._installProfileItems.set(profile.id, installItem);
            }
        }
    }

    setMetrics(summary) {
        if (summary)
            this._statusItem.label.text = summary;
    }

    setStatusMessage(message) {
        if (message)
            this._statusItem.label.text = message;
    }

    setVoiceLevel(rms) {
        if (!this._listening)
            return;
        const level = normalizeVoiceLevel(rms);
        this._icon.ease({
            scale_x: 1.0 + level * 0.08,
            scale_y: 1.0 + level * 0.16,
            duration: 100,
            mode: Clutter.AnimationMode.EASE_OUT_QUAD,
        });

        const multipliers = [0.55, 1.0, 0.75, 0.45];
        this._levelBars.forEach((bar, index) => {
            const scale = 0.25 + level * multipliers[index] * 1.35;
            bar.ease({
                scale_y: Math.max(0.2, Math.min(1.6, scale)),
                duration: 100,
                mode: Clutter.AnimationMode.EASE_OUT_QUAD,
            });
        });
    }

    _syncInstallActions() {
        for (const item of this._installProfileItems.values())
            item.setSensitive(!this._installing);
    }

    _setListening(listening) {
        if (this._listening === listening)
            return;
        this._listening = listening;
        this._levelBox.ease({
            opacity: listening ? 255 : 0,
            duration: 120,
            mode: Clutter.AnimationMode.EASE_OUT_QUAD,
        });
        if (!listening)
            this._resetVoiceLevel();
    }

    _resetVoiceLevel() {
        this._icon.ease({
            scale_x: 1.0,
            scale_y: 1.0,
            duration: 120,
            mode: Clutter.AnimationMode.EASE_OUT_QUAD,
        });
        for (const bar of this._levelBars)
            bar.scale_y = 0.2;
    }
});

class TextInjector {
    constructor() {
        this._lastSession = 0;
        this._lastSeq = 0;
        this._insertedText = '';
        this._inputMethod = null;
        this._pendingDeltaIds = new Set();
    }

    reset(sessionId) {
        this._clearPendingDeltas();
        this._lastSession = Number(sessionId);
        this._lastSeq = 0;
        this._insertedText = '';
    }

    insertDelta(sessionId, seq, text, delayMs = 0) {
        if (delayMs > 0) {
            const sourceId = GLib.timeout_add(
                GLib.PRIORITY_DEFAULT,
                delayMs,
                () => {
                    this._pendingDeltaIds.delete(sourceId);
                    this._insertDeltaNow(sessionId, seq, text);
                    return GLib.SOURCE_REMOVE;
                });
            this._pendingDeltaIds.add(sourceId);
            return;
        }
        this._insertDeltaNow(sessionId, seq, text);
    }

    _insertDeltaNow(sessionId, seq, text) {
        const numericSession = Number(sessionId);
        const numericSeq = Number(seq);
        if (numericSession !== this._lastSession)
            this.reset(numericSession);
        if (numericSeq <= this._lastSeq || !text)
            return;
        this._lastSeq = numericSeq;

        const inputMethod = this._getInputMethod();
        if (!inputMethod) {
            log(`Wordpipe text delta without input method: ${text}`);
            return;
        }
        inputMethod.commit(text);
        this._insertedText += text;
    }

    insertCommit(sessionId, seq, text) {
        const numericSession = Number(sessionId);
        const numericSeq = Number(seq);
        if (numericSession !== this._lastSession)
            this.reset(numericSession);
        else
            this._clearPendingDeltas();
        if (numericSeq <= this._lastSeq || !text)
            return;
        this._lastSeq = numericSeq;

        let textToInsert = text;
        if (this._insertedText) {
            if (text === this._insertedText || this._insertedText.startsWith(text))
                return;
            if (text.startsWith(this._insertedText))
                textToInsert = text.slice(this._insertedText.length);
            else {
                log(`Wordpipe commit differs from streamed text; keeping streamed text: ${text}`);
                return;
            }
        }

        const inputMethod = this._getInputMethod();
        if (!inputMethod) {
            log(`Wordpipe commit without input method: ${textToInsert}`);
            return;
        }
        inputMethod.commit(textToInsert);
        this._insertedText += textToInsert;
    }

    _getInputMethod() {
        if (this._inputMethod)
            return this._inputMethod;

        const backend = Clutter.get_default_backend?.();
        this._inputMethod = backend?.get_input_method?.() ?? null;
        return this._inputMethod;
    }

    _clearPendingDeltas() {
        for (const sourceId of this._pendingDeltaIds)
            GLib.Source.remove(sourceId);
        this._pendingDeltaIds.clear();
    }
}

export default class WordpipeExtension extends Extension {
    enable() {
        this._settings = this.getSettings();
        this._state = {};
        this._profiles = [];
        this._signalIds = [];
        this._syncingSettings = false;
        this._injector = new TextInjector();

        this._indicator = new Indicator(this);
        Main.panel.addToStatusArea(this.uuid, this._indicator);

        this._bindShortcut();
        this._connectSettings();
        this._connectProxy();
    }

    disable() {
        this._settings?.disconnectObject(this);
        Main.wm.removeKeybinding('toggle-shortcut');

        if (this._proxy) {
            for (const id of this._signalIds)
                this._proxy.disconnectSignal(id);
            this._signalIds = [];
            this._proxy = null;
        }

        this._indicator?.destroy();
        this._indicator = null;
        this._injector = null;
        this._settings = null;
    }

    toggleDictation() {
        this._callRemote('Toggle');
    }

    installModel(profile) {
        this._callRemote('InstallModel', profile);
    }

    selectModelProfile(profile) {
        this._settings.set_string('model-profile', profile);
        this._callRemote('SetModelProfile', profile);
    }

    _bindShortcut() {
        Main.wm.addKeybinding(
            'toggle-shortcut',
            this._settings,
            Meta.KeyBindingFlags.IGNORE_AUTOREPEAT,
            Shell.ActionMode.NORMAL | Shell.ActionMode.OVERVIEW,
            () => this.toggleDictation());
    }

    _connectSettings() {
        this._settings.connectObject('changed', () => {
            if (!this._syncingSettings)
                this._pushSettings();
        }, this);
    }

    _connectProxy() {
        this._proxy = new WordpipeProxy(
            Gio.DBus.session,
            BUS_NAME,
            OBJECT_PATH,
            (proxy, error) => {
                if (error) {
                    this._setAvailable(false);
                    logError(error, 'Wordpipe could not connect to service');
                    return;
                }
                this._setAvailable(true);
                this._subscribeSignals();
                this._refreshState();
                this._refreshProfiles();
                this._refreshConfigFromService(() => this._pushSettings());
            });
    }

    _subscribeSignals() {
        this._signalIds.push(this._proxy.connectSignal('StateChanged',
            (_proxy, _sender, [state]) => this._handleState(deepUnpackMap(state))));
        this._signalIds.push(this._proxy.connectSignal('ConfigChanged',
            (_proxy, _sender, [config]) => {
                this._syncSettingsFromConfig(deepUnpackMap(config));
                this._syncProfileMenu();
                this._refreshState();
            }));
        this._signalIds.push(this._proxy.connectSignal('SessionStarted',
            (_proxy, _sender, [sessionId]) => {
                this._injector.reset(sessionId);
            }));
        this._signalIds.push(this._proxy.connectSignal('TextDelta',
            (_proxy, _sender, [sessionId, seq, text]) => {
                if (this._settings.get_boolean('insert-partials')) {
                    this._injector.insertDelta(
                        sessionId,
                        seq,
                        text,
                        this._settings.get_uint('stream-insert-delay-ms'));
                }
            }));
        this._signalIds.push(this._proxy.connectSignal('Commit',
            (_proxy, _sender, [sessionId, seq, text]) => {
                this._injector.insertCommit(sessionId, seq, text);
            }));
        this._signalIds.push(this._proxy.connectSignal('InstallProgress',
            (_proxy, _sender, [profile, progress]) => {
                const values = deepUnpackMap(progress);
                if (values.phase === 'complete' || values.phase === 'error')
                    this._refreshProfiles();
            }));
        this._signalIds.push(this._proxy.connectSignal('Metrics',
            (_proxy, _sender, [metrics]) => {
                const values = deepUnpackMap(metrics);
                const summary = formatMetrics(values);
                if (summary)
                    this._indicator?.setMetrics(summary);
                this._indicator?.setVoiceLevel(numberValue(values.last_rms) ?? 0.0);
            }));
        this._signalIds.push(this._proxy.connectSignal('Error',
            (_proxy, _sender, [message]) => {
                this._indicator?.setStatusMessage(message);
                log(`Wordpipe service error: ${message}`);
            }));
    }

    _refreshState() {
        this._callRemote('GetState', (state) => this._handleState(deepUnpackMap(state)));
    }

    _refreshConfigFromService(callback = null) {
        this._callRemote('GetConfig', config => {
            this._syncSettingsFromConfig(deepUnpackMap(config));
            this._syncProfileMenu();
            if (callback)
                callback();
        });
    }

    _refreshProfiles() {
        this._callRemote('ListModelProfiles', profiles => {
            this._profiles = profiles.map(profile => {
                const values = deepUnpackMap(profile);
                return {
                    id: values.id ?? '',
                    title: values.title ?? values.id ?? '',
                    installed: Boolean(values.installed),
                };
            }).filter(profile => profile.id);
            this._syncProfileMenu();
        });
    }

    _syncSettingsFromConfig(config) {
        this._syncingSettings = true;
        try {
            if (typeof config.backend === 'string')
                this._settings.set_string('backend', config.backend);
            if (typeof config.model_profile === 'string')
                this._settings.set_string('model-profile', config.model_profile);
            if (typeof config.input_device === 'string')
                this._settings.set_string('input-device', config.input_device);
            if (typeof config.model_root === 'string')
                this._settings.set_string('model-root', config.model_root);
            if (typeof config.worker_path === 'string')
                this._settings.set_string('worker-path', config.worker_path);
            if (typeof config.model_installer_path === 'string')
                this._settings.set_string('model-installer-path', config.model_installer_path);
            if (typeof config.shortcut === 'string')
                this._settings.set_strv('toggle-shortcut', config.shortcut ? [config.shortcut] : []);
            if (typeof config.num_threads === 'number')
                this._settings.set_uint('num-threads', config.num_threads);
            if (typeof config.sample_rate === 'number')
                this._settings.set_uint('sample-rate', config.sample_rate);
            if (typeof config.spoken_punctuation === 'boolean')
                this._settings.set_boolean('spoken-punctuation', config.spoken_punctuation);
            if (typeof config.insert_partials === 'boolean')
                this._settings.set_boolean('insert-partials', config.insert_partials);
            if (typeof config.stream_insert_delay_ms === 'number')
                this._settings.set_uint('stream-insert-delay-ms', config.stream_insert_delay_ms);
            if (typeof config.show_overlay === 'boolean')
                this._settings.set_boolean('show-overlay', false);
        } finally {
            this._syncingSettings = false;
        }
    }

    _pushSettings() {
        if (!this._proxy)
            return;

        const backend = this._settings.get_string('backend');
        const profile = this._settings.get_string('model-profile');
        const inputDevice = this._settings.get_string('input-device');
        const modelRoot = this._settings.get_string('model-root');
        const workerPath = this._settings.get_string('worker-path');
        const modelInstallerPath = this._settings.get_string('model-installer-path');
        const shortcuts = this._settings.get_strv('toggle-shortcut');
        const shortcut = shortcuts.length > 0 ? shortcuts[0] : '';
        const options = {
            spoken_punctuation: new GLib.Variant('b',
                this._settings.get_boolean('spoken-punctuation')),
            insert_partials: new GLib.Variant('b',
                this._settings.get_boolean('insert-partials')),
            stream_insert_delay_ms: new GLib.Variant('u',
                this._settings.get_uint('stream-insert-delay-ms')),
            show_overlay: new GLib.Variant('b', false),
        };

        this._callRemote('SetBackend', backend);
        this._callRemote('SetModelProfile', profile);
        this._callRemote('SetInputDevice', inputDevice);
        this._callRemote('SetShortcut', shortcut);
        this._callRemote('SetInsertionOptions', options);
        this._callRemote('SetRuntimeOptions', {
            model_root: new GLib.Variant('s', modelRoot),
            worker_path: new GLib.Variant('s', workerPath),
            model_installer_path: new GLib.Variant('s', modelInstallerPath),
            num_threads: new GLib.Variant('u',
                this._settings.get_uint('num-threads')),
            sample_rate: new GLib.Variant('u',
                this._settings.get_uint('sample-rate')),
        });
    }

    _syncProfileMenu() {
        this._indicator?.setProfiles(
            this._profiles,
            this._settings.get_string('model-profile'));
    }

    _callRemote(method, ...args) {
        const callback = typeof args.at(-1) === 'function' ? args.pop() : null;
        const remote = this._proxy?.[`${method}Remote`];
        if (!remote) {
            this._setAvailable(false);
            return;
        }
        remote.call(this._proxy, ...args, (result, error) => {
            if (error) {
                this._setAvailable(true);
                const message = formatError(error);
                this._indicator?.setStatusMessage(message);
                logError(error, `Wordpipe ${method} failed`);
                return;
            }
            this._setAvailable(true);
            if (callback)
                callback(...result);
        });
    }

    _handleState(state) {
        this._state = state;
        this._indicator?.setState(this._state, true);
        const installSummary = formatInstallProgress(state.last_install_progress ?? {});
        const metricsSummary = formatMetrics(state.last_metrics ?? {});
        if (state.installing && installSummary)
            this._indicator?.setMetrics(installSummary);
        else if (metricsSummary)
            this._indicator?.setMetrics(metricsSummary);
        this._indicator?.setVoiceLevel(numberValue(state.last_metrics?.last_rms) ?? 0.0);
    }

    _setAvailable(available) {
        this._indicator?.setState(this._state, available);
    }
}

function deepUnpackMap(value) {
    if (!value)
        return {};
    const unpacked = deepUnpackValue(value);
    if (!unpacked || typeof unpacked !== 'object' || Array.isArray(unpacked))
        return {};
    return unpacked;
}

function deepUnpackValue(value) {
    const unpacked = value?.deep_unpack ? value.deep_unpack() : value;
    if (!unpacked || typeof unpacked !== 'object' || Array.isArray(unpacked))
        return unpacked;
    const result = {};
    for (const [key, variant] of Object.entries(unpacked))
        result[key] = deepUnpackValue(variant);
    return result;
}

function statusText(state, selectedModelInstalled) {
    if (state?.installing)
        return _('Installing model');
    if (state?.loading_model)
        return _('Loading model');
    if (state?.listening)
        return _('Listening');
    if (state?.stopping)
        return _('Stopping');
    if (!selectedModelInstalled)
        return _('Model missing');
    return _('Ready');
}

function formatMetrics(metrics) {
    const rtf = numberValue(metrics.real_audio_real_time_factor ?? metrics.real_time_factor);
    const audioSeconds = numberValue(metrics.audio_seconds);
    const droppedChunks = numberValue(metrics.dropped_audio_chunks);
    if (rtf === null && audioSeconds === null && droppedChunks === null)
        return '';

    const parts = [];
    if (rtf !== null)
        parts.push(`RTF ${rtf.toFixed(3)}`);
    if (audioSeconds !== null)
        parts.push(`${audioSeconds.toFixed(1)}s`);
    if (droppedChunks)
        parts.push(`${droppedChunks} ${_('dropped')}`);
    return parts.join(' - ');
}

function formatInstallProgress(progress) {
    const profile = typeof progress.profile === 'string' ? progress.profile : '';
    const message = typeof progress.message === 'string'
        ? progress.message
        : typeof progress.phase === 'string'
            ? progress.phase
            : '';
    if (!profile && !message)
        return '';
    if (!profile)
        return message;
    if (!message)
        return profile;
    return `${profile}: ${message}`;
}

function numberValue(value) {
    if (typeof value === 'number' && Number.isFinite(value))
        return value;
    return null;
}

function normalizeVoiceLevel(rms) {
    const value = numberValue(rms) ?? 0.0;
    return Math.max(0.0, Math.min(1.0, (value - 0.004) * 18.0));
}

function formatError(error) {
    const message = error?.message ?? String(error);
    return message.replace(/^GDBus\.Error:[^:]+:\s*/, '');
}
