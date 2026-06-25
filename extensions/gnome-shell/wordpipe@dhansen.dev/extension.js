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

        this._icon = new St.Icon({
            icon_name: 'audio-input-microphone-symbolic',
            style_class: 'system-status-icon wordpipe-panel-icon',
        });
        this.add_child(this._icon);

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
        this._profileSection = new PopupMenu.PopupMenuSection();
        this.menu.addMenuItem(this._profileSection);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        const prefsItem = new PopupMenu.PopupMenuItem(_('Preferences'));
        prefsItem.connect('activate', () => this._extension.openPreferences());
        this.menu.addMenuItem(prefsItem);
    }

    setState(state, available) {
        const listening = Boolean(state?.listening);
        const stopping = Boolean(state?.stopping);
        this._icon.icon_name = listening || stopping
            ? 'media-record-symbolic'
            : 'audio-input-microphone-symbolic';
        this._toggleItem.label.text = listening || stopping
            ? _('Stop Dictation')
            : _('Start Dictation');
        this._statusItem.label.text = available
            ? listening ? _('Listening') : stopping ? _('Stopping') : _('Ready')
            : _('Service unavailable');
    }

    setProfiles(profiles, selectedProfile) {
        for (const item of this._profileItems)
            item.destroy();
        this._profileItems = [];

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
                const missingItem = new PopupMenu.PopupMenuItem(
                    `${profile.title} ${_('not installed')}`,
                    {reactive: false});
                this._profileSection.addMenuItem(missingItem);
                this._profileItems.push(missingItem);

                const installItem = new PopupMenu.PopupMenuItem(
                    `${_('Install')} ${profile.title}`);
                installItem.connect('activate', () => this._extension.installModel(profile.id));
                this._profileSection.addMenuItem(installItem);
                this._profileItems.push(installItem);
            }
        }
    }
});

class DictationOverlay {
    constructor() {
        this._box = new St.BoxLayout({
            style_class: 'wordpipe-overlay',
            vertical: true,
            visible: false,
        });
        this._label = new St.Label({
            style_class: 'wordpipe-overlay-label',
            text: _('Wordpipe listening'),
            x_align: Clutter.ActorAlign.CENTER,
        });
        this._subtitle = new St.Label({
            style_class: 'wordpipe-overlay-subtitle',
            text: '',
            x_align: Clutter.ActorAlign.CENTER,
        });
        this._box.add_child(this._label);
        this._box.add_child(this._subtitle);
        Main.uiGroup.add_child(this._box);
        this._reposition();
        this._monitorSignalId = Main.layoutManager.connect(
            'monitors-changed', () => this._reposition());
    }

    destroy() {
        if (this._monitorSignalId) {
            Main.layoutManager.disconnect(this._monitorSignalId);
            this._monitorSignalId = 0;
        }
        this._box.destroy();
    }

    setVisible(visible) {
        this._box.visible = visible;
        if (visible)
            this._reposition();
    }

    setSubtitle(text) {
        this._subtitle.text = text;
    }

    _reposition() {
        const monitor = Main.layoutManager.primaryMonitor;
        this._box.set_position(
            Math.floor(monitor.x + monitor.width / 2 - this._box.width / 2),
            Math.floor(monitor.y + monitor.height - 150));
    }
}

class TextInjector {
    constructor() {
        this._lastSession = 0;
        this._lastSeq = 0;
        this._insertedText = '';
        this._inputMethod = null;
    }

    reset(sessionId) {
        this._lastSession = Number(sessionId);
        this._lastSeq = 0;
        this._insertedText = '';
    }

    insertDelta(sessionId, seq, text) {
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

        this._overlay = new DictationOverlay();
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

        this._overlay?.destroy();
        this._overlay = null;
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
                this._overlay?.setSubtitle(_('Starting stream'));
            }));
        this._signalIds.push(this._proxy.connectSignal('TextDelta',
            (_proxy, _sender, [sessionId, seq, text]) => {
                if (this._settings.get_boolean('insert-partials'))
                    this._injector.insertDelta(sessionId, seq, text);
                this._overlay?.setSubtitle(text);
            }));
        this._signalIds.push(this._proxy.connectSignal('Commit',
            (_proxy, _sender, [sessionId, seq, text]) => {
                this._injector.insertCommit(sessionId, seq, text);
                this._overlay?.setSubtitle(text);
            }));
        this._signalIds.push(this._proxy.connectSignal('SessionStopped',
            () => this._overlay?.setSubtitle(_('Stopped'))));
        this._signalIds.push(this._proxy.connectSignal('InstallProgress',
            (_proxy, _sender, [profile, progress]) => {
                const values = deepUnpackMap(progress);
                this._overlay?.setSubtitle(`${profile}: ${values.message ?? values.phase ?? ''}`);
                if (values.phase === 'complete' || values.phase === 'error')
                    this._refreshProfiles();
            }));
        this._signalIds.push(this._proxy.connectSignal('Error',
            (_proxy, _sender, [message]) => {
                this._overlay?.setSubtitle(message);
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
                this._settings.set_boolean('show-overlay', config.show_overlay);
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
        const shortcuts = this._settings.get_strv('toggle-shortcut');
        const shortcut = shortcuts.length > 0 ? shortcuts[0] : '';
        const options = {
            spoken_punctuation: new GLib.Variant('b',
                this._settings.get_boolean('spoken-punctuation')),
            insert_partials: new GLib.Variant('b',
                this._settings.get_boolean('insert-partials')),
            stream_insert_delay_ms: new GLib.Variant('u',
                this._settings.get_uint('stream-insert-delay-ms')),
            show_overlay: new GLib.Variant('b',
                this._settings.get_boolean('show-overlay')),
        };

        this._callRemote('SetBackend', backend);
        this._callRemote('SetModelProfile', profile);
        this._callRemote('SetInputDevice', inputDevice);
        this._callRemote('SetShortcut', shortcut);
        this._callRemote('SetInsertionOptions', options);
        this._callRemote('SetRuntimeOptions', {
            model_root: new GLib.Variant('s', modelRoot),
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
                this._setAvailable(false);
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
        const visible = Boolean(state.listening || state.stopping || state.loading_model) &&
            this._settings.get_boolean('show-overlay');
        this._overlay?.setVisible(visible);
        if (visible) {
            let subtitle = state.model_profile ?? '';
            if (state.loading_model)
                subtitle = _('Loading model');
            else if (state.stopping)
                subtitle = _('Stopping');
            this._overlay?.setSubtitle(subtitle);
        }
    }

    _setAvailable(available) {
        this._indicator?.setState(this._state, available);
        if (!available)
            this._overlay?.setVisible(false);
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
