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

        this._installFastItem = new PopupMenu.PopupMenuItem(_('Install Fast Model'));
        this._installFastItem.connect('activate', () => this._extension.installModel('fast'));
        this.menu.addMenuItem(this._installFastItem);

        this._installCompactItem = new PopupMenu.PopupMenuItem(_('Install Compact Model'));
        this._installCompactItem.connect('activate', () => this._extension.installModel('compact'));
        this.menu.addMenuItem(this._installCompactItem);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        const prefsItem = new PopupMenu.PopupMenuItem(_('Preferences'));
        prefsItem.connect('activate', () => this._extension.openPreferences());
        this.menu.addMenuItem(prefsItem);
    }

    setState(state, available) {
        const listening = Boolean(state?.listening);
        this._icon.icon_name = listening
            ? 'media-record-symbolic'
            : 'audio-input-microphone-symbolic';
        this._toggleItem.label.text = listening ? _('Stop Dictation') : _('Start Dictation');
        this._statusItem.label.text = available
            ? listening ? _('Listening') : _('Ready')
            : _('Service unavailable');
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
    }

    reset(sessionId) {
        this._lastSession = Number(sessionId);
        this._lastSeq = 0;
    }

    insertDelta(sessionId, seq, text) {
        const numericSession = Number(sessionId);
        const numericSeq = Number(seq);
        if (numericSession !== this._lastSession)
            this.reset(numericSession);
        if (numericSeq <= this._lastSeq || !text)
            return;
        this._lastSeq = numericSeq;

        // GNOME Shell's OSK/text-insertion API is internal and version-specific.
        // Keep the boundary narrow so this method can become the Shell 50 injector.
        log(`Wordpipe text delta: ${text}`);
    }
}

export default class WordpipeExtension extends Extension {
    enable() {
        this._settings = this.getSettings();
        this._state = {};
        this._signalIds = [];
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

    _bindShortcut() {
        Main.wm.addKeybinding(
            'toggle-shortcut',
            this._settings,
            Meta.KeyBindingFlags.IGNORE_AUTOREPEAT,
            Shell.ActionMode.NORMAL | Shell.ActionMode.OVERVIEW,
            () => this.toggleDictation());
    }

    _connectSettings() {
        this._settings.connectObject('changed', () => this._pushSettings(), this);
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
                this._pushSettings();
            });
    }

    _subscribeSignals() {
        this._signalIds.push(this._proxy.connectSignal('StateChanged',
            (_proxy, _sender, [state]) => this._handleState(deepUnpackMap(state))));
        this._signalIds.push(this._proxy.connectSignal('ConfigChanged',
            () => this._refreshState()));
        this._signalIds.push(this._proxy.connectSignal('SessionStarted',
            (_proxy, _sender, [sessionId]) => {
                this._injector.reset(sessionId);
                this._overlay?.setSubtitle(_('Starting stream'));
            }));
        this._signalIds.push(this._proxy.connectSignal('TextDelta',
            (_proxy, _sender, [sessionId, seq, text]) => {
                this._injector.insertDelta(sessionId, seq, text);
                this._overlay?.setSubtitle(text);
            }));
        this._signalIds.push(this._proxy.connectSignal('Commit',
            (_proxy, _sender, [_sessionId, _seq, text]) => {
                this._overlay?.setSubtitle(text);
            }));
        this._signalIds.push(this._proxy.connectSignal('SessionStopped',
            () => this._overlay?.setSubtitle(_('Stopped'))));
        this._signalIds.push(this._proxy.connectSignal('InstallProgress',
            (_proxy, _sender, [profile, progress]) => {
                const values = deepUnpackMap(progress);
                this._overlay?.setSubtitle(`${profile}: ${values.message ?? values.phase ?? ''}`);
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

    _pushSettings() {
        if (!this._proxy)
            return;

        const backend = this._settings.get_string('backend');
        const profile = this._settings.get_string('model-profile');
        const inputDevice = this._settings.get_string('input-device');
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
        const visible = Boolean(state.listening) && this._settings.get_boolean('show-overlay');
        this._overlay?.setVisible(visible);
        if (visible)
            this._overlay?.setSubtitle(state.model_profile ?? '');
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
