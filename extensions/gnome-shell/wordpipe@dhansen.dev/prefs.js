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
    <method name="GetState"><arg name="state" type="a{sv}" direction="out"/></method>
    <method name="GetConfig"><arg name="config" type="a{sv}" direction="out"/></method>
    <method name="ListInputDevices"><arg name="devices" type="aa{sv}" direction="out"/></method>
    <method name="SetBackend"><arg name="backend" type="s" direction="in"/></method>
    <method name="SetModelProfile"><arg name="profile" type="s" direction="in"/></method>
    <method name="SetInputDevice"><arg name="selector" type="s" direction="in"/></method>
    <method name="SetShortcut"><arg name="accelerator" type="s" direction="in"/></method>
    <method name="SetInsertionOptions"><arg name="options" type="a{sv}" direction="in"/></method>
    <method name="InstallModel"><arg name="profile" type="s" direction="in"/></method>
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
        this._deviceSelectors = [''];
        this._buildModelGroup();
        this._buildInputGroup();
        this._buildBehaviorGroup();
        this._buildServiceGroup();
        this._connectProxy();
    }

    _buildModelGroup() {
        const group = new Adw.PreferencesGroup({
            title: _('Model'),
        });
        this.add(group);

        const backendModel = new Gtk.StringList();
        BACKENDS.forEach(([_id, title]) => backendModel.append(title));
        const backendRow = new Adw.ComboRow({
            title: _('Backend'),
            model: backendModel,
        });
        backendRow.selected = Math.max(0, BACKENDS.findIndex(([id]) =>
            id === this._settings.get_string('backend')));
        backendRow.connect('notify::selected', row => {
            this._settings.set_string('backend', BACKENDS[row.selected][0]);
            this._callRemote('SetBackend', BACKENDS[row.selected][0]);
        });
        group.add(backendRow);

        const profileModel = new Gtk.StringList();
        PROFILES.forEach(([_id, title]) => profileModel.append(title));
        const profileRow = new Adw.ComboRow({
            title: _('Model Profile'),
            subtitle: _('Choose the speed, memory, and disk footprint tradeoff.'),
            model: profileModel,
        });
        profileRow.selected = Math.max(0, PROFILES.findIndex(([id]) =>
            id === this._settings.get_string('model-profile')));
        profileRow.connect('notify::selected', row => {
            const profile = PROFILES[row.selected][0];
            this._settings.set_string('model-profile', profile);
            this._callRemote('SetModelProfile', profile);
        });
        group.add(profileRow);

        for (const [id, title, subtitle] of PROFILES) {
            const row = new Adw.ActionRow({
                title: _(`Install ${title}`),
                subtitle,
            });
            const button = new Gtk.Button({
                icon_name: 'folder-download-symbolic',
                valign: Gtk.Align.CENTER,
                tooltip_text: _(`Download and export the ${title} model`),
            });
            button.connect('clicked', () => this._callRemote('InstallModel', id));
            row.add_suffix(button);
            group.add(row);
        }
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
            this._settings.set_boolean('spoken-punctuation', widget.active);
            this._pushInsertionOptions();
        });
        group.add(row);

        row = new Adw.SwitchRow({
            title: _('Stream Text Immediately'),
            active: this._settings.get_boolean('insert-partials'),
        });
        row.connect('notify::active', widget => {
            this._settings.set_boolean('insert-partials', widget.active);
            this._pushInsertionOptions();
        });
        group.add(row);

        row = new Adw.SwitchRow({
            title: _('Show Overlay'),
            active: this._settings.get_boolean('show-overlay'),
        });
        row.connect('notify::active', widget => {
            this._settings.set_boolean('show-overlay', widget.active);
            this._pushInsertionOptions();
        });
        group.add(row);

        const delayRow = Adw.SpinRow.new_with_range(0, 1000, 25);
        delayRow.title = _('Insertion Delay');
        delayRow.subtitle = _('Additional delay in milliseconds before inserting streamed text.');
        delayRow.value = this._settings.get_uint('stream-insert-delay-ms');
        delayRow.connect('notify::value', widget => {
            this._settings.set_uint('stream-insert-delay-ms', Math.round(widget.value));
            this._pushInsertionOptions();
        });
        group.add(delayRow);

        const shortcutRow = new Adw.EntryRow({
            title: _('Shortcut'),
            text: this._settings.get_strv('toggle-shortcut')[0] ?? '',
        });
        shortcutRow.connect('changed', row => {
            const accelerator = row.text.trim();
            this._settings.set_strv('toggle-shortcut', accelerator ? [accelerator] : []);
            this._callRemote('SetShortcut', accelerator);
        });
        group.add(shortcutRow);
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

        const actions = new Adw.ActionRow({
            title: _('Dictation'),
        });
        const startButton = new Gtk.Button({
            icon_name: 'media-record-symbolic',
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Start dictation'),
        });
        startButton.connect('clicked', () => this._callRemote('Start'));
        const stopButton = new Gtk.Button({
            icon_name: 'media-playback-stop-symbolic',
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Stop dictation'),
        });
        stopButton.connect('clicked', () => this._callRemote('Stop'));
        actions.add_suffix(startButton);
        actions.add_suffix(stopButton);
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
                this._refreshState();
                this._refreshInputDevices();
                this._pushInsertionOptions();
            });
    }

    _refreshState() {
        this._callRemote('GetState', state => {
            const values = deepUnpackMap(state);
            this._statusRow.subtitle = values.listening ? _('Listening') : _('Ready');
        });
    }

    _refreshInputDevices() {
        this._callRemote('ListInputDevices', devices => {
            while (this._deviceModel.get_n_items() > 0)
                this._deviceModel.remove(0);
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
            this._deviceRow.selected = selected;
        });
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
