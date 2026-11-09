#####################################################################
#                                                                   #
# /NI_DAQmx/blacs_tab.py                                            #
#                                                                   #
# Copyright 2018, Monash University, JQI, Christopher Billington    #
#                                                                   #
# This file is part of the module labscript_devices, in the         #
# labscript suite (see http://labscriptsuite.org), and is           #
# licensed under the Simplified BSD License. See the license.txt    #
# file in the root of the project for the full license.             #
#                                                                   #
#####################################################################
import labscript_utils.h5_lock
import h5py
import json
from labscript_utils import dedent

from blacs.device_base_class import DeviceTab
from blacs.tab_base_classes import define_state, MODE_MANUAL
from qtutils.qt import QtCore, QtWidgets
from .utils import split_conn_AO, split_conn_DO
from . import models
import warnings


DEFAULT_MANUAL_COUNTER_CONFIG = {
    'enabled': True,
    'sample_clock': 'connection_table',
    'rate': 10.0,
    'buffer_size': 1024,
    'polling_interval': 0.1,
}


def _style_counter_button(button):
    """Keep controls readable under BLACS's transparent dock/scroll styling."""
    button.setStyleSheet("""
        QPushButton, QToolButton {
            background-color: palette(button);
            color: palette(button-text);
            border: 1px solid palette(mid);
            border-radius: 3px;
            padding: 4px 10px;
        }
        QPushButton:hover, QToolButton:hover {
            background-color: palette(midlight);
        }
        QPushButton:pressed, QToolButton:pressed, QToolButton:checked {
            background-color: palette(highlight);
            color: palette(highlighted-text);
        }
        QPushButton:disabled, QToolButton:disabled {
            background-color: palette(window);
            color: palette(text);
            border: 1px dashed palette(mid);
        }
        QPushButton:focus, QToolButton:focus {
            border: 1px solid palette(highlight);
        }
    """)


def _ensure_str(value):
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "decode"):
        return value.decode()
    return str(value)


def _row_get(row, name, default):
    if name not in row.dtype.names:
        return default
    return row[name]


def _read_counter_declarations(group):
    if group is None or 'COUNTERS' not in group:
        return []
    counters = []
    for row in group['COUNTERS'][:]:
        if 'monitor' not in row.dtype.names:
            raise RuntimeError('Recompile the connection table with ChannelMonitor support')
        manual_config = dict(DEFAULT_MANUAL_COUNTER_CONFIG)
        manual_config.update(
            {
                'enabled': bool(
                    _row_get(row, 'manual enabled', manual_config['enabled'])
                ),
                'sample_clock': _ensure_str(
                    _row_get(
                        row,
                        'manual sample clock',
                        manual_config['sample_clock'],
                    )
                ),
                'rate': float(_row_get(row, 'manual rate', manual_config['rate'])),
                'buffer_size': int(
                    _row_get(row, 'manual buffer size', manual_config['buffer_size'])
                ),
                'polling_interval': float(
                    _row_get(
                        row,
                        'manual polling interval',
                        manual_config['polling_interval'],
                    )
                ),
            }
        )
        counters.append(
            {
                'name': _ensure_str(row['name']),
                'connection': _ensure_str(row['connection']),
                'edge': _ensure_str(row['edge terminal']),
                'gate': _ensure_str(row['gate terminal']),
                'sample': _ensure_str(row['sample terminal']),
                'manual': manual_config,
                'monitor': json.loads(_ensure_str(_row_get(row, 'monitor', 'null'))),
            }
        )
    return counters


class NI_DAQmxTab(DeviceTab):
    def initialise_GUI(self):
        # Get capabilities from connection table properties:
        connection_table = self.settings['connection_table']
        properties = connection_table.find_by_name(self.device_name).properties

        # Raise an error on old connection tables, since we are not backward compatible
        # with them. In terms of old shot files being added to the queue once BLACS has
        # started, BLACS will reject them as being incomatible since the connection
        # table properties won't match. So we do not need to add a check for that.
        version = properties.get('__version__', None)
        if version is None:
            msg = """Connection table was compiled with the old version of the NI_DAQmx
                device class. The new BLACS tab is not backward compatible with old shot
                files (including connection tables). Either downgrade labscript_devices
                to 2.4.0 or less, or recompile the connection table with
                labscript_devices 2.5.0 or greater.
                """
            raise RuntimeError(dedent(msg))

        num_AO = properties['num_AO']
        num_AI = properties['num_AI']
        try:
            AI_chans = properties['AI_chans']
        except KeyError:
            # new code being run on older model specification file
            # assume legacy behavior, warn user to update
            AI_chans = [f'ai{i:d}' for i in range(num_AI)]
            msg = """Connection table was compiled with old model specifications for {0}.
                     Please recompile the connection table.
                  """
            warnings.warn(dedent(msg.format(properties['MAX_name'])), FutureWarning)

        ports = properties['ports']
        num_CI = properties['num_CI']

        AO_base_units = 'V'
        if num_AO > 0:
            AO_base_min, AO_base_max = properties['AO_range']
        else:
            AO_base_min, AO_base_max = None, None
        AO_base_step = 0.1
        AO_base_decimals = 3

        clock_terminal = properties['clock_terminal']
        clock_mirror_terminal = properties['clock_mirror_terminal']
        # get to avoid error on older connection tables
        connected_terminals = properties.get('connected_terminals', None)
        static_AO = properties['static_AO']
        static_DO = properties['static_DO']
        clock_limit = properties['clock_limit']
        min_semiperiod_measurement = properties['min_semiperiod_measurement']

        # And the Measurement and Automation Explorer (MAX) name we will need to
        # communicate with the device:
        self.MAX_name = properties['MAX_name']
        with h5py.File(connection_table.filepath, 'r') as f:
            waits = f['waits']
            wait_acq_device = waits.attrs['wait_monitor_acquisition_device']
            wait_acq_connection = waits.attrs['wait_monitor_acquisition_connection']
            wait_timeout_device = waits.attrs['wait_monitor_timeout_device']
            wait_timeout_connection = waits.attrs['wait_monitor_timeout_connection']
            try:
                timeout_trigger_type = waits.attrs['wait_monitor_timeout_trigger_type']
            except KeyError:
                timeout_trigger_type = 'rising'

        # Check hardware lines connected to PFI ports used as counter inputs.
        counter_reserved_dio_lines = []
        counter_declarations = []
        try:
            with h5py.File(connection_table.filepath, 'r') as f:
                group = f.get(f"devices/{self.device_name}")
                counter_declarations = _read_counter_declarations(group)
                if group and 'CI' in group:
                    ci_attrs = group['CI'].attrs
                    if 'counter_reserved_dio_lines' in ci_attrs:
                        counter_reserved_dio_lines = [
                            _ensure_str(line)
                            for line in ci_attrs['counter_reserved_dio_lines']
                        ]
                    elif 'used_pfies' in ci_attrs:
                        # Backward compatibility for connection tables compiled while
                        # counter support only recorded PFI terminals.
                        counter_reserved_dio_lines = [
                            properties['PFI_connections'][_ensure_str(pfi)]
                            for pfi in ci_attrs['used_pfies']
                        ]
        except (KeyError, OSError):
            raise

        # Create output objects:
        AO_prop = {}
        for i in range(num_AO):
            AO_prop['ao%d' % i] = {
                'base_unit': AO_base_units,
                'min': AO_base_min,
                'max': AO_base_max,
                'step': AO_base_step,
                'decimals': AO_base_decimals,
            }

        DO_proplist = []
        DO_hardware_names = []
        for port_num in range(len(ports)):
            port_str ='port%d' % port_num
            port_props = {}
            for line in range(ports[port_str]['num_lines']):
                hardware_name = 'port%d/line%d' % (port_num, line)
                if hardware_name in counter_reserved_dio_lines:
                    continue
                port_props[hardware_name] = {}
                DO_hardware_names.append(hardware_name)
            DO_proplist.append((port_str, port_props))

        # Create the output objects
        self.create_analog_outputs(AO_prop)

        # Create widgets for outputs defined so far (i.e. analog outputs only)
        _, AO_widgets, _ = self.auto_create_widgets()

        # now create the digital output objects one port at a time
        for _, DO_prop in DO_proplist:
            self.create_digital_outputs(DO_prop)

        # Manually create the digital output widgets so they are grouped separately
        DO_widgets_by_port = {}
        for port_str, DO_prop in DO_proplist:
            DO_widgets_by_port[port_str] = self.create_digital_widgets(DO_prop)

        # Auto place the widgets in the UI, specifying sort keys for ordering them:
        widget_list = [("Analog outputs", AO_widgets, split_conn_AO)]
        for port_num in range(len(ports)):
            port_str ='port%d' % port_num
            if port_str not in DO_widgets_by_port:
                continue
            DO_widgets = DO_widgets_by_port[port_str]
            name = "Digital outputs: %s" % port_str
            if ports[port_str]['supports_buffered']:
                name += ' (buffered)'
            else:
                name += ' (static)'
            widget_list.append((name, DO_widgets, split_conn_DO))
        self.auto_place_widgets(*widget_list)

        self.manual_counter_widgets = {}
        self.manual_counter_active = set()
        if counter_declarations:
            self._create_manual_counter_widgets(counter_declarations)

        # Create and set the primary worker
        self.create_worker(
            "main_worker",
            'labscript_devices.NI_DAQmx.blacs_workers.NI_DAQmxOutputWorker',
            {
                'MAX_name': self.MAX_name,
                'Vmin': AO_base_min,
                'Vmax': AO_base_max,
                'num_AO': num_AO,
                'ports': ports,
                'clock_limit': clock_limit,
                'clock_terminal': clock_terminal,
                'clock_mirror_terminal': clock_mirror_terminal,
                'static_AO': static_AO,
                'static_DO': static_DO,
                'DO_hardware_names': DO_hardware_names,
                'counter_reserved_dio_lines': counter_reserved_dio_lines,
                'wait_timeout_device': wait_timeout_device,
                'wait_timeout_connection': wait_timeout_connection,
                'wait_timeout_rearm_value': int(timeout_trigger_type == 'falling')
            },
        )
        self.primary_worker = "main_worker"

        if wait_acq_device == self.device_name:
            if wait_timeout_device:
                wait_timeout_device = connection_table.find_by_name(wait_timeout_device)
                wait_timeout_MAX_name = wait_timeout_device.properties['MAX_name']
            else:
                wait_timeout_MAX_name = None

            if num_CI == 0:
                msg = """Device cannot be the wait monitor acquisiiton device as it has
                    no counter inputs"""
                raise RuntimeError(dedent(msg))

            self.create_worker(
                "wait_monitor_worker",
                'labscript_devices.NI_DAQmx.blacs_workers.NI_DAQmxWaitMonitorWorker',
                {
                    'MAX_name': self.MAX_name,
                    'wait_acq_connection': wait_acq_connection,
                    'wait_timeout_MAX_name': wait_timeout_MAX_name,
                    'wait_timeout_connection': wait_timeout_connection,
                    'timeout_trigger_type': timeout_trigger_type,
                    'min_semiperiod_measurement': min_semiperiod_measurement,
                },
            )
            self.add_secondary_worker("wait_monitor_worker")

        # Only need an acquisition worker if we have analog inputs. It is important that
        # the acquisition worker is created after the wait monitor worker if there is
        # one, because the creation order determines the order that transition_to_manual
        # runs, and the acquisition processing requires processing that is done in the
        # wait monitor during transition_to_manual.
        if num_AI > 0:
            self.create_worker(
                "acquisition_worker",
                'labscript_devices.NI_DAQmx.blacs_workers.NI_DAQmxAcquisitionWorker',
                {
                    'MAX_name': self.MAX_name,
                    'num_AI': num_AI,
                    'AI_chans': AI_chans,
                    'AI_term': properties['AI_term'],
                    'AI_range': properties['AI_range'],
                    'AI_start_delay': properties['AI_start_delay'],
                    'AI_start_delay_ticks': properties['AI_start_delay_ticks'],
                    'AI_timebase_terminal': properties.get('AI_timebase_terminal',None),
                    'AI_timebase_rate': properties.get('AI_timebase_rate',None),
                    'clock_terminal': clock_terminal,
                },
            )
            self.add_secondary_worker("acquisition_worker")

        if num_CI > 0:
            self.create_worker(
                "counter_worker",
                'labscript_devices.NI_DAQmx.blacs_workers.NI_DAQmxCounterWorker',
                {
                    'MAX_name': self.MAX_name,
                    'declared_counters': counter_declarations,
                },
            )
            self.add_secondary_worker("counter_worker")

        # Set the capabilities of this device
        self.supports_remote_value_check(False)
        self.supports_smart_programming(False)

    def _append_counter_palette(self, name):
        # Place the manual counter panels in the same ToolPaletteGroup as the
        # analog and digital output palettes, so they get a matching
        # collapsible header and flowing grid layout. Imported here rather
        # than at module level as it requires a Qt library:
        from labscript_utils.qtwidgets.toolpalette import ToolPaletteGroup

        layout = self.get_tab_layout()
        for index in range(layout.count()):
            widget = layout.itemAt(index).widget()
            if widget is not None and isinstance(widget.layout(), ToolPaletteGroup):
                toolpalettegroup = widget.layout()
                break
        else:
            # No output palettes exist (device without AO/DO widgets):
            widget = QtWidgets.QWidget()
            toolpalettegroup = ToolPaletteGroup(widget)
            layout.addWidget(widget)
        if toolpalettegroup.has_palette(name):
            return toolpalettegroup.get_palette(name)
        return toolpalettegroup.append_new_palette(name)

    def _create_manual_counter_widgets(self, counter_declarations):
        toolpalette = self._append_counter_palette("Counter inputs")

        for counter in counter_declarations:
            config = counter['manual']
            if not config.get('enabled', True):
                continue
            panel = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(panel)
            grid.setContentsMargins(6, 6, 6, 6)
            title = QtWidgets.QLabel("%s\n%s" % (counter['connection'], counter['name']))
            title.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(title, 0, 0, 1, 2)

            start_button = QtWidgets.QPushButton("Start")
            stop_button = QtWidgets.QPushButton("Stop")
            _style_counter_button(start_button)
            _style_counter_button(stop_button)
            stop_button.setEnabled(False)

            formula = QtWidgets.QLabel()
            latest = QtWidgets.QLabel("Value: --")
            status = QtWidgets.QLabel("idle")

            widgets = {
                'counter': counter,
                'config': config,
                'start': start_button,
                'stop': stop_button,
                'formula': formula,
                'latest': latest,
                'status': status,
            }
            self.manual_counter_widgets[counter['name']] = widgets

            grid.addWidget(latest, 1, 0, 1, 2)
            grid.addWidget(start_button, 2, 0)
            grid.addWidget(stop_button, 2, 1)
            grid.addWidget(status, 3, 0, 1, 2)
            details_button = QtWidgets.QToolButton()
            details_button.setText("Details")
            details_button.setCheckable(True)
            _style_counter_button(details_button)
            grid.addWidget(details_button, 4, 0, 1, 2)
            details = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(details)
            form.setContentsMargins(0, 0, 0, 0)
            monitor = counter.get('monitor') or {}
            fast = monitor.get('fast')
            for label, text in [
                ("Edge terminal", counter['edge']),
                ("Gate terminal", counter['gate'] or "None"),
                ("Sample clock", counter['sample'] if config['sample_clock'] == 'connection_table'
                 else "%s, %.6g Hz" % (config['sample_clock'], config['rate'])),
                ("Buffer length", str(config['buffer_size'])),
                ("Fast logger", "%s:%s" % (fast['host'], fast['port']) if fast else "Off"),
                ("Metrics", "On" if monitor.get('metrics') is not None else "Off"),
            ]:
                form.addRow(label, QtWidgets.QLabel(text))
            form.addRow(formula)
            grid.addWidget(details, 5, 0, 1, 2)
            details.hide()
            def toggle_details(visible, details=details, panel=panel):
                details.setVisible(visible)
                panel.layout().activate()
                # ToolPalette caches row heights until explicitly laid out.
                toolpalette._layout_widgets(force_relayout=True)
            details_button.toggled.connect(toggle_details)

            start_button.clicked.connect(
                lambda _=None, name=counter['name']: self.start_manual_counter(name)
            )
            stop_button.clicked.connect(
                lambda _=None, name=counter['name']: self.stop_manual_counter(name)
            )
            self._update_counter_formula(counter['name'])
            toolpalette.addWidget(panel, True)

    def _manual_counter_config(self, counter_name):
        widgets = self.manual_counter_widgets[counter_name]
        return dict(widgets['config'])

    def _set_counter_running(self, counter_name, running):
        widgets = self.manual_counter_widgets[counter_name]
        widgets['start'].setEnabled(not running)
        widgets['stop'].setEnabled(running)

    def _update_counter_formula(self, counter_name):
        widgets = self.manual_counter_widgets[counter_name]
        config = widgets['config']
        sample_clock = config['sample_clock']
        if sample_clock == 'connection_table':
            text = "Count difference (external clock; timing unknown)"
        else:
            text = "Count rate = difference × %.6g Hz" % config['rate']
        widgets['formula'].setText(text)

    def _ensure_manual_counter_monitor(self):
        if len(self.manual_counter_active) == 1:
            self.statemachine_timeout_add(20, self.manual_counter_monitor)

    def _remove_manual_counter_monitor_if_idle(self):
        if not self.manual_counter_active:
            self.statemachine_timeout_remove(self.manual_counter_monitor)

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=False)
    def start_manual_counter(self, counter_name):
        widgets = self.manual_counter_widgets[counter_name]
        config = self._manual_counter_config(counter_name)
        try:
            result = yield (
                self.queue_work(
                    "counter_worker",
                    "start_manual_counter",
                    counter_name,
                    widgets['counter'],
                    config,
                )
            )
        except Exception as exc:
            widgets['status'].setText("start failed: %s" % exc)
            self._set_counter_running(counter_name, False)
            return
        self.manual_counter_active.add(counter_name)
        self._set_counter_running(counter_name, True)
        widgets['status'].setText(result.get('status', 'running'))
        self._ensure_manual_counter_monitor()

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=False)
    def stop_manual_counter(self, counter_name):
        widgets = self.manual_counter_widgets[counter_name]
        result = {'status': 'stopped'}
        try:
            result = yield (
                self.queue_work("counter_worker", "stop_manual_counter", counter_name)
            )
        finally:
            self.manual_counter_active.discard(counter_name)
            self._set_counter_running(counter_name, False)
            self._remove_manual_counter_monitor_if_idle()
        widgets['status'].setText(result.get('status', 'stopped'))

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=True)
    def manual_counter_monitor(self):
        active = list(self.manual_counter_active)
        if not active:
            yield (self.queue_work("counter_worker", "noop"))
            return
        for counter_name in active:
            widgets = self.manual_counter_widgets[counter_name]
            result = yield (
                self.queue_work("counter_worker", "fetch_manual_counter", counter_name)
            )
            if result.get('status') == 'stopped':
                # The worker-side counter is gone, e.g. it failed to restart
                # after a buffered shot:
                self.manual_counter_active.discard(counter_name)
                self._set_counter_running(counter_name, False)
                self._remove_manual_counter_monitor_if_idle()
                widgets['status'].setText('stopped')
                continue
            if result.get('values'):
                widgets['latest'].setText(
                    "%.6g %s" % (result['latest_value'], result.get('unit', 'counts'))
                )
            widgets['status'].setText(result.get('status', 'running'))
