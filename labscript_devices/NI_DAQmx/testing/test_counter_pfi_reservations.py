import logging
import os
import sys
import tempfile
import types
import unittest
import threading

os.environ.setdefault('READTHEDOCS', 'True')

import numpy as np

from labscript import LabscriptError
from labscript_devices.NI_DAQmx.labscript_devices import NI_DAQmx


class CounterStub:
    def __init__(self, name='counter', edge=None, gate=None, sample=None):
        self.name = name
        self.edge_terminal = edge
        self.gate_terminal = gate
        self.sample_terminal = sample
        self.acquisitions = []


class TestCounterPFIReservations(unittest.TestCase):
    def make_device(self, PFI_connections=None):
        device = object.__new__(NI_DAQmx)
        device.name = 'DevTest'
        device.PFI_connections = PFI_connections or {}
        return device

    def test_counter_reserves_mapped_dio_line(self):
        device = self.make_device({'PFI0': 'port0/line0'})
        reserved = device._check_counter_dio_conflicts(
            {'port0/line1': object()},
            {'ctr0': CounterStub(edge='PFI0')},
        )
        self.assertEqual(reserved, {'port0/line0': 'PFI0'})

    def test_counter_conflicts_with_declared_digital_output(self):
        device = self.make_device({'PFI0': 'port0/line0'})
        with self.assertRaises(LabscriptError):
            device._check_counter_dio_conflicts(
                {'port0/line0': object()},
                {'ctr0': CounterStub(edge='PFI0')},
            )

    def test_missing_pfi_mapping_raises(self):
        device = self.make_device()
        with self.assertRaises(LabscriptError):
            device._check_counter_dio_conflicts(
                {},
                {'ctr0': CounterStub(edge='PFI0')},
            )

    def test_counter_declaration_table_does_not_require_acquire(self):
        device = self.make_device()
        table = device._make_counter_declaration_table(
            {
                'ctr0': CounterStub(
                    name='counts', edge='PFI0', gate=None, sample='PFI1'
                )
            }
        )
        self.assertEqual(len(table), 1)
        self.assertEqual(table['name'][0], b'counts')
        self.assertEqual(table['connection'][0], b'ctr0')
        self.assertEqual(table['edge terminal'][0], b'PFI0')
        self.assertEqual(table['gate terminal'][0], b'')
        self.assertEqual(table['sample terminal'][0], b'PFI1')
        self.assertTrue(table['manual enabled'][0])
        self.assertEqual(table['manual sample clock'][0], b'connection_table')
        self.assertEqual(table['manual rate'][0], 10.0)
        self.assertEqual(table['manual buffer size'][0], 1024)
        self.assertEqual(table['manual polling interval'][0], 0.1)

    def test_buffered_counter_acquisitions_cannot_duplicate_ctr(self):
        device = self.make_device()
        counter = CounterStub(name='counts', edge='PFI0', sample='PFI1')
        counter.acquisitions = [
            {
                'label': 'first',
                'max_sampling_rate': 10.0,
                'buffer_size': 10,
                'polling_interval': 0.1,
            },
            {
                'label': 'second',
                'max_sampling_rate': 10.0,
                'buffer_size': 10,
                'polling_interval': 0.1,
            },
        ]
        with self.assertRaises(LabscriptError):
            device._make_counter_input_table({'ctr0': counter})


def _stub_module(name):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _ensure_blacs_stub():
    try:
        import blacs.tab_base_classes  # noqa: F401
        import blacs.device_base_class  # noqa: F401
    except Exception:
        pass
    _stub_module('blacs')
    tab_base_classes = _stub_module('blacs.tab_base_classes')
    device_base_class = _stub_module('blacs.device_base_class')

    if not hasattr(tab_base_classes, 'Worker'):
        class Worker:
            pass

        tab_base_classes.Worker = Worker
    if not hasattr(tab_base_classes, 'MODE_MANUAL'):
        tab_base_classes.MODE_MANUAL = 1
    if not hasattr(tab_base_classes, 'define_state'):
        def define_state(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

        tab_base_classes.define_state = define_state
    if not hasattr(device_base_class, 'DeviceTab'):
        class DeviceTab:
            pass

        device_base_class.DeviceTab = DeviceTab


def _ensure_pydaqmx_stub():
    try:
        import PyDAQmx  # noqa: F401
        return
    except Exception:
        pass
    pydaqmx = _stub_module('PyDAQmx')
    for submodule_name in ('DAQmxConstants', 'DAQmxTypes', 'DAQmxCallBack'):
        submodule = _stub_module('PyDAQmx.' + submodule_name)
        setattr(pydaqmx, submodule_name, submodule)


def _ensure_qtutils_stub():
    try:
        from qtutils.qt import QtWidgets  # noqa: F401
        return
    except Exception:
        pass
    _stub_module('qtutils')
    qtutils_qt = _stub_module('qtutils.qt')
    if not hasattr(qtutils_qt, 'QtWidgets'):
        qtutils_qt.QtWidgets = object()


def import_counter_worker():
    _ensure_pydaqmx_stub()
    _ensure_blacs_stub()

    from labscript_devices.NI_DAQmx.blacs_workers import NI_DAQmxCounterWorker

    return NI_DAQmxCounterWorker


def import_counter_tab():
    _ensure_blacs_stub()
    _ensure_qtutils_stub()

    from labscript_devices.NI_DAQmx.blacs_tabs import NI_DAQmxTab

    import inspect
    for name in ('manual_counter_monitor',):
        method = getattr(NI_DAQmxTab, name)
        original = inspect.getclosurevars(method).nonlocals.get('function')
        if original is not None:
            setattr(NI_DAQmxTab, name, original)
    return NI_DAQmxTab


class TestManualCounterProcessing(unittest.TestCase):
    def test_manual_counter_values_wait_for_two_counts_without_previous(self):
        worker_class = import_counter_worker()
        worker = object.__new__(worker_class)
        values, previous = worker._manual_counter_values(
            np.array([5], dtype=np.uint32), None, 10.0
        )
        self.assertEqual(values.tolist(), [])
        self.assertEqual(previous, 5)

    def test_manual_counter_values_use_previous_count_and_scale(self):
        worker_class = import_counter_worker()
        worker = object.__new__(worker_class)
        values, previous = worker._manual_counter_values(
            np.array([8, 11], dtype=np.uint32), 5, 20.0
        )
        self.assertEqual(values.tolist(), [60.0, 60.0])
        self.assertEqual(previous, 11)

    def test_manual_counter_start_rejects_active_sample_clock_as_input(self):
        worker_class = import_counter_worker()
        worker = object.__new__(worker_class)
        worker.tasklock = threading.RLock()
        worker.manual_counters = {
            'existing': {
                'counter': {'connection': 'ctr1'},
                'sample_clock': 'connection_table',
            }
        }
        with self.assertRaises(RuntimeError):
            worker.start_manual_counter(
                'new',
                {'connection': 'ctr0', 'edge': 'PFI0', 'gate': '', 'sample': ''},
                {
                    'sample_clock': 'ctr1',
                    'rate': 10.0,
                    'buffer_size': 1024,
                    'polling_interval': 0.1,
                    'multiplication_factor': 1.0,
                    'average_count': 10,
                    'udp_host': '127.0.0.1',
                    'udp_port': 9000,
                },
            )

    def test_manual_counter_start_rejects_active_sample_clock_as_output(self):
        worker_class = import_counter_worker()
        worker = object.__new__(worker_class)
        worker.tasklock = threading.RLock()
        worker.manual_counters = {
            'existing': {
                'counter': {'connection': 'ctr0'},
                'sample_clock': 'ctr1',
            }
        }
        with self.assertRaises(RuntimeError):
            worker.start_manual_counter(
                'new',
                {'connection': 'ctr1', 'edge': 'PFI0', 'gate': '', 'sample': 'PFI1'},
                {
                    'sample_clock': 'connection_table',
                    'rate': 10.0,
                    'buffer_size': 1024,
                    'polling_interval': 0.1,
                    'multiplication_factor': 1.0,
                    'average_count': 10,
                    'udp_host': '127.0.0.1',
                    'udp_port': 9000,
                },
            )

    def test_manual_counter_monitor_yields_when_no_active_counters(self):
        tab_class = import_counter_tab()
        tab = object.__new__(tab_class)
        tab.manual_counter_active = set()
        calls = []

        def queue_work(worker, method, *args):
            calls.append((worker, method, args))
            return calls[-1]

        tab.queue_work = queue_work
        generator = tab.manual_counter_monitor()

        self.assertEqual(next(generator), ("counter_worker", "noop", ()))
        self.assertEqual(calls, [("counter_worker", "noop", ())])
        with self.assertRaises(StopIteration):
            next(generator)

    def test_manual_counter_monitor_handles_baseline_only_read(self):
        tab_class = import_counter_tab()
        tab = object.__new__(tab_class)
        tab.manual_counter_active = {'counts'}
        texts = {}

        class Label:
            def __init__(self, name):
                self.name = name

            def setText(self, text):
                texts[self.name] = text

        tab.manual_counter_widgets = {
            'counts': {
                'latest': Label('latest'),
                'moving_average': Label('moving_average'),
                'status': Label('status'),
            }
        }

        def queue_work(worker, method, *args):
            return {'status': 'running', 'samples_read': 1}

        tab.queue_work = queue_work
        generator = tab.manual_counter_monitor()

        self.assertEqual(next(generator), {'status': 'running', 'samples_read': 1})
        with self.assertRaises(StopIteration):
            generator.send({'status': 'running', 'samples_read': 1})
        self.assertEqual(texts, {'status': 'running'})


class TestManualCounterSuspendResume(unittest.TestCase):
    def test_transition_to_buffered_suspends_manual_counters(self):
        worker_class = import_counter_worker()
        import labscript_utils.h5_lock
        import h5py

        worker = object.__new__(worker_class)
        worker.logger = logging.getLogger('test')
        worker.tasklock = threading.RLock()
        worker.manual_counters = {
            'counts': {
                'counter': {'connection': 'ctr0'},
                'config': {'rate': 10.0},
            },
        }
        stopped = []

        def stop_manual_counter(counter_name):
            stopped.append(counter_name)
            worker.manual_counters.pop(counter_name, None)

        worker.stop_manual_counter = stop_manual_counter

        with tempfile.TemporaryDirectory() as tmpdir:
            h5file = os.path.join(tmpdir, 'shot.h5')
            with h5py.File(h5file, 'w') as f:
                f.create_group('/devices/DevTest')
            worker.transition_to_buffered('DevTest', h5file, {}, False)

        self.assertEqual(stopped, ['counts'])
        self.assertEqual(
            worker.suspended_manual_counters,
            [('counts', {'connection': 'ctr0'}, {'rate': 10.0})],
        )

    def test_transition_to_manual_restarts_suspended_counters(self):
        worker_class = import_counter_worker()
        worker = object.__new__(worker_class)
        worker.logger = logging.getLogger('test')
        worker.buffered_mode = False
        worker.suspended_manual_counters = [
            ('counts', {'connection': 'ctr0'}, {'rate': 10.0}),
        ]
        started = []

        def start_manual_counter(counter_name, counter, config):
            started.append((counter_name, counter, config))
            return {'status': 'running'}

        worker.start_manual_counter = start_manual_counter

        self.assertTrue(worker.transition_to_manual())
        self.assertEqual(
            started, [('counts', {'connection': 'ctr0'}, {'rate': 10.0})]
        )
        self.assertEqual(worker.suspended_manual_counters, [])

    def test_transition_to_manual_logs_failed_restarts(self):
        worker_class = import_counter_worker()
        worker = object.__new__(worker_class)
        worker.logger = logging.getLogger('test')
        worker.buffered_mode = False
        worker.suspended_manual_counters = [
            ('counts', {'connection': 'ctr0'}, {'rate': 10.0}),
        ]

        def start_manual_counter(counter_name, counter, config):
            raise RuntimeError('resource in use')

        worker.start_manual_counter = start_manual_counter

        self.assertTrue(worker.transition_to_manual())
        self.assertEqual(worker.suspended_manual_counters, [])

    def test_manual_counter_monitor_syncs_stopped_counter(self):
        tab_class = import_counter_tab()
        tab = object.__new__(tab_class)
        tab.manual_counter_active = {'counts'}
        texts = {}
        enabled = {}
        removed = []

        class Label:
            def __init__(self, name):
                self.name = name

            def setText(self, text):
                texts[self.name] = text

        class Button:
            def __init__(self, name):
                self.name = name

            def setEnabled(self, state):
                enabled[self.name] = state

        tab.manual_counter_widgets = {
            'counts': {
                'latest': Label('latest'),
                'moving_average': Label('moving_average'),
                'status': Label('status'),
                'start': Button('start'),
                'stop': Button('stop'),
            }
        }
        tab.statemachine_timeout_remove = lambda func: removed.append(func)

        def queue_work(worker, method, *args):
            return {'status': 'stopped', 'samples_read': 0}

        tab.queue_work = queue_work
        generator = tab.manual_counter_monitor()

        next(generator)
        with self.assertRaises(StopIteration):
            generator.send({'status': 'stopped', 'samples_read': 0})
        self.assertEqual(tab.manual_counter_active, set())
        self.assertEqual(texts, {'status': 'stopped'})
        self.assertEqual(enabled, {'start': True, 'stop': False})
        self.assertEqual(removed, [tab.manual_counter_monitor])


if __name__ == '__main__':
    unittest.main()
