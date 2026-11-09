"""Counter monitoring contract, without NI hardware."""
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from test_counter_pfi_reservations import (
    CounterStub, import_counter_worker, import_counter_tab,
)
from labscript import Counter, LabscriptError
from labscript_devices.NI_DAQmx.labscript_devices import NI_DAQmx
from pythonlib.online_monitor import ChannelMonitor, FastMonitor, MetricsMonitor, MonitorStreams


def test_declaration_roundtrip_and_legacy_rejection(tmp_path):
    device = object.__new__(NI_DAQmx)
    counter = CounterStub('photons', edge='PFI0', sample='PFI1')
    counter.monitor = ChannelMonitor(
        fast=FastMonitor('127.0.0.1', 9011), metrics=MetricsMonitor(source='PMT')
    ).to_dict()
    table = device._make_counter_declaration_table({'ctr0': counter})
    import_counter_tab()
    from labscript_devices.NI_DAQmx.blacs_tabs import _read_counter_declarations
    with h5py.File(tmp_path/'declarations.h5', 'w') as f:
        f.create_dataset('COUNTERS', data=table)
        readback = _read_counter_declarations(f)[0]
    assert readback['monitor'] == counter.monitor
    assert readback['name'] == 'photons'
    counter.manual_acquisition_defaults = {'udp_host': 'localhost'}
    with pytest.raises(LabscriptError, match='ChannelMonitor'):
        device._make_counter_declaration_table({'ctr0': counter})
    with pytest.raises(LabscriptError, match='ChannelMonitor'):
        Counter.configure_manual_acquisition(object.__new__(Counter), udp_port=9000)


def make_worker():
    worker = object.__new__(import_counter_worker())
    worker.logger = logging.getLogger('test.counter')
    return worker


def sinks():
    fast = Mock()
    fast.send.return_value = True
    client = Mock()
    streams = MonitorStreams(
        fast=fast, metrics=MetricsMonitor(source='PMT'),
        identity={'device_name': 'daq', 'channel': 'photons', 'mode': 'manual'},
        metrics_client_factory=lambda: client,
    )
    return streams, fast, client


def test_manual_rate_fans_out_same_values_and_honours_poll_interval():
    worker = make_worker()
    streams, fast, client = sinks()
    state = dict(ci_task=object(), buffer=np.array([100, 103, 109], dtype=np.uint32),
                 daq_status={}, last_count=None, scale=10., streams=streams,
                 sample_clock_rate=10., last_fetch=None, fetch_interval=100.)
    worker.manual_counters = {'photons': state}
    worker._read_data = Mock(return_value=3)
    import PyDAQmx
    PyDAQmx.DAQmxConstants.DAQmx_Val_Auto = -1
    result = worker.fetch_manual_counter('photons')
    assert result['values'] == [30., 60.]
    assert result['unit'] == 'counts/s'
    assert [c.args[0] for c in fast.send.call_args_list] == [30., 60.]
    assert [c.kwargs['fields']['value'] for c in client.send_point.call_args_list] == [30., 60.]
    worker.fetch_manual_counter('photons')
    assert worker._read_data.call_count == 1


@pytest.mark.parametrize('rate', [10., float('nan')])
def test_buffered_raw_and_rate_across_batches_and_rollover(tmp_path, rate):
    worker = make_worker()
    streams, fast, client = sinks()
    with h5py.File(tmp_path/'sidecar.h5', 'w') as f:
        worker.data_saver = [SimpleNamespace(get_open_file=lambda: f)]
        worker.instructions = [{'max_sampling_rate': rate}]
        worker.previous_counts = [None]
        worker.counter_streams = [streams]
        worker._save_counter_samples(0, np.array([2**32-2], dtype=np.uint32))
        worker._save_counter_samples(0, np.array([1, 5], dtype=np.uint32))
        assert list(f['data'][:]) == [2**32-2, 1, 5]
        if np.isfinite(rate):
            assert list(f['rate'][:]) == [30., 40.]
            assert f['rate'].attrs['sample_rate_hz'] == rate
            expected = [30., 40.]
        else:
            assert 'rate' not in f
            expected = [2**32-2, 1, 5]
        assert [c.args[0] for c in fast.send.call_args_list] == expected
        assert [c.kwargs['fields']['value'] for c in client.send_point.call_args_list] == expected


def test_monitor_failure_does_not_break_acquisition(tmp_path):
    worker = make_worker()
    streams, fast, client = sinks()
    client.send_point.side_effect = OSError('offline')
    with h5py.File(tmp_path/'sidecar.h5', 'w') as f:
        worker.data_saver = [SimpleNamespace(get_open_file=lambda: f)]
        worker.instructions = [{'max_sampling_rate': 10.}]
        worker.previous_counts = [None]
        worker.counter_streams = [streams]
        worker._save_counter_samples(0, np.array([2, 4], dtype=np.uint32))
        assert list(f['data'][:]) == [2, 4]
        assert list(f['rate'][:]) == [20.]
        assert streams.errors()['metrics'] == 'offline'


def test_stop_drains_remaining_samples():
    worker = make_worker()
    worker.tasklock = threading.RLock()
    worker.tasks = [Mock()]
    worker.thread_polling_data = [Mock()]
    worker.event_task_stopped = threading.Event()
    worker.data_buffer = [np.array([10, 12], dtype=np.uint32)]
    worker._read_data = Mock(side_effect=[2, 0, 0])
    worker._save_counter_samples = Mock()
    worker._stop_task()
    np.testing.assert_array_equal(worker._save_counter_samples.call_args.args[1], [10, 12])


@pytest.mark.parametrize('abort', [False, True])
def test_transition_closes_streams_and_merges_only_successful_shots(abort):
    worker = make_worker()
    saver, streams = Mock(), Mock()
    worker.buffered_mode = True
    worker.buffered_chans = ['ctr0']
    worker.data_saver = [saver]
    worker.daq_status = [{}]
    worker.counter_streams = [streams]
    worker._stop_task = Mock()
    worker._restore_manual_counters = Mock()
    assert worker.transition_to_manual(abort)
    streams.close.assert_called_once()
    saver.close_data_file.assert_called_once()
    if abort:
        saver.cleanup.assert_called_once()
        saver.merge_into_shot.assert_not_called()
    else:
        saver.merge_into_shot.assert_called_once_with(cleanup=True)
    worker._restore_manual_counters.assert_called_once()
