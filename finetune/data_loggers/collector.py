#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import contextlib
from functools import partial, reduce
import operator
import xlsxwriter
import enum
import yaml
import os
from sys import float_info
from collections import OrderedDict
from contextlib import contextmanager
import torch
from torchnet.meter import AverageValueMeter
import logging
from math import sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
import concurrent.futures

msglogger = logging.getLogger()

__all__ = ['SummaryActivationStatsCollector', 'RecordsActivationStatsCollector', 'QuantCalibrationStatsCollector',
           'ActivationHistogramsCollector', 'RawActivationsCollector', 'CollectorDirection',
           'collect_quant_stats', 'collect_histograms', 'collect_raw_outputs',
           'collector_context', 'collectors_context']


class CollectorDirection(enum.Enum):
    OUT = 0
    OFM = 0
    IN = 1
    IFM = 1
    IFMS = 1





class WeightedAverageValueMeter(AverageValueMeter):
    """
    A correction to torchnet's AverageValueMeter which doesn't implement
    std collection correctly by taking into account the batch size.
    """
    def add(self, value, n=1):
        self.sum += value*n
        if n <= 0:
            raise ValueError("Cannot use a non-positive weight for the running stat.")
        elif self.n == 0:
            self.mean = 0.0 + value  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + n * (value - self.mean_old) / float(self.n+n)
            self.m_s += n*(value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n + n - 1.0))
        self.var = self.std**2

        self.n += n


class SummaryActivationStatsCollector(ActivationStatsCollector):
    """This class collects activations statistical summaries.

    This Collector computes the mean of some statistic of the activation.  It is rather
    light-weight and quicker than collecting a record per activation.
    The statistic function is configured in the constructor.

    collector_direction - enum type: IN for IFMs, OUT for OFM
    inputs_consolidate_func is called on tuple of tensors, and returns a tensor.
    """
    def __init__(self, model, stat_name, summary_fn,
                 classes=(torch.nn.ReLU, torch.nn.ReLU6, torch.nn.LeakyReLU),
                 collector_direction=CollectorDirection.OUT,
                 inputs_consolidate_func=torch.cat):
        super(SummaryActivationStatsCollector, self).__init__(model, stat_name, classes)
        self.summary_fn = summary_fn
        self.collector_direction = collector_direction
        self.inputs_func = inputs_consolidate_func

    def _activation_stats_cb(self, module, inputs, output):
        """Record the activation sparsity of 'module'

        This is a callback from the forward() of 'module'.
        """
        feature_map = output if self.collector_direction == CollectorDirection.OUT else self.inputs_func(inputs)
        try:
            getattr(module, self.stat_name).add(self.summary_fn(feature_map.data), feature_map.data.numel())
        except RuntimeError as e:
            if "The expanded size of the tensor" in e.args[0]:
                raise ValueError("ActivationStatsCollector: a module ({} - {}) was encountered twice during model.apply().\n"
                                 "This is an indication that your model is using the same module instance, "
                                 "in multiple nodes in the graph.  This usually occurs with ReLU modules: \n"
                                 "For example in TorchVision's ResNet model, self.relu = nn.ReLU(inplace=True) is "
                                 "instantiated once, but used multiple times.  This is not permissible when using "
                                 "instances of ActivationStatsCollector.".
                                 format(module.distiller_name, type(module)))
            else:
                msglogger.info("Exception in _activation_stats_cb: {} {}".format(module.distiller_name, type(module)))
                raise

    def _start_counter(self, module):
        if not hasattr(module, self.stat_name):
            setattr(module, self.stat_name, WeightedAverageValueMeter())
            # Assign a name to this summary
            if hasattr(module, 'distiller_name'):
                getattr(module, self.stat_name).name = module.distiller_name
            else:
                getattr(module, self.stat_name).name = '_'.join((
                    module.__class__.__name__, str(id(module))))

    def _reset_counter(self, module):
        if hasattr(module, self.stat_name):
            getattr(module, self.stat_name).reset()

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if hasattr(module, self.stat_name):
            mean = getattr(module, self.stat_name).mean
            if isinstance(mean, torch.Tensor):
                mean = mean.tolist()
            activation_stats[getattr(module, self.stat_name).name] = mean

    def save(self, fname):
        """Save the stats to an Excel workbook"""
        if not fname.endswith('.xlsx'):
            fname = '.'.join([fname, 'xlsx'])
        with contextlib.suppress(OSError):
            os.remove(fname)

        def _add_worksheet(workbook, tab_name, record):
            try:
                worksheet = workbook.add_worksheet(tab_name)
            except xlsxwriter.exceptions.InvalidWorksheetName:
                worksheet = workbook.add_worksheet()

            col_names = []
            for col, (module_name, module_summary_data) in enumerate(record.items()):
                if not isinstance(module_summary_data, list):
                    module_summary_data = [module_summary_data]
                worksheet.write_column(1, col, module_summary_data)
                col_names.append(module_name)
            worksheet.write_row(0, 0, col_names)

        with xlsxwriter.Workbook(fname) as workbook:
            _add_worksheet(workbook, self.stat_name, self.value())

        return fname




class _QuantStatsRecord(object):
    @staticmethod
    def create_records_dict():
        records = OrderedDict()
        records['min'] = float_info.max
        records['max'] = -float_info.max
        for stat_name in ['avg_min', 'avg_max', 'mean', 'std', 'b']:
            records[stat_name] = 0
        records['shape'] = ''
        records['total_numel'] = 0
        return records

    def __init__(self):
        # We don't know the number of inputs at this stage so we defer records creation to the actual callback
        self.inputs = []
        self.output = self.create_records_dict()


def _verify_no_dataparallel(model):
    if torch.nn.DataParallel in [type(m) for m in model.modules()]:
        raise ValueError('Model contains DataParallel modules, which can cause inaccurate stats collection. '
                         'Either create a model without DataParallel modules, or call '
                         'distiller.utils.make_non_parallel_copy on the model before invoking the collector')





class RawActivationsCollector(ActivationStatsCollector):
    def __init__(self, model, classes=None):
        super(RawActivationsCollector, self).__init__(model, "raw_acts", classes)

        _verify_no_dataparallel(model)

    def _activation_stats_cb(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            if output.is_quantized:
                module.raw_outputs.append(output.dequantize())
            else:
                module.raw_outputs.append(output.cpu())

    def _start_counter(self, module):
        module.raw_outputs = []

    def _reset_counter(self, module):
        if hasattr(module, 'raw_outputs'):
            module.raw_outputs = []

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if not hasattr(module, 'raw_outputs'):
            return

        if isinstance(module.raw_outputs, list) and len(module.raw_outputs) > 0:
            module.raw_outputs = torch.stack(module.raw_outputs)
        activation_stats[module.distiller_name] = module.raw_outputs

    def save(self, dir_name):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for idx, (layer_name, raw_outputs) in enumerate(self.value().items()):
                idx_str = '{:03d}'.format(idx + 1)
                executor.submit(torch.save, raw_outputs, os.path.join(dir_name,
                                                                      '-'.join((idx_str, layer_name)) + '.pt'))

        return dir_name


def collect_quant_stats(model, test_fn, save_dir=None, classes=None, inplace_runtime_check=False,
                        disable_inplace_attrs=False, inplace_attr_names=('inplace',),
                        modules_to_collect=None):
    """
    Helper function for collecting quantization calibration statistics for a model using QuantCalibrationStatsCollector

    Args:
        model (nn.Module): The model for which to collect stats
        test_fn (function): Test/Evaluation function for the model. It must have an argument named 'model' that
          accepts the model. All other arguments should be set in advance (can be done using functools.partial), or
          they will be left with their default values.
        save_dir (str): Path to directory where stats YAML file will be saved. If None then YAML will not be saved
          to disk.
        classes (iterable): See QuantCalibrationStatsCollector
        inplace_runtime_check (bool): See QuantCalibrationStatsCollector
        disable_inplace_attrs (bool): See QuantCalibrationStatsCollector
        inplace_attr_names (iterable): See QuantCalibrationStatsCollector
        modules_to_collect (iterable): enable stats collection for a predefined modules (specified by names).
          if None - will track stats for all layers.

    Returns:
        Dictionary with quantization stats (see QuantCalibrationStatsCollector for a description of the dictionary
        contents)
    """
    msglogger.info('Collecting quantization calibration stats for model')
    quant_stats_collector = QuantCalibrationStatsCollector(model, classes=classes,
                                                           inplace_runtime_check=inplace_runtime_check,
                                                           disable_inplace_attrs=disable_inplace_attrs,
                                                           inplace_attr_names=inplace_attr_names)
    with collector_context(quant_stats_collector, modules_to_collect):
        msglogger.info('Pass 1: Collecting min, max, avg_min, avg_max, mean')
        test_fn(model=model)
        # Collect Laplace distribution stats:
        msglogger.info('Pass 2: Collecting b, std parameters')
        quant_stats_collector.start_second_pass()
        test_fn(model=model)
        quant_stats_collector.stop_second_pass()

    msglogger.info('Stats collection complete')
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'acts_quantization_stats.yaml')
        quant_stats_collector.save(save_path)
        msglogger.info('Stats saved to ' + save_path)

    return quant_stats_collector.value()


def collect_histograms(model, test_fn, save_dir=None, activation_stats=None,
                       classes=None, nbins=2048, save_hist_imgs=False, hist_imgs_ext='.svg'):
    """
    Helper function for collecting activation histograms for a model using ActivationsHistogramCollector.
    Will perform 2 passes - one to collect the required stats and another to collect the histograms. The first
    pass can be skipped by passing pre-calculated stats.

    Args:
        model (nn.Module): The model for which to collect histograms
        test_fn (function): Test/Evaluation function for the model. It must have an argument named 'model' that
          accepts the model. All other arguments should be set in advance (can be done using functools.partial), or
          they will be left with their default values.
        save_dir (str): Path to directory where histograms will be saved. If None then data will not be saved to disk.
        activation_stats (str / dict / None): Either a path to activation stats YAML file, or a dictionary containing
          the stats. The stats are expected to be in the same structure as generated by QuantCalibrationStatsCollector.
          If None, then a stats collection pass will be performed.
        classes: See ActivationsHistogramCollector
        nbins: See ActivationsHistogramCollector
        save_hist_imgs: See ActivationsHistogramCollector
        hist_imgs_ext: See ActivationsHistogramCollector

    Returns:
        Dictionary with histograms data (See ActivationsHistogramCollector for a description of the dictionary
        contents)
    """
    msglogger.info('Pass 1: Stats collection')
    if activation_stats is not None:
        msglogger.info('Pre-computed activation stats passed, skipping stats collection')
    else:
        activation_stats = collect_quant_stats(model, test_fn, save_dir=save_dir, classes=classes,
                                               inplace_runtime_check=True, disable_inplace_attrs=True)

    msglogger.info('Pass 2: Histograms generation')
    histogram_collector = ActivationHistogramsCollector(model, activation_stats, classes=classes, nbins=nbins,
                                                        save_hist_imgs=save_hist_imgs, hist_imgs_ext=hist_imgs_ext)
    with collector_context(histogram_collector):
        test_fn(model=model)
    msglogger.info('Histograms generation complete')
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'acts_histograms.pt')
        histogram_collector.save(save_path)
        msglogger.info("Histogram data saved to " + save_path)
        if save_hist_imgs:
            msglogger.info('Histogram images saved in ' + os.path.join(save_dir, 'histogram_imgs'))

    return histogram_collector.value()


def collect_raw_outputs(model, test_fn, save_dir=None, classes=None):
    msglogger.info('Collecting raw layer outputs for model')
    collector = RawActivationsCollector(model, classes=classes)
    with collector_context(collector):
        test_fn(model=model)
    msglogger.info('Outputs collection complete')
    if save_dir is not None:
        msglogger.info('Saving outputs to disk...')
        save_path = os.path.join(save_dir, 'raw_outputs')
        collector.save(save_path)
        msglogger.info('Outputs saved to ' + save_path)
    return collector.value()


@contextmanager
def collector_context(collector, modules_list=None):
    """A context manager for an activation collector"""
    if collector is not None:
        collector.reset().start(modules_list)
    yield collector
    if collector is not None:
        collector.stop()


@contextmanager
def collectors_context(collectors_dict):
    """A context manager for a dictionary of collectors"""
    if len(collectors_dict) == 0:
        yield collectors_dict
        return
    for collector in collectors_dict.values():
        collector.reset().start()
    yield collectors_dict
    for collector in collectors_dict.values():
        collector.stop()


class TrainingProgressCollector(object):
    def __init__(self, stats={}):
        super(TrainingProgressCollector, self).__init__()
        object.__setattr__(self, '_stats', stats)

    def __setattr__(self, name, value):
        stats = self.__dict__.get('_stats')
        stats[name] = value

    def __getattr__(self, name):
        if name in self.__dict__['_stats']:
            return self.__dict__['_stats'][name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def value(self):
        return self._stats
