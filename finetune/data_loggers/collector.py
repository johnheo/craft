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
#import xlsxwriter
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
