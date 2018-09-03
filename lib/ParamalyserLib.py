#!/usr/bin/env python
# -*- coding: utf-8 -*-

# history:
# v1.1:   (skipped)
# v1.2:   config system changed, read_datafile() improved, nan-filtering, curve_no, step_no, more output
# v1.3:    modals, ...
# v1.4:    output_dir, general_output_dir improved
# v1.5:    plot_averages, filters, Igor compatibility, output of experiment-specific data
# v1.6:    curve selection, auto-splitting
# v1.6.1:  cumulated data, output improved
# v1.6.2:  fixed problems with 0 curves, histogram_ranges used to calculate modals, correlated plots use plot_ranges
# v1.6.3:  histogram_norm_*, avg_slope, legend location = best
# v1.6.4:  max_slope
# v1.6.5:  plot_params, baseline curvature
# v1.6.6:  ?
# v1.6.7:  adaption to new Curvalyser data structure, get_filter_limits, get_curvalyser_config, read_curvalyser_data, record arrays
# v1.6.8:  fixed_input_dir, fixed_output_dir, autosplit_overlap, first_paramalyser_curve, last_paramalyser_curve
# v1.6.9:  command line help
# v1.7:    first_curve -> curves_range_start, last_curve -> curves_range_stop, curves_range_step
# v1.7.1:  c['exp_id'], Bugfixes
# v1.7.2:  6 additional fit_curve() params
# v1.7.3:  bugfix for autosplit feature: curves -> all_curves, paramalyser_config_file, autosplit_overlap -> autosplit_shift, option -p(arams), option -s(ubsets), improvement for CDFs, parameter CDF_histogram_bins, parameter steps_ctr_averaging, bugfix for write_curve_data_cum() and write_step_data_cum()
# v1.8:    cumulation_mode
# v1.8.1:  Bugfix average()
# v1.8.2:  again Bugfix average()
# v1.8.3:  bugfixes for statistical tests
# v1.8.4:  write_averages_data, write_medians_data, write_modals_data, write_averages_data_cum, write_medians_data_cum, write_modals_data_cum
# v1.8.5:  bugfix for plot_errorbar_cum() in task 'other', len_by_exp_no_by_exp_type() removed
# v1.8.6:  histos_cum: handling of empty experiments, denoising_param2 added
# v1.8.7:  fit_indentation()
# v1.8.8:  data_cum: experiments.txt added (contains experiment-type-specific data)
# v1.8.9:  param_info added to write_averages_data() & co
# v1.8.10: get_curvalyser_config() doesn't return exp_id any more, Paramalyser.py: exp_id = os.path.basename(config_file)
# v1.8.11: scatter plots: step heights vs. extensions
# v1.9:    Adaptions for Curvalyser v1.9
# v1.9.1:  Mann-Whitney test improved (output file changed)
# v1.9.2:  file name in output files (curves) for tasks 'data' and 'data_cum'
# v1.10:   compatibility to Curvalyser v1.10 (lt_pos/rt_pos -> lt_edge/rt_edge)

__version__ = '1.10'

import math
import os
import string

import matplotlib.pyplot as pl
import numpy as np

from CurvalyserLib import read_config_file, curve_file_fields, step_file_fields, load_curvalyser_curve_data, \
    load_curvalyser_step_data

default_plot_size = (pl.rcParams['figure.figsize'][0] * pl.rcParams['savefig.dpi'],
                     pl.rcParams['figure.figsize'][1] * pl.rcParams['savefig.dpi'])


def get_curvalyser_config(config_file, curvalyser_config, get_experiment_type=None):
    c = read_config_file(config_file, {'get_experiment_type': get_experiment_type})
    c.update(curvalyser_config)
    return c


def read_curvalyser_data(c, config_file_no, config_file, exp_id, len_config_files, ignore_ids=[], exp_types=[],
                         analyse_all=True):
    if c['output_dir'] is None: c['output_dir'] = exp_id
    c['output_dir'] = c['base_output_dir'] + '/' + c['output_dir']
    if exp_types is None: exp_types = []
    if 'experiment_type' in c and c['experiment_type'] is not None:
        exp_type = c['experiment_type']
    else:
        exp_type = 'undefined'
    if exp_id in ignore_ids:
        print '%s (%.1f%%, exp. %d of %d, %s):\tskipped (excluded by ignore list)' % (
        exp_id, float(config_file_no) / len_config_files * 100., config_file_no + 1, len_config_files, exp_type)
        return None, None, exp_type, exp_types
    if exp_type not in exp_types:
        if analyse_all:
            exp_types.append(exp_type)
        else:
            print '%s (%.1f%%, exp. %d of %d, %s):\tskipped (excluded by experiment type)' % (
            exp_id, float(config_file_no) / len_config_files * 100., config_file_no + 1, len_config_files, exp_type)
            return None, None, exp_type, exp_types
    if not os.path.exists(c['output_dir']):
        print '%s (%.1f%%, exp. %d of %d, %s)' % (
        exp_id, float(config_file_no) / len_config_files * 100., config_file_no + 1, len_config_files, exp_type)
        print "[ERROR]   directory '" + c['output_dir'] + "' doesn't exist"
        return None, None, exp_type, exp_types
    data_curves = load_curvalyser_curve_data(c['output_dir'] + '/curves.txt')
    if data_curves is None:
        print '%s (%.1f%%, exp. %d of %d, %s)' % (
        exp_id, float(config_file_no) / len_config_files * 100., config_file_no + 1, len_config_files, exp_type)
        print "[warning] '" + c['output_dir'] + '/curves.txt' + "' is empty"
        return None, None, exp_type, exp_types
    data_steps = load_curvalyser_step_data(c['output_dir'] + '/steps.txt')
    if data_steps is None:
        print '%s (%.1f%%, exp. %d of %d, %s)' % (
        exp_id, float(config_file_no) / len_config_files * 100., config_file_no + 1, len_config_files, exp_type)
        print "[warning] '" + c['output_dir'] + '/steps.txt' + "' is empty"
        return None, None, exp_type, exp_types
    return data_curves, data_steps, exp_type, exp_types


def extract_data(all_curves, all_steps, start, stop):
    if start < 0:
        start += all_curves.size
    elif not start >= 0:
        start = 0
    if stop < 0:
        stop += all_curves.size
    elif not stop >= 0:
        stop = all_curves.size
    if start == 0 and stop == all_curves.size: return all_curves, all_steps
    curves = all_curves[start:stop]
    cond = (all_steps.curve_no >= start + 1) & (all_steps.curve_no <= stop)
    steps = all_steps[cond]
    if start > 0 and steps.size > 0: steps.curve_no -= start
    return curves, steps


def get_filter_limits(step_filter_limits, curve_filter_limits):
    step_filter_curve_limits = {}
    step_filter_step_limits = {}
    if step_filter_limits is not None:
        for param, dt in curve_file_fields:
            if param in step_filter_limits and step_filter_limits[param] is not None: step_filter_curve_limits[param] = \
            step_filter_limits[param]
        for param, dt in step_file_fields:
            if param in step_filter_limits and step_filter_limits[param] is not None: step_filter_step_limits[param] = \
            step_filter_limits[param]
    curve_filter_curve_limits = {}
    curve_filter_step_limits = {}
    if curve_filter_limits is not None:
        for param, dt in curve_file_fields:
            if param in curve_filter_limits and curve_filter_limits[param] is not None: curve_filter_curve_limits[
                param] = curve_filter_limits[param]
        for param, dt in step_file_fields:
            if param in curve_filter_limits and curve_filter_limits[param] is not None: curve_filter_step_limits[
                param] = curve_filter_limits[param]
    return step_filter_curve_limits, step_filter_step_limits, curve_filter_curve_limits, curve_filter_step_limits


def filter_steps(curves, steps, curve_limits, step_limits):
    curves_mask = np.ones(curves.size, dtype=bool)
    for param, (
    min, max) in curve_limits.iteritems():  # curve filters (-> remove all steps and update curves.steps_ctr)
        if min is not None: curves_mask &= curves[param] >= min
        if max is not None: curves_mask &= curves[param] <= max
    steps_mask = np.ones(steps.size, dtype=bool)
    for curve_no in (~curves_mask).nonzero()[
        0] + 1: steps_mask &= steps.curve_no != curve_no  # filter all steps of a curve if a curve filter applies
    for param, (min, max) in step_limits.iteritems():  # step filters (-> remove some steps and update curves.steps_ctr)
        if min is not None: steps_mask &= steps[param] >= min
        if max is not None: steps_mask &= steps[param] <= max
    changed_curves = np.unique(steps.curve_no[~steps_mask])
    steps = steps[steps_mask]  # remove steps filtered by steps filter
    for curve_no in changed_curves: curves.steps_ctr[curve_no - 1] = np.sum(
        steps.curve_no == curve_no)  # update curves.steps_ctr
    return curves, steps


def filter_curves(curves, steps, curve_limits, step_limits):
    curves_mask = np.ones(curves.size, dtype=bool)
    for param, (min, max) in curve_limits.iteritems():  # curve filters (-> remove curve AND all steps)
        if min is not None: curves_mask &= curves[param] >= min
        if max is not None: curves_mask &= curves[param] <= max
    skip_curves = []
    for param, (min, max) in step_limits.iteritems():  # step filters (-> remove curve AND all steps)
        if min is not None: skip_curves.extend(steps.curve_no[steps[param] < min])
        if max is not None: skip_curves.extend(steps.curve_no[steps[param] > max])
    if len(skip_curves): curves_mask[np.unique(skip_curves) - 1] = False  # filter curves if a step filter applies
    curves = curves[curves_mask]  # remove filtered curves
    steps_mask = np.ones(steps.size, dtype=bool)
    for curve_no in (~curves_mask).nonzero()[
        0] + 1: steps_mask &= steps.curve_no != curve_no  # filter steps whose curves have been removed
    steps = steps[steps_mask]  # remove filtered steps
    for i, curve_no in enumerate(curves_mask.nonzero()[0] + 1): steps.curve_no[
        steps.curve_no == curve_no] = i + 1  # update steps.curve_no
    return curves, steps


def finite_values(y):
    return y[np.isfinite(y)]


def average(y, ignore_zeros=False, weights=None):
    y = np.array(y)
    if weights is not None: weights = np.array(weights)
    if ignore_zeros:
        if weights is not None: weights = weights[y != 0]
        y = y[y != 0]
    if weights is not None: weights = weights[np.isfinite(y)]
    y = finite_values(y)
    if not y.size: return np.nan
    if weights is not None: y = y * weights * y.size / float(np.sum(weights))
    return np.average(y)


def median(y, ignore_zeros=False):
    if ignore_zeros: y = y[y != 0]
    y = finite_values(y)
    if not y.size: return np.nan
    return np.median(y)


def modal(y, hist_range=None, bins=None, ignore_zeros=False):
    if ignore_zeros: y = y[y != 0]
    y = finite_values(y)
    if not y.size: return np.nan
    values, counts = histogram(y, hist_range, bins)
    return values[np.argmax(counts)]


def stderr(y, ignore_zeros=False):
    if ignore_zeros: y = y[y != 0]
    y = finite_values(y)
    if not y.size: return np.nan
    return np.std(y) / math.sqrt(y.size)


def flatten_keys(keys, data):
    flat_keys = []
    offset = 0
    for exp_no in data:
        flat_keys.extend([i + offset for i in keys[exp_no]])
        offset += len(data[exp_no])
    return flat_keys


def get_averages(y, keys=None, mode=0, ignore_zeros=False, cum_averaging_fctn=median):
    averages = {}
    stderrs = {}
    for exp_type in y:
        averages[exp_type] = {}
        stderrs[exp_type] = {}
        if mode == 0:  # global averaging over all experiments of each type
            y_merged = np.concatenate(y[exp_type].values())
            if keys is None:
                averages[exp_type] = average(y_merged, ignore_zeros)
                stderrs[exp_type] = stderr(y_merged, ignore_zeros)
            else:
                flat_keys = flatten_keys(keys[exp_type], y[exp_type])
                averages[exp_type] = average(y_merged[flat_keys], ignore_zeros)
                stderrs[exp_type] = stderr(y_merged[flat_keys], ignore_zeros)
        else:  # averaging over values of each experiment
            if keys is None:
                for exp_no, values in y[exp_type].iteritems():
                    averages[exp_type][exp_no] = average(values, ignore_zeros)
                    stderrs[exp_type][exp_no] = stderr(values, ignore_zeros)
            else:
                for exp_no, values in y[exp_type].iteritems():
                    averages[exp_type][exp_no] = average(values[keys[exp_type][exp_no]], ignore_zeros)
                    stderrs[exp_type][exp_no] = stderr(values[keys[exp_type][exp_no]], ignore_zeros)
            if mode == 1:  # cum_averaging_fctn()'s of average values
                averages[exp_type] = cum_averaging_fctn(np.array(averages[exp_type].values()))
                stderrs[exp_type] = stderr(np.array(stderrs[exp_type].values()))
    return averages, stderrs


def get_medians(y, keys=None, mode=0, ignore_zeros=False, cum_averaging_fctn=median):
    medians = {}
    stderrs = {}
    for exp_type in y:
        medians[exp_type] = {}
        stderrs[exp_type] = {}
        if mode == 0:  # global averaging over all experiments of each type
            y_merged = np.concatenate(y[exp_type].values())
            if keys is None:
                medians[exp_type] = median(y_merged, ignore_zeros)
                stderrs[exp_type] = stderr(y_merged, ignore_zeros)
            else:
                flat_keys = flatten_keys(keys[exp_type], y[exp_type])
                medians[exp_type] = median(y_merged[flat_keys], ignore_zeros)
                stderrs[exp_type] = stderr(y_merged[flat_keys], ignore_zeros)
        else:  # averaging over values of each experiment
            if keys is None:
                for exp_no, values in y[exp_type].iteritems():
                    if not len(values): continue
                    medians[exp_type][exp_no] = median(values, ignore_zeros)
                    stderrs[exp_type][exp_no] = stderr(values, ignore_zeros)
            else:
                for exp_no, values in y[exp_type].iteritems():
                    if not len(values): continue
                    medians[exp_type][exp_no] = median(values[keys[exp_type][exp_no]], ignore_zeros)
                    stderrs[exp_type][exp_no] = stderr(values[keys[exp_type][exp_no]], ignore_zeros)
            if mode == 1:  # cum_averaging_fctn()'s of medians
                medians[exp_type] = cum_averaging_fctn(np.array(medians[exp_type].values()))
                stderrs[exp_type] = stderr(np.array(stderrs[exp_type].values()))
    return medians, stderrs


def get_modals(y, bins, hist_range=None, keys=None, mode=0, ignore_zeros=False, cum_averaging_fctn=median):
    modals = {}
    stderrs = {}
    for exp_type in y:
        modals[exp_type] = {}
        stderrs[exp_type] = {}
        if mode == 0:  # global averaging over all experiments of each type
            y_merged = np.concatenate(y[exp_type].values())
            if keys is None:
                modals[exp_type] = modal(y_merged, hist_range, bins, ignore_zeros)
                stderrs[exp_type] = stderr(y_merged, ignore_zeros)
            else:
                flat_keys = flatten_keys(keys[exp_type], y[exp_type])
                modals[exp_type] = modal(y_merged[flat_keys], hist_range, bins, ignore_zeros)
                stderrs[exp_type] = stderr(y_merged[flat_keys], ignore_zeros)
        else:  # averaging over values of each experiment
            if keys is None:
                for exp_no, values in y[exp_type].iteritems():
                    if not len(values): continue
                    modals[exp_type][exp_no] = modal(values, hist_range, bins, ignore_zeros)
                    stderrs[exp_type][exp_no] = stderr(values, ignore_zeros)
            else:
                for exp_no, values in y[exp_type].iteritems():
                    if not len(values): continue
                    modals[exp_type][exp_no] = modal(values[keys[exp_type][exp_no]], hist_range, bins, ignore_zeros)
                    stderrs[exp_type][exp_no] = stderr(values[keys[exp_type][exp_no]], ignore_zeros)
            if mode == 1:  # cum_averaging_fctn()'s of modals
                modals[exp_type] = cum_averaging_fctn(modals[exp_type].values())
                stderrs[exp_type] = stderr(stderrs[exp_type].values())
    return modals, stderrs


def write_curve_data(output_file_name, output_format, param_info, curve_file_params, curves):
    output_file = open(output_file_name, 'w')
    if output_format == 'igor':
        output_file.write(string.join(
            ["'" + param_info[param][0] + ' [' + param_info[param][2] + ']' + "'" for param in curve_file_params],
            '\t') + '\n')
    else:
        output_file.write('# ' + string.join(
            [param_info[param][0] + ' [' + param_info[param][2] + ']' for param in curve_file_params], '; ') + '\n')
    for i in range(curves.size): output_file.write(
        string.join([str(curves[param][i]) for param in curve_file_params[:-1]], '\t') + '\t# ' + curves['filename'][
            i] + '\n')
    output_file.close()


def write_step_data(output_file_name, output_format, param_info, step_file_params, steps, total_curves, steps_ctr):
    output_file = open(output_file_name, 'w')
    if output_format == 'igor':
        output_file.write(string.join(
            ["'" + param_info[param][0] + ' [' + param_info[param][2] + ']' + "'" for param in step_file_params],
            '\t') + "\t'conseq. curve # [1]'\t'# steps [1]'\n")
    else:
        output_file.write(
            '# ' + string.join([param_info[param][0] + ' [' + param_info[param][2] + ']' for param in step_file_params],
                               '; ') + '; conseq. curve # [1]; # steps [1]\n')
    for i in range(steps.size): output_file.write(
        string.join([str(steps[param][i]) for param in step_file_params], '\t') + '\t' + str(
            total_curves + steps.curve_no[i]) + '\t' + str(steps_ctr[steps.curve_no[i] - 1]) + '\n')
    output_file.close()


def write_curve_data_cum(output_file_name, output_format, write_header, exp_id, exp_no, param_info, curve_file_params,
                         curves):
    if write_header:
        output_file = open(output_file_name, 'w')
        if output_format == 'igor':
            output_file.write(string.join(
                ["'" + param_info[param][0] + ' [' + param_info[param][2] + ']' + "'" for param in
                 curve_file_params[:-1]], '\t') + "\t'exp. id' [1]\t'file name' [?]\n")
        else:
            output_file.write('# ' + string.join(
                [param_info[param][0] + ' [' + param_info[param][2] + ']' for param in curve_file_params[:-1]],
                '; ') + '; exp. id [1]; file name [?]\n')
    else:
        output_file = open(output_file_name, 'a')
    if output_format != 'igor': output_file.write('# %s (%d)\n' % (exp_id, exp_no))
    for i in range(curves.size): output_file.write(
        string.join([str(curves[param][i]) for param in curve_file_params[:-1]], '\t') + '\t' + str(exp_id) + '\t# ' +
        curves['filename'][i] + '\n')
    output_file.close()


def write_step_data_cum(output_file_name, output_format, write_header, exp_id, exp_no, param_info, step_file_params,
                        steps, total_curves, steps_ctr):
    if write_header:
        output_file = open(output_file_name, 'w')
        if output_format == 'igor':
            output_file.write(string.join(
                ["'" + param_info[param][0] + ' [' + param_info[param][2] + ']' + "'" for param in step_file_params],
                '\t') + "\t'conseq. curve # [1]'\t'# steps [1]'\t'exp. id' [1]\n")
        else:
            output_file.write('# ' + string.join(
                [param_info[param][0] + ' [' + param_info[param][2] + ']' for param in step_file_params],
                '; ') + '; conseq. curve # [1]; # steps [1]; exp. id [1]\n')
    else:
        output_file = open(output_file_name, 'a')
    if output_format != 'igor': output_file.write('# %s (%d)\n' % (exp_id, exp_no))
    for i in range(steps.size): output_file.write(
        string.join([str(steps[param][i]) for param in step_file_params], '\t') + '\t' + str(
            total_curves + steps.curve_no[i]) + '\t' + str(steps_ctr[steps.curve_no[i] - 1]) + '\t' + str(
            exp_id) + '\n')
    output_file.close()


def histogram(y, hist_range=None, bins=None, binsize=None, normed=False):
    if not y.size: return [], []
    if hist_range is None:
        hist_range = (min(y), max(y))
    elif hist_range[0] is None and hist_range[1] is None:
        hist_range = (min(y), max(y))
    elif hist_range[0] is None:
        hist_range = (min(y), hist_range[1])
    elif hist_range[1] is None:
        hist_range = (hist_range[0], max(y))
    if bins is None:
        if binsize is None: binsize = float(hist_range[1] - hist_range[0]) / y.size * 10
        if binsize == 0: binsize = 1
        bins = int((hist_range[1] - hist_range[0]) / binsize) + 1
    else:
        binsize = float(hist_range[1] - hist_range[0]) / bins
        if binsize == 0: binsize = 1
    counts = np.zeros(bins, dtype=int)
    for i in range(y.size):
        if y[i] == hist_range[0]:
            bin = 0
        elif y[i] == hist_range[1]:
            bin = bins - 1
        else:
            bin = int((y[i] - hist_range[0]) / binsize)
        if bin >= 0 and bin < bins: counts[bin] += 1
    if normed: counts = counts.astype(float) / np.sum(counts)
    return np.array([hist_range[0] + binsize * i for i in range(bins)]), counts


def CDF(y, hist_range=None, histogram_bins=None, complementary=False):
    if histogram_bins is None: histogram_bins = y.size * 100
    values, counts = histogram(y, hist_range, histogram_bins, normed=True)
    if complementary:
        sum_counts = np.flipud(np.cumsum(np.flipud(counts)))
    else:
        sum_counts = np.cumsum(counts)
    return values, sum_counts


def legend(pl, loc=0):
    labels_exist = False
    ax = pl.gca()
    handles = ax.lines
    handles.extend(ax.patches)
    # handles.extend([c for c in ax.collections if isinstance(c, LineCollection)])
    for handle in handles:
        label = handle.get_label()
        if label is not None and label != '' and not label.startswith('_'):
            labels_exist = True
            break
    if not labels_exist: return
    leg = pl.legend(loc=loc, numpoints=1)
    if leg is None: return
    frame = leg.get_frame()
    frame.set_alpha(0)
    frame.set_facecolor(None)
    # for t in leg.get_texts(): t.set_fontsize('small')


def set_plot_size(plot_size):
    if plot_size is None: plot_size = default_plot_size
    if plot_size is not None: pl.rcParams['figure.figsize'] = float(plot_size[0]) / pl.rcParams['savefig.dpi'], float(
        plot_size[1]) / pl.rcParams['savefig.dpi']


def plot_histogr(y, basename, hist_range=None, plot_size=None, bins=None, xlabel=None, xlim=None, ylim=None,
                 norm_type=None, norm=None):
    y = finite_values(y)
    if not y.size: return
    values, counts = histogram(y, hist_range, bins)
    if bins is None: bins = len(values)
    if norm_type == 1:
        norm = 1. / y.size
    elif norm_type == 2:
        norm = float(norm)
    else:
        norm = 1
    set_plot_size(plot_size)
    pl.bar(values, counts * norm, (values.max() - values.min()) / bins, facecolor='blue')
    if xlim is None and hist_range is not None: xlim = hist_range
    if xlim is not None: pl.xlim(xlim)
    if ylim is not None: pl.ylim(ylim)
    if xlabel is not None: pl.xlabel(xlabel)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_histogr_cum(y, basename, hist_range=None, plot_size=None, bins=None, xlabel=None, xlim=None, ylim=None,
                     subset_keys=None, norm_type=None, norm=None):
    y_merged = np.concatenate(y.values())
    y_merged = finite_values(y_merged)
    if not y_merged.size: return
    values, counts = histogram(y_merged, hist_range, bins, normed=False)
    if norm_type == 1:
        norm = 1. / y_merged.size
    elif norm_type == 2:
        norm = float(norm)
    else:
        norm = 1
    set_plot_size(plot_size)
    pl.bar(values, counts * norm, (values.max() - values.min()) / bins, facecolor='blue')
    if subset_keys is not None:
        values, counts = histogram(np.concatenate([finite_values(y[i][keys]) for i, keys in subset_keys.iteritems()]),
                                   hist_range, bins, normed=False)
        if len(values): pl.bar(values, counts * norm, (values.max() - values.min()) / bins, facecolor='green', alpha=.5)
    if xlim is None and hist_range is not None: xlim = hist_range
    if xlim is not None: pl.xlim(xlim)
    if ylim is not None: pl.ylim(ylim)
    if xlabel is not None: pl.xlabel(xlabel)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_CDF(y, basename, CDF_range, histogram_bins, plot_size=None, xlabel=None, xlim=None, ylim=(0, 100),
             exp_ids=None, complementary=False):
    if CDF_range is None: return
    colors = pl.cm.spectral(np.linspace(0, 255, len(y)).astype(np.int)).tolist()
    set_plot_size(plot_size)
    for i, exp_no in enumerate(sorted(y.keys())):
        curr_values = finite_values(y[exp_no])
        if not len(curr_values): continue
        values, sum_counts = CDF(curr_values, None, histogram_bins, complementary)
        if exp_ids is not None:
            label = exp_ids[exp_no]
        else:
            label = None
        pl.plot(values, sum_counts * 100, '.-', c=colors[i], label=label)
    pl.plot([CDF_range[0], CDF_range[1]], [50, 50], 'k--')
    if xlim is None and CDF_range is not None: xlim = CDF_range
    if xlim is not None: pl.xlim(xlim)
    if ylim is not None: pl.ylim(ylim)
    if xlabel is not None: pl.xlabel(xlabel)
    if complementary:
        pl.ylabel('complementary CDF [%]')
    else:
        pl.ylabel('CDF [%]')
    legend(pl)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_CDF_cum(y, exp_types, basename, CDF_range, histogram_bins, plot_size=None, xlabel=None, xlim=None,
                 ylim=(0, 100), complementary=False, experiment_type_labels=None, experiment_type_colors=None):
    if CDF_range is None: return
    colors = pl.cm.spectral(np.linspace(0, 255, len(exp_types)).astype(np.int)).tolist()
    set_plot_size(plot_size)
    for i, exp_type in enumerate(exp_types):
        y_merged = finite_values(np.concatenate(y[exp_type].values()))
        if not y_merged.size: continue
        values, sum_counts = CDF(y_merged, None, histogram_bins, complementary)
        if experiment_type_labels is not None and exp_type in experiment_type_labels:
            label = experiment_type_labels[exp_type]
        else:
            label = exp_type
        if experiment_type_colors is not None and exp_type in experiment_type_colors:
            color = experiment_type_colors[exp_type]
        else:
            color = colors[i]
        pl.plot(values, sum_counts * 100, '.-', label=label, c=color)
    pl.plot([CDF_range[0], CDF_range[1]], [50, 50], 'k--')
    if xlim is None and CDF_range is not None: xlim = CDF_range
    if xlim is not None: pl.xlim(xlim)
    if ylim is not None: pl.ylim(ylim)
    if xlabel is not None: pl.xlabel(xlabel)
    if complementary:
        pl.ylabel('complementary CDF [%]')
    else:
        pl.ylabel('CDF [%]')
    legend(pl)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def write_averages_data(data, params, param_info, exp_types, output_file_name, exp_ids=None):
    output_file = open(output_file_name, 'w')
    output_file.write(
        '# parameter; experiment type; experiment %s; average value; std error\n' % ('no' if exp_ids is None else 'ID'))
    for param in params:
        averages, stderrs = get_averages(data[param], None, 2)
        for exp_type in exp_types:
            for exp_no in averages[exp_type]: output_file.write('%s [%s]\t%s\t%s\t%f\t%f\n' % (
            param_info[param][0], param_info[param][2], exp_type, exp_no if exp_ids is None else exp_ids[exp_no],
            averages[exp_type][exp_no], stderrs[exp_type][exp_no]))
    output_file.close()


def write_medians_data(data, params, param_info, exp_types, output_file_name, exp_ids=None):
    output_file = open(output_file_name, 'w')
    output_file.write(
        '# parameter; experiment type; experiment %s; median; std error\n' % ('no' if exp_ids is None else 'ID'))
    for param in params:
        medians, stderrs = get_medians(data[param], None, 2)
        for exp_type in exp_types:
            for exp_no in medians[exp_type]: output_file.write('%s [%s]\t%s\t%s\t%f\t%f\n' % (
            param_info[param][0], param_info[param][2], exp_type, exp_no if exp_ids is None else exp_ids[exp_no],
            medians[exp_type][exp_no], stderrs[exp_type][exp_no]))
    output_file.close()


def write_modals_data(data, params, param_info, exp_types, output_file_name, histogram_bins, histogram_ranges,
                      exp_ids=None):
    output_file = open(output_file_name, 'w')
    output_file.write(
        '# parameter; experiment type; experiment %s; modal; std error\n' % ('no' if exp_ids is None else 'ID'))
    for param in params:
        modals, stderrs = get_modals(data[param], histogram_bins, histogram_ranges[param], None, 2)
        for exp_type in exp_types:
            for exp_no in modals[exp_type]: output_file.write('%s [%s]\t%s\t%s\t%f\t%f\n' % (
            param_info[param][0], param_info[param][2], exp_type, exp_no if exp_ids is None else exp_ids[exp_no],
            modals[exp_type][exp_no], stderrs[exp_type][exp_no]))
    output_file.close()


def write_averages_data_cum(data, params, param_info, exp_types, output_file_name, cumulation_mode=0):
    output_file = open(output_file_name, 'w')
    output_file.write('# parameter; experiment type; average value; std error\n')
    output_file.write('# cumulation mode: %d\n' % cumulation_mode)
    for param in params:
        averages, stderrs = get_averages(data[param], None, cumulation_mode)
        for exp_type in exp_types: output_file.write('%s [%s]\t%s\t%f\t%f\n' % (
        param_info[param][0], param_info[param][2], exp_type, averages[exp_type], stderrs[exp_type]))
    output_file.close()


def write_medians_data_cum(data, params, param_info, exp_types, output_file_name, cumulation_mode=0):
    output_file = open(output_file_name, 'w')
    output_file.write('# parameter; experiment type; median; std error\n')
    output_file.write('# cumulation mode: %d\n' % cumulation_mode)
    for param in params:
        medians, stderrs = get_medians(data[param], None, cumulation_mode)
        for exp_type in exp_types: output_file.write('%s [%s]\t%s\t%f\t%f\n' % (
        param_info[param][0], param_info[param][2], exp_type, medians[exp_type], stderrs[exp_type]))
    output_file.close()


def write_modals_data_cum(data, params, param_info, exp_types, output_file_name, histogram_bins, histogram_ranges,
                          cumulation_mode=0):
    output_file = open(output_file_name, 'w')
    output_file.write('# parameter; experiment type; modal; std error\n')
    output_file.write('# cumulation mode: %d\n' % cumulation_mode)
    for param in params:
        modals, stderrs = get_modals(data[param], histogram_bins, histogram_ranges[param], None, cumulation_mode)
        for exp_type in exp_types: output_file.write('%s [%s]\t%s\t%f\t%f\n' % (
        param_info[param][0], param_info[param][2], exp_type, modals[exp_type], stderrs[exp_type]))
    output_file.close()


def write_CDF_data_cum(data, params, param_info, exp_types, output_file_name, histogram_bins):
    output_file = open(output_file_name, 'w')
    for param, (title, label) in [(i, param_info[i][:2]) for i in params]:
        # if hist_range[param] is None: continue
        for i, exp_type in enumerate(exp_types):
            y_merged = finite_values(np.concatenate(data[param][exp_type].values()))
            if not y_merged.size: continue
            values, sum_counts = CDF(y_merged, None, histogram_bins)
            for i in range(1, len(values)):
                if sum_counts[i] > .5:
                    m = float(sum_counts[i] - sum_counts[i - 1]) / (values[i] - values[i - 1])
                    t = sum_counts[i] - m * values[i]
                    CDF50 = (.5 - t) / m
                    output_file.write(
                        '%s [%s]\t%s\t%f\n' % (param_info[param][0], param_info[param][2], exp_type, CDF50))
                    break
    output_file.close()


def plot_scatter(x, y, basename, plot_size=None, xlabel=None, ylabel=None, xlim=None, ylim=None):
    if not x.size or not y.size: return
    set_plot_size(plot_size)
    pl.plot(x, y, '.')
    if xlim is not None: pl.xlim(xlim)
    if ylim is not None: pl.ylim(ylim)
    if xlabel is not None: pl.xlabel(xlabel)
    if ylabel is not None: pl.ylabel(ylabel)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_scatter_cum(x, y, exp_types, basename, plot_size=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
                     exp_ids=None, keys=None):
    set_plot_size(plot_size)
    pl.figure(2)
    ax = pl.gca()
    ax.set_color_cycle(pl.cm.spectral(np.linspace(0, 255, len(exp_types)).astype(np.int)).tolist())
    for exp_type in exp_types:
        if keys is None:
            curr_keys = dict((i, [j for j in range(len(v))]) for i, v in x[exp_type].iteritems())
        else:
            curr_keys = dict((i, keys[exp_type][i]) for i, v in x[exp_type].iteritems())
        pl.figure(1)
        ax = pl.gca()
        ax.set_color_cycle(pl.cm.spectral(np.linspace(0, 255, len(curr_keys)).astype(np.int)).tolist())
        for exp_no, v in curr_keys.iteritems():
            if exp_ids is not None:
                label = exp_ids[exp_no]
            else:
                label = None
            pl.plot([x[exp_type][exp_no][i] for i in v], [y[exp_type][exp_no][i] for i in v], '.', label=label)
        if xlim is not None: pl.xlim(xlim)
        if ylim is not None: pl.ylim(ylim)
        if xlabel is not None: pl.xlabel(xlabel)
        if ylabel is not None: pl.ylabel(ylabel)
        legend(pl)
        pl.title(os.path.basename(basename) + ' - ' + exp_type)
        pl.savefig(basename + ' - ' + exp_type + '.png', format='png')
        pl.close(1)
        pl.figure(2)
        pl.plot([x[exp_type][i][j] for i, v in curr_keys.iteritems() for j in v],
                [y[exp_type][i][j] for i, v in curr_keys.iteritems() for j in v], '.', label=exp_type)
    if xlim is not None: pl.xlim(xlim)
    if ylim is not None: pl.ylim(ylim)
    if xlabel is not None: pl.xlabel(xlabel)
    if ylabel is not None: pl.ylabel(ylabel)
    legend(pl)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close(2)


def plot_boxplot(y, basename, plot_size=None, ylabel=None, ylim=None, exp_ids=None, subset_keys=None):
    y_merged = []
    nonempty_experiments = []
    for exp_no in sorted(y.keys()):
        curr_values = finite_values(y[exp_no])
        if len(curr_values):
            y_merged.append(curr_values)
            nonempty_experiments.append(exp_no)
    if not len(nonempty_experiments): return
    set_plot_size(plot_size)
    pl.boxplot(y_merged, 1, whis=1)
    if subset_keys is not None:
        y_merged = []
        for exp_no in nonempty_experiments:
            curr_values = finite_values(y[exp_no][subset_keys[exp_no]])
            if len(curr_values):
                y_merged.append(curr_values)
            else:
                y_merged.append([np.nan])
        boxplot = pl.boxplot(y_merged, 1, sym='g.', whis=1, widths=.25)
        for inst in boxplot['medians']: inst.set_alpha(.5)
        for inst in boxplot['fliers']: inst.set_alpha(.5)
        for inst in boxplot['whiskers']:
            inst.set_color('green')
            inst.set_alpha(.5)
        for inst in boxplot['boxes']:
            inst.set_color('green')
            inst.set_alpha(.5)
        for inst in boxplot['caps']:
            inst.set_color('green')
            inst.set_alpha(.5)
    if ylim is not None: pl.ylim(ylim)
    if ylabel is not None: pl.ylabel(ylabel)
    ax = pl.gca()
    if exp_ids is not None:
        ax.set_xticklabels([exp_ids[exp_no] for exp_no in nonempty_experiments], size='small', rotation=90)
    else:
        ax.set_xticklabels(nonempty_experiments, size='small', rotation=90)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_boxplot_cum(y, exp_types, basename, plot_size=None, ylabel=None, ylim=None, subset_keys=None):
    y_merged = {}
    nonempty_exp_types = []
    for exp_type in exp_types:
        curr_values = y[exp_type].values()
        if len(curr_values):
            if type(curr_values[0]) == list or type(curr_values[0]) == np.ndarray:
                curr_values = finite_values(np.concatenate(curr_values))
            else:
                curr_values = finite_values(np.array(curr_values))
            if len(curr_values):
                y_merged[exp_type] = curr_values
                nonempty_exp_types.append(exp_type)
    if not len(nonempty_exp_types): return
    set_plot_size(plot_size)
    pl.boxplot([y_merged[exp_type] for exp_type in nonempty_exp_types], 1, sym='b.', whis=1)
    if subset_keys is not None:
        y_merged = {}
        for exp_type in nonempty_exp_types:
            y_merged[exp_type] = np.concatenate(
                [finite_values(y[exp_type][i][keys]) for i, keys in subset_keys[exp_type].iteritems()])
            if not len(y_merged[exp_type]): y_merged[exp_type] = [np.nan]
        boxplot = pl.boxplot([y_merged[exp_type] for exp_type in nonempty_exp_types], 1, sym='g.', whis=1, widths=.25)
        for inst in boxplot['medians']: inst.set_alpha(.5)
        for inst in boxplot['fliers']: inst.set_alpha(.5)
        for inst in boxplot['whiskers']:
            inst.set_color('green')
            inst.set_alpha(.5)
        for inst in boxplot['boxes']:
            inst.set_color('green')
            inst.set_alpha(.5)
        for inst in boxplot['caps']:
            inst.set_color('green')
            inst.set_alpha(.5)
    ax = pl.gca()
    if ylim is not None: ax.set_ylim(ylim)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.set_xticklabels(nonempty_exp_types, size='small', rotation=90)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_errorbar(y, Dy, exp_types, basename, plot_size=None, ylabel=None, ylim=None, ticks=None, format='o'):
    set_plot_size(plot_size)
    ax = pl.gca()
    ax.set_color_cycle(pl.cm.spectral(np.linspace(0, 255, len(exp_types)).astype(np.int)).tolist())
    for exp_type in exp_types:
        keys = [i for i, v in y[exp_type].iteritems() if np.isfinite(v)]
        if len(keys):
            if Dy is None:
                pl.errorbar(keys, [y[exp_type][i] for i in keys], fmt=format, label=exp_type)
            else:
                pl.errorbar(keys, [y[exp_type][i] for i in keys], [Dy[exp_type][i] for i in keys], fmt=format,
                            label=exp_type)
    if ylim is not None: pl.ylim(ylim)
    if ticks is not None:
        pl.xticks(range(len(ticks)), ticks, rotation=90, fontsize='small')  # ! check it
    else:
        pl.xlabel('experiment no.')
    if ylabel is not None: pl.ylabel(ylabel)
    if Dy is None:
        loc = 0
    else:
        loc = None  # bugfix
    legend(pl, loc)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_errorbar_cum(y, Dy, exp_types, basename, plot_size=None, ylabel=None, ylim=None, format='ko',
                      experiment_type_labels=None):
    set_plot_size(plot_size)
    pl.errorbar(range(1, len(y) + 1), [y[exp_type] for exp_type in exp_types],
                None if Dy is None else [Dy[exp_type] for exp_type in exp_types], fmt=format)
    # if experiment_type_labels is None: pl.xticks(range(1, len(y) + 1), exp_types, rotation=90, fontsize='small')
    # else:                              pl.xticks(range(1, len(y) + 1), ['%20s' % experiment_type_labels[exp_type] for exp_type in exp_types], rotation=45, fontsize='small')
    pl.xticks(range(1, len(y) + 1), exp_types, rotation=90, fontsize='small')
    pl.xlim(0, len(y) + 1)
    if ylim is not None: pl.ylim(ylim)
    if ylabel is not None: pl.ylabel(ylabel)
    pl.subplots_adjust(bottom=0.2)
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_errorbar_correlated(x, Dx, y, Dy, exp_types, basename, plot_size=None, xlabel=None, ylabel=None, xlim=None,
                             ylim=None, format='o'):
    set_plot_size(plot_size)
    ax = pl.gca()
    ax.set_color_cycle(pl.cm.spectral(np.linspace(0, 255, len(exp_types)).astype(np.int)).tolist())
    for exp_type in exp_types:
        keys = [i for i in x[exp_type] if np.isfinite(x[exp_type][i]) and np.isfinite(y[exp_type][i])]
        if len(keys): pl.errorbar([x[exp_type][i] for i in keys], [y[exp_type][i] for i in keys],
                                  [Dy[exp_type][i] for i in keys], [Dx[exp_type][i] for i in keys], fmt='o',
                                  label=exp_type)
    if xlim is not None: pl.xlim(xlim)
    if ylim is not None: pl.ylim(ylim)
    if xlabel is not None: pl.xlabel(xlabel)
    if ylabel is not None: pl.ylabel(ylabel)
    legend(pl, loc=None)  # bugfix
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def plot_errorbar_correlated_cum(x, Dx, y, Dy, exp_types, basename, plot_size=None, xlabel=None, ylabel=None, xlim=None,
                                 ylim=None, format='o'):
    set_plot_size(plot_size)
    ax = pl.gca()
    ax.set_color_cycle(pl.cm.spectral(np.linspace(0, 255, len(exp_types)).astype(np.int)).tolist())
    for exp_type in exp_types: pl.errorbar(x[exp_type], y[exp_type], Dy[exp_type], Dx[exp_type], fmt=format,
                                           label=exp_type)
    if xlim is not None: pl.xlim(xlim)
    if ylim is not None: pl.ylim(ylim)
    if xlabel is not None: pl.xlabel(xlabel)
    if ylabel is not None: pl.ylabel(ylabel)
    legend(pl, loc=None)  # bugfix
    pl.title(os.path.basename(basename))
    pl.savefig(basename + '.png', format='png')
    pl.close()


def kruskal_wallis_test(groups):
    from scipy.stats import tiecorrect, rankdata, chisqprob
    n = map(len, groups)
    all = []
    for i in range(len(groups)): all.extend(groups[i].tolist())
    ranked = list(rankdata(all))
    T = tiecorrect(ranked)
    groups = list(groups)
    for i in range(len(groups)):
        groups[i] = ranked[0:n[i]]
        del ranked[0:n[i]]
    rsums = []
    for i in range(len(groups)):
        rsums.append(np.sum(groups[i], axis=0) ** 2)
        rsums[i] = rsums[i] / float(n[i])
    ssbn = np.sum(rsums, axis=0)
    totaln = np.sum(n, axis=0)
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    df = len(groups) - 1
    if T == 0: raise ValueError, 'all numbers are identical in kruskal'
    h = h / float(T)
    return h, chisqprob(h, df)


def plot_averages(data, keys, cumulation_mode, output_dir, title, label, plot_size_errorbar, plot_size_errorbar_cum,
                  plot_size_boxplot, plot_range, tasks, exp_types, exp_ids):
    averages, stderrs = get_averages(data, keys, 2)
    if 'averages' in tasks: plot_errorbar(averages, stderrs, exp_types, output_dir + '/averages/' + title,
                                          plot_size_errorbar, label, plot_range, exp_ids)
    if 'averages_boxplots' in tasks: plot_boxplot_cum(averages, exp_types,
                                                      output_dir + '/boxplots of averages/' + title, plot_size_boxplot,
                                                      label, plot_range)  # ! check if it makes sense
    if 'averages_cum' in tasks:
        averages, stderrs = get_averages(data, keys, cumulation_mode)
        plot_errorbar_cum(averages, stderrs, exp_types, output_dir + '/cumulated averages/' + title,
                          plot_size_errorbar_cum, label)


def plot_averages_correlated(data_x, data_y, keys_x, keys_y, cumulation_mode, output_dir, title, label_x, label_y,
                             plot_size_errorbar_corr, xlim, ylim, tasks, exp_types):
    averages_x, stderrs_x = get_averages(data_x, keys_x, 2)
    averages_y, stderrs_y = get_averages(data_y, keys_y, 2)
    if 'averages' in tasks: plot_errorbar_correlated(averages_x, stderrs_x, averages_y, stderrs_y, exp_types,
                                                     output_dir + '/averages/' + title, plot_size_errorbar_corr,
                                                     label_x, label_y, xlim, ylim)
    if 'averages_cum' in tasks:
        averages_x, stderrs_x = get_averages(data_x, keys_x, cumulation_mode)
        averages_y, stderrs_y = get_averages(data_y, keys_y, cumulation_mode)
        plot_errorbar_correlated_cum(averages_x, stderrs_x, averages_y, stderrs_y, exp_types,
                                     output_dir + '/cumulated averages/' + title, plot_size_errorbar_corr, label_x,
                                     label_y)


def plot_medians(data, keys, cumulation_mode, output_dir, title, label, plot_size_errorbar, plot_size_errorbar_cum,
                 plot_size_boxplot, plot_range, tasks, exp_types, exp_ids):
    medians, stderrs = get_medians(data, keys, 2)
    if 'medians' in tasks: plot_errorbar(medians, stderrs, exp_types, output_dir + '/medians/' + title,
                                         plot_size_errorbar, label, plot_range, exp_ids)
    if 'medians_boxplots' in tasks: plot_boxplot_cum(medians, exp_types, output_dir + '/boxplots of medians/' + title,
                                                     plot_size_boxplot, label, plot_range)
    if 'medians_cum' in tasks:
        medians, stderrs = get_medians(data, keys, cumulation_mode)
        plot_errorbar_cum(medians, stderrs, exp_types, output_dir + '/cumulated medians/' + title,
                          plot_size_errorbar_cum, label)


def plot_medians_correlated(data_x, data_y, keys_x, keys_y, cumulation_mode, output_dir, title, label_x, label_y,
                            plot_size_errorbar_corr, xlim, ylim, tasks, exp_types):
    medians_x, stderrs_x = get_medians(data_x, keys_x, 2)
    medians_y, stderrs_y = get_medians(data_y, keys_y, 2)
    if 'medians' in tasks: plot_errorbar_correlated(medians_x, stderrs_x, medians_y, stderrs_y, exp_types,
                                                    output_dir + '/medians/' + title, plot_size_errorbar_corr, label_x,
                                                    label_y, xlim, ylim)
    if 'medians_cum' in tasks:
        medians_x, stderrs_x = get_medians(data_x, keys_x, cumulation_mode)
        medians_y, stderrs_y = get_medians(data_y, keys_y, cumulation_mode)
        plot_errorbar_correlated_cum(medians_x, stderrs_x, medians_y, stderrs_y, exp_types,
                                     output_dir + '/cumulated medians/' + title, plot_size_errorbar_corr, label_x,
                                     label_y)


def plot_modals(data, keys, cumulation_mode, bins, hist_range, output_dir, title, label, plot_size_errorbar,
                plot_size_errorbar_cum, plot_size_boxplot, plot_range, tasks, exp_types, exp_ids):
    modals, stderrs = get_modals(data, bins, hist_range, keys, 2)
    if 'modals' in tasks: plot_errorbar(modals, stderrs, exp_types, output_dir + '/modals/' + title, plot_size_errorbar,
                                        label, plot_range, exp_ids)
    if 'modals_boxplots' in tasks: plot_boxplot_cum(modals, exp_types, output_dir + '/boxplots of modals/' + title,
                                                    plot_size_boxplot, label, plot_range)
    if 'modals_cum' in tasks:
        modals, stderrs = get_modals(data, bins, hist_range, keys, cumulation_mode)
        plot_errorbar_cum(modals, stderrs, exp_types, output_dir + '/cumulated modals/' + title, plot_size_errorbar_cum,
                          label)


def plot_modals_correlated(data_x, data_y, keys_x, keys_y, cumulation_mode, bins, hist_range_x, hist_range_y,
                           output_dir, title, label_x, label_y, plot_size_errorbar_corr, xlim, ylim, tasks, exp_types):
    modals_x, stderrs_x = get_modals(data_x, bins, hist_range_x, keys_x, 2)
    modals_y, stderrs_y = get_modals(data_y, bins, hist_range_y, keys_y, 2)
    if 'modals' in tasks: plot_errorbar_correlated(modals_x, stderrs_x, modals_y, stderrs_y, exp_types,
                                                   output_dir + '/modals/' + title, plot_size_errorbar_corr, label_x,
                                                   label_y, xlim, ylim)
    if 'modals_cum' in tasks:
        modals_x, stderrs_x = get_modals(data_x, bins, hist_range_x, keys_x, cumulation_mode)
        modals_y, stderrs_y = get_modals(data_y, bins, hist_range_y, keys_y, cumulation_mode)
        plot_errorbar_correlated_cum(modals_x, stderrs_x, modals_y, stderrs_y, exp_types,
                                     output_dir + '/cumulated modals/' + title, plot_size_errorbar_corr, label_x,
                                     label_y)
