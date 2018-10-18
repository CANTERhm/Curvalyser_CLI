#!/usr/bin/env python
# -*- coding: utf-8 -*-

param_info = {
  # experiment-specific:
  'num_curves':            ('total number of curves',        'number of curves',      '1'),
  'num_steps':             ('total number of steps',         'number of steps',       '1'),
  'adhesions':             ('number of adhesive curves',     'number of adhesions',   '1'),
  'adhesion_rate':         ('adhesion rate',                 'adhesion rate [%]',     '%'),
  'steps_ctr_avg':         ('average number of steps',       'number of steps',       '1'),
  'steps_ctr_stderr':      ('stderr(number of steps)',       'standard error',        '1'),
  # curve-specific:
  'steps_ctr':             ('number of steps',               'number of steps',       '1'),
  'peak_pos':              ('peak position',                 'data point',            '1'),
  'peak_extension':        ('peak position',                 r'extension [$\mu m$]',  'um'),
  'peak_force':            ('peak force',                    'force [$pN$]',          'pN'),
  'indent_force':          ('indentation force',             'force [$pN$]',          'pN'),
  'work':                  ('work',                          'work [$aJ$]',           'aJ'),
  'bl_itcpt':              ('baseline interception',         'interception [$pN$]',   'pN'),
  'bl_slope':              ('baseline slope',                r'slope [$pN/\mu m$]',   'pN/um'),
  'bl_crvtr':              ('baseline curvature',            r'slope [$pN/\mu m^2$]', 'pN/um²'),
  'bl_fit_len':            ('baseline fit length',           'data points',           '1'),
  'bl_avg_RSS':            ('baseline RSS',                  'RSS',                   'a.u.'),
  'contact_pos':           ('contact point',                 'data point',            '1'),
  'indent_slope_l':        ('indent. slope (left)',          r'slope [$pN/\mu m$]',   'pN/um'),
  'indent_slope_r':        ('indent. slope (right)',         r'slope [$pN/\mu m$]',   'pN/um'),
  'indent_slope_r_normed': ('indent. slope (right, normed)', r'slope [$pN/\mu m$]',   'pN/um'),
  'trace_fit1':            ('trace fit param #1',            'param #1',              '?'),
  'trace_fit2':            ('trace fit param #2',            'param #2',              '?'),
  'trace_fit3':            ('trace fit param #3',            'param #3',              '?'),
  'trace_fit_RMS':         ('RMS of trace fit',              'RMS',                   '?'),
  'retrace_fit1':          ('retrace fit param #1',          'param #1',              '?'),
  'retrace_fit2':          ('retrace fit param #2',          'param #2',              '?'),
  'retrace_fit3':          ('retrace fit param #3',          'param #3',              '?'),
  'retrace_fit_RMS':       ('RMS of retrace fit',            'RMS',                   '?'),
  'noise_sigma':           ('std dev of noise',              'force [$pN$]',          'pN'),
  'denoising_param':       ('denoising parameter 1',         'parameter',             '1'),
  'denoising_param2':      ('denoising parameter 2',         'parameter',             '1'),
  'indicator_thld':        ('indicator threshold',           'threshold',             '1'),
  'filename':              ('file name',                     'file name',             '?'),
  # step-specific:
  'lt_pos':                ('step position (left flank)',    'data point',            '1'),
  'pos':                   ('step position',                 'data point',            '1'),
  'rt_pos':                ('step position (right flank)',   'data point',            '1'),
  'extension':             ('step position',                 r'extension [$\mu m$]',  'um'),
  'force':                 ('rupture force',                 'force [$pN$]',          'pN'),
  'height':                ('step height',                   'force [$pN$]',          'pN'),
  'width':                 ('step width',                    'extension [$\mu m$]',   'um'),
  'avg_slope':             ('avg step slope',                r'slope [$pN/\mu m$]',   'pN/um'),
  'max_slope':             ('max step slope',                r'slope [$pN/\mu m$]',   'pN/um'),
  'slope_l':               ('slope (left)',                  r'slope [$pN/\mu m$]',   'pN/um'),
  'slope_r':               ('slope (right)',                 r'slope [$pN/\mu m$]',   'pN/um'),
  'RMS_l':                 ('RMS of left step fit',          'RMS',                   '?'),
  'RMS_r':                 ('RMS of right step fit',         'RMS',                   '?'),
  'step_no':               ('step #',                        '',                      '1'),
  'curve_no':              ('curve #',                       '',                      '1'),
  'stiffness':             ('stiffness',                     r'slope [$pN/\mu m$]',   'pN/um')
}
statistics_dir   = 'statistics'                  # relative output directory (relative to base_output_dir from Curvalyser config)
fixed_input_dir  = None                          # absolute input directory;  default: default: c['base_output_dir']
fixed_output_dir = None                          # absolute output directory; default: default: input directory > statistics_dir
multiplier_x = 1                                 # multiplier for extension
multiplier_y = 1                                 # multiplier for force
config_file_pattern = 'config/*'                 # default pattern for config files
experiment_types = []                            # range and order of experiment types to be analysed; see below
analyse_all = 1                                  # 0 = analyse only experiment types specified by experiment_types (in that order), 1 = analyse all experiment types (order specified by experiment_types)
ignore_ids = []                                  # ignore experiments matching the specified IDs
cumulation_mode = 0                              # cumulation method for errorbar plots (0 = cumulate globally; 1 = cumulate and calculate averages/medians/modals by exp_type)
curves_range_start = None                        # first curve to include (counting starts with 1; negative value: count from the end); can be overwritten by paramalyser_curves_range_start in Curvalyser config
curves_range_stop  = None                        # last  curve to include (counting starts with 1; negative value: count from the end); can be overwritten by paramalyser_curves_range_stop  in Curvalyser config
curves_range_step  = None                        # step size (negative value: go backwards); default: auto +1/-1; can be overwritten by paramalyser_curves_range_step in Curvalyser config
autosplit_size  = 0                              # split curves into chunks of size autosplit_size
autosplit_shift = 0                              # shift between the autosplit chunks
# rausgenommen am 26.08.16, da schon im config_paramalyser drin --> tasks = ['data', 'data_cum', 'averages_data', 'medians_data', 'modals_data', 'averages_data_cum', 'medians_data_cum', 'modals_data_cum', 'CDF_data_cum', 'histos', 'histos_cum', 'CDF', 'CDF_cum', 'scatter', 'scatter_cum', 'boxplots', 'boxplots_cum', 'averages', 'averages_cum', 'averages_boxplots', 'medians', 'medians_cum', 'medians_boxplots', 'modals', 'modals_cum', 'modal_boxplots', 'other', 'tests', 'custom'] # tasks to be performed
plot_params = None                               # parameters to be plotted
# rausgenommen am 26.08.16, da schon im config_paramalyser drin --> plot_steps = ['all', 'single', 'first', 'last']  # create extra plots/figures for certain curves (all: all curves, single: only curves with exactly one step, first: only first step, last: only last step)
steps_ctr_averaging = 1                          # how to calculate steps_ctr_avg (0 = consider all curves, 1 = consider only adhesive curves)
custom_script = None                             # custom script to be executed
step_filter_limits  = None                       # filter steps only (applied before curve_filter_limits)
curve_filter_limits = None                       # filter curves AND steps (applied after step_filter_limits)
plot_ranges        = {}                          # default plot ranges
histogram_ranges   = {}                          # plot ranges for histograms
boxplot_ranges     = {}                          # plot ranges for boxplots
scatterplot_ranges = {}                          # plot ranges for scatterplots
experiment_type_labels = None                    # custom labels for experiment types
experiment_type_colors = None                    # custom colors for experiment types
plot_size               = None                   # default plot size
plot_size_histogram     = None                   # plot size for histograms
plot_size_CDF           = None                   # plot size for CDFs
plot_size_scatter       = None                   # plot size for scatterplots
plot_size_boxplot       = None                   # plot size for boxplots
plot_size_errorbar      = None                   # plot size for errorbar plots
plot_size_errorbar_cum  = None                   # plot size for cumulated errorbar plots
plot_size_errorbar_corr = None                   # plot size for correlated errorbar plots
histogram_bins = 50                              # number of histogram bins
histogram_norm_curves = 0                        # 0 = no normalization, 1 = normalize to one, 2 = normalize to average number of steps, 3 = normalize to adhesion rate
histogram_norm_steps  = 0                        # 0 = no normalization, 1 = normalize to one, 2 = normalize to average number of steps, 3 = normalize to adhesion rate
histogram_ylim = None                            # vertical range of histograms
CDF_histogram_bins = None                        # number of histogram bins for CDF plots (default: sum of counts)
output_format = None                             # output format for data files (None = default text format, 'igor' = Igor format)
curvalyser_config = {}                           # overwrites Curvalyser config

import sys
import glob
import os
import os.path
import shutil
import datetime
from optparse import OptionParser
from ParamalyserLib import *
from CurvalyserLib import curve_file_fields, step_file_fields, nat2py_index

#del(curve_file_fields[22]) # for old data files without denoising_parameter2

t0 = datetime.datetime.now()
parser = OptionParser(usage='usage: %prog [options] [config files]')
parser.add_option('-C', '--config_file', dest='config_file',                        metavar='FILE', help='set main config file (default: config-paramalyser)')
parser.add_option('-c', '--cfg_filter',  dest='config_filter',                      metavar='CODE', help='Python expression to select config files')
parser.add_option('-e', '--exp_types',   dest='experiment_types',                   metavar='LIST', help='selection and order of experiment types (comma-separated list)')
parser.add_option('-a', '--analyse_all', dest='analyse_all',   action='store_true',                 help='analyse all experiment types')
parser.add_option('-i', '--input_dir',   dest='input_dir',                          metavar='DIR',  help='set absolute input directory (default: Curvalyser output directory)')
parser.add_option('-o', '--output_dir',  dest='output_dir',                         metavar='DIR',  help='set absolute output directory (default: input directory > statistics_dir)')
parser.add_option('-t', '--tasks',       dest='tasks',                              metavar='LIST', help='list of tasks to perform (comma-separated list)')
parser.add_option('-p', '--params',      dest='params',                             metavar='LIST', help='list of parameters to plot (comma-separated list)')
parser.add_option('-s', '--subsets',     dest='subsets',                            metavar='LIST', help='list of subsets to plot (comma-separated list)')
parser.add_option('-V', '--version',     dest='print_version', action='store_true',                 help='show program version')
(options, args) = parser.parse_args()
if options.print_version:
  import ParamalyserLib
  print 'Paramalyser v' + ParamalyserLib.__version__
  exit(0)
curve_file_params = [v[0] for v in curve_file_fields] # read from file and create array structure
step_file_params  = [v[0] for v in step_file_fields]  # read from file and create array structure
other_params = ['num_curves', 'num_steps', 'adhesions', 'adhesion_rate', 'steps_ctr_avg', 'steps_ctr_stderr', 'indent_slope_r_normed', 'avg_slope', 'stiffness'] # calculated parameters; only create array structure
all_curve_params = curve_file_params[:-1] + ['indent_slope_r_normed'] # automatic plotting
all_step_params  = step_file_params[:-2]  + ['avg_slope', 'stiffness'] # automatic plotting
histogram_norm_multiplier_curves = None
histogram_norm_multiplier_steps  = None
if options.config_file is not None:
  if os.path.isfile(options.config_file): execfile(options.config_file)
  else:
    print "ERROR: config file '%s' doesn't exist!" % options.config_file
    exit(1)
  paramalyser_config_file = options.config_file
elif os.path.isfile('config-paramalyser'):
  execfile('config-paramalyser')
  paramalyser_config_file = 'config-paramalyser'
else:
  paramalyser_config_file = None
  print 'using default configuration'
if options.input_dir        is not None: fixed_input_dir  = options.input_dir
if options.output_dir       is not None: fixed_output_dir = options.output_dir
if options.tasks            is not None: tasks            = options.tasks.split(',')
if options.params           is not None: plot_params      = options.params.split(',')
if options.subsets          is not None: plot_steps       = options.subsets.split(',')
if options.experiment_types is not None: experiment_types = [v.strip() for v in options.experiment_types.split(',')]
if options.analyse_all      is not None: analyse_all      = options.analyse_all
if len(args) >= 1:
  config_files = []
  for arg in args:
    glob_config_dir = glob.glob('config/' + arg)
    if len(glob_config_dir): config_files += glob_config_dir
    else: config_files += glob.glob(arg)
else: config_files = glob.glob(config_file_pattern)
config_files = [v for v in config_files if os.path.isfile(v)]
if not len(config_files):
  print 'ERROR: no config files!'
  exit(1)
config_files.sort()
if plot_params is None:
  curve_params = all_curve_params
  step_params  = all_step_params
else:
  curve_params = []
  step_params = []
  for param in plot_params:
    if   param in all_curve_params: curve_params.append(param)
    elif param in all_step_params:  step_params.append(param)
for param in all_curve_params + all_step_params:
  if param not in plot_ranges:        plot_ranges[param]        = None
  if param not in histogram_ranges:   histogram_ranges[param]   = None
  if param not in boxplot_ranges:     boxplot_ranges[param]     = None
  if param not in scatterplot_ranges: scatterplot_ranges[param] = None
if plot_size is not None: default_plot_size = plot_size
step_filter_curve_limits, step_filter_step_limits, curve_filter_curve_limits, curve_filter_step_limits = get_filter_limits(step_filter_limits, curve_filter_limits)
np.seterr(divide='ignore', invalid='ignore')
total_experiments = 0
total_curves      = {}
total_steps       = {}
total_adhesions   = {}
exp_no = 0
exp_ids = {}
data = {}
for param in curve_file_params + step_file_params + other_params: data[param] = {}
for config_file_no, config_file in enumerate(config_files):
  if 'get_experiment_type' not in locals() or not callable(get_experiment_type): get_experiment_type = None
  c = get_curvalyser_config(config_file, curvalyser_config, get_experiment_type)
  if c['exp_id'] is None: exp_id = os.path.basename(config_file)
  else:                   exp_id = c['exp_id']
  if options.config_filter is not None and not eval(options.config_filter):
    print "%s (%.1f%%, exp. %d of %d):\tskipped (excluded by config filter)" % (exp_id, float(config_file_no)/len(config_files)*100, config_file_no+1, len(config_files))
    continue
  if fixed_input_dir  is not None: c['base_output_dir'] = fixed_input_dir
  if fixed_output_dir is not None: output_dir           = fixed_output_dir
  else:                            output_dir           = c['base_output_dir'] + '/' + statistics_dir
  all_curves, all_steps, experiment_type, experiment_types = read_curvalyser_data(c, config_file_no, config_file, exp_id, len(config_files), ignore_ids, experiment_types, analyse_all)
  if all_curves is None or all_steps is None: continue
  if not os.path.exists(output_dir): os.makedirs(output_dir)
  if 'data'     in tasks and not os.path.exists(output_dir + '/data'):           os.mkdir(output_dir + '/data')
  if 'data_cum' in tasks and not os.path.exists(output_dir + '/cumulated data'): os.mkdir(output_dir + '/cumulated data')
  if 'histos'   in tasks and not os.path.exists(output_dir + '/histograms'):     os.mkdir(output_dir + '/histograms')
  if 'scatter'  in tasks and not os.path.exists(output_dir + '/scatter plots'):  os.mkdir(output_dir + '/scatter plots')
  if 'custom'   in tasks and not os.path.exists(output_dir + '/custom'):         os.mkdir(output_dir + '/custom')
  if paramalyser_config_file is not None and os.path.exists(paramalyser_config_file) and os.path.abspath(paramalyser_config_file) != os.path.abspath(output_dir): shutil.copy2(paramalyser_config_file, output_dir)
  extract_start, extract_stop = nat2py_index(curves_range_start if c['paramalyser_curves_range_start'] is None else c['paramalyser_curves_range_start'], curves_range_stop if c['paramalyser_curves_range_stop'] is None else c['paramalyser_curves_range_stop'], all_curves.size)
  if c['paramalyser_curves_range_step'] is not None: curves_range_step = c['paramalyser_curves_range_step']
  if curves_range_step is None: curves_range_step = 1 if extract_start <= extract_stop else -1
  full_size = all_curves[extract_start:extract_stop:curves_range_step].size
  if autosplit_size > 0:
    part_size = min(autosplit_size, full_size)
    step_size = autosplit_shift if autosplit_shift else part_size
    parts = int(math.ceil(float(full_size) / step_size))
  else:
    part_size = step_size = full_size
    parts = 1
  print '%s (%.1f%%, exp. %d of %d, %s):\t%d curve(s), %d step(s)' % (exp_id, float(config_file_no)/len(config_files)*100., config_file_no+1, len(config_files), experiment_type, all_curves.size, all_steps.size)
  global_exp_id = exp_id
  for part_no in range(parts):
    if autosplit_size > 0:
      exp_id = global_exp_id + '.' + '%03d' % part_no
      if exp_id in ignore_ids:
        print '  %s: skipped (excluded by ignore list)' % exp_id
        continue
    curves, steps = extract_data(all_curves, all_steps, extract_start + part_no * step_size, extract_start + part_no * step_size + part_size)
    if autosplit_size > 0: print '  %s (part %d of %d): %d curve(s), %d step(s) - [%d:%d]' % (exp_id, part_no+1, parts, curves.size, steps.size, extract_start + part_no * step_size, extract_start + part_no * step_size + part_size)
    elif extract_start != 0 or extract_stop != all_curves.size: print '  selected: %d curve(s), %d step(s)' % (curves.size, steps.size)
    if multiplier_x != 1 or multiplier_y != 1:
      curves.peak_extension *= multiplier_x
      curves.peak_force     *= multiplier_y
      curves.indent_force   *= multiplier_y
      curves.work           *= multiplier_x * multiplier_y
      curves.indent_slope_l *= multiplier_y / multiplier_x
      curves.indent_slope_r *= multiplier_y / multiplier_x
      curves.noise_sigma    *= multiplier_y
      steps.extension       *= multiplier_x
      steps.force           *= multiplier_y
      steps.height          *= multiplier_y
      steps.width           *= multiplier_x
      steps.max_slope       *= multiplier_y / multiplier_x
      steps.avg_slope       *= multiplier_y / multiplier_x
      steps.slope_l         *= multiplier_y / multiplier_x
      steps.slope_r         *= multiplier_y / multiplier_x
    if len(step_filter_curve_limits)  or len(step_filter_step_limits):  curves, steps = filter_steps(curves, steps, step_filter_curve_limits, step_filter_step_limits)
    if len(curve_filter_curve_limits) or len(curve_filter_step_limits): curves, steps = filter_curves(curves, steps, curve_filter_curve_limits, curve_filter_step_limits)
    if len(curve_filter_curve_limits) or len(curve_filter_step_limits) or len(step_filter_curve_limits) or len(step_filter_step_limits): print '  filtered: %d curve(s), %d step(s)' % (curves.size, steps.size)
    if experiment_type not in total_curves:
      total_curves[experiment_type]    = 0
      total_steps[experiment_type]     = 0
      total_adhesions[experiment_type] = 0
    if 'data' in tasks:
      write_curve_data(output_dir + '/data/curves - ' + experiment_type + ' - ' + exp_id + '.txt', output_format, param_info, curve_file_params, curves)
      write_step_data(output_dir  + '/data/steps - '  + experiment_type + ' - ' + exp_id + '.txt', output_format, param_info, step_file_params,  steps,  total_curves[experiment_type], curves.steps_ctr)
    if 'data_cum' in tasks:
      write_curve_data_cum(output_dir + '/cumulated data/curves - ' + experiment_type + '.txt', output_format, experiment_type not in data['num_curves'], exp_id, exp_no, param_info, curve_file_params, curves)
      write_step_data_cum(output_dir  + '/cumulated data/steps - '  + experiment_type + '.txt', output_format, experiment_type not in data['num_curves'], exp_id, exp_no, param_info, step_file_params,  steps,  total_curves[experiment_type], curves.steps_ctr)
    for param in curve_file_params + step_file_params + other_params:
      if experiment_type not in data[param]: data[param][experiment_type] = {}
    for param in curve_file_params: data[param][experiment_type][exp_no] = curves[param]
    for param in step_file_params:  data[param][experiment_type][exp_no] = steps[param]
    data['indent_slope_r_normed'][experiment_type][exp_no] = curves.indent_slope_r / curves.steps_ctr
    data['avg_slope'][experiment_type][exp_no]  = steps.height / steps.width
    data['stiffness'][experiment_type][exp_no]  = steps.height / steps.extension
    data['num_curves'][experiment_type][exp_no] = curves.size
    data['num_steps'][experiment_type][exp_no]  = steps.size
    adhesions = np.sum(curves.steps_ctr > 0)
    data['adhesions'][experiment_type][exp_no] = adhesions
    data['adhesion_rate'][experiment_type][exp_no] = float(adhesions)  / curves.size * 100 if curves.size else np.nan
    if steps_ctr_averaging == 1:
      data['steps_ctr_avg'][experiment_type][exp_no] = float(steps.size) / adhesions   if adhesions   else np.nan
      data['steps_ctr_stderr'][experiment_type][exp_no] = stderr(curves.steps_ctr[curves.steps_ctr > 0])
    else:
      data['steps_ctr_avg'][experiment_type][exp_no] = float(steps.size) / curves.size if curves.size else np.nan
      data['steps_ctr_stderr'][experiment_type][exp_no] = stderr(curves.steps_ctr)
    if 'histos' in tasks or 'scatter' in tasks:
      single_step_keys = [i for i in range(steps.size) if curves.steps_ctr[steps.curve_no[i]-1] == 1]
      first_step_keys  = [i for i in range(steps.size) if steps.step_no[i] == 1]
      last_step_keys   = [i for i in range(steps.size) if steps.step_no[i] == curves.steps_ctr[steps.curve_no[i]-1]]
    if 'histos' in tasks:
      path = output_dir + '/histograms/' + exp_id + '/'
      if not os.path.exists(path): os.makedirs(path)
      if curves.size:
        if   histogram_norm_curves == 2: histogram_norm_multiplier_curves = float(steps.size)/curves.size**2 # normalize to average number of steps
        elif histogram_norm_curves == 3: histogram_norm_multiplier_curves = data['adhesion_rate'][experiment_type][exp_no] / 100 / curves.size # normalize to adhesion rate
        for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]: plot_histogr(data[param][experiment_type][exp_no], path + title, histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, min(histogram_norm_curves, 2), histogram_norm_multiplier_curves)
      if curves.size and steps.size:
        if   histogram_norm_steps  == 2: histogram_norm_multiplier_steps  = 1. / curves.size # normalize to average number of steps
        elif histogram_norm_steps  == 3: histogram_norm_multiplier_steps  = data['adhesion_rate'][experiment_type][exp_no] / 100 / steps.size # normalize to adhesion rate
        for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:
          if 'all'    in plot_steps: plot_histogr(data[param][experiment_type][exp_no], path + title + ' (all steps)', histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, min(histogram_norm_steps, 2), histogram_norm_multiplier_steps)
          if 'single' in plot_steps: plot_histogr(data[param][experiment_type][exp_no][single_step_keys], path + title + ' (single step)', histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, min(histogram_norm_steps, 2), histogram_norm_multiplier_steps)
          if 'first'  in plot_steps: plot_histogr(data[param][experiment_type][exp_no][first_step_keys],  path + title + ' (first step)',  histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, min(histogram_norm_steps, 2), histogram_norm_multiplier_steps)
          if 'last'   in plot_steps: plot_histogr(data[param][experiment_type][exp_no][last_step_keys],   path + title + ' (last step)',   histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, min(histogram_norm_steps, 2), histogram_norm_multiplier_steps)
    if 'scatter' in tasks:
      path = output_dir + '/scatter plots/' + exp_id + '/'
      if not os.path.exists(path): os.makedirs(path)
      if 'all'    in plot_steps: plot_scatter(data['extension'][experiment_type][exp_no], data['force'][experiment_type][exp_no], path + 'rupture points (all steps)', plot_size_scatter, param_info['extension'][1], param_info['force'][1], scatterplot_ranges['extension'], scatterplot_ranges['force'])
      if 'single' in plot_steps: plot_scatter(data['extension'][experiment_type][exp_no][single_step_keys], data['force'][experiment_type][exp_no][single_step_keys], path + 'rupture points (single step)', plot_size_scatter, param_info['extension'][1], param_info['force'][1], scatterplot_ranges['extension'], scatterplot_ranges['force'])
      if 'first'  in plot_steps: plot_scatter(data['extension'][experiment_type][exp_no][first_step_keys],  data['force'][experiment_type][exp_no][first_step_keys],  path + 'rupture points (first step)',  plot_size_scatter, param_info['extension'][1], param_info['force'][1], scatterplot_ranges['extension'], scatterplot_ranges['force'])
      if 'last'   in plot_steps: plot_scatter(data['extension'][experiment_type][exp_no][last_step_keys],   data['force'][experiment_type][exp_no][last_step_keys],   path + 'rupture points (last step)',   plot_size_scatter, param_info['extension'][1], param_info['force'][1], scatterplot_ranges['extension'], scatterplot_ranges['force'])
      if 'all'    in plot_steps: plot_scatter(data['extension'][experiment_type][exp_no], data['height'][experiment_type][exp_no], path + 'step heights vs. extensions (all steps)', plot_size_scatter, param_info['extension'][1], param_info['height'][1], scatterplot_ranges['extension'], scatterplot_ranges['height'])
      if 'single' in plot_steps: plot_scatter(data['extension'][experiment_type][exp_no][single_step_keys], data['height'][experiment_type][exp_no][single_step_keys], path + 'step heights vs. extensions (single step)', plot_size_scatter, param_info['extension'][1], param_info['height'][1], scatterplot_ranges['extension'], scatterplot_ranges['height'])
      if 'first'  in plot_steps: plot_scatter(data['extension'][experiment_type][exp_no][first_step_keys],  data['height'][experiment_type][exp_no][first_step_keys],  path + 'step heights vs. extensions (first step)',  plot_size_scatter, param_info['extension'][1], param_info['height'][1], scatterplot_ranges['extension'], scatterplot_ranges['height'])
      if 'last'   in plot_steps: plot_scatter(data['extension'][experiment_type][exp_no][last_step_keys],   data['height'][experiment_type][exp_no][last_step_keys],   path + 'step heights vs. extensions (last step)',   plot_size_scatter, param_info['extension'][1], param_info['height'][1], scatterplot_ranges['extension'], scatterplot_ranges['height'])
    total_experiments                += 1
    total_curves[experiment_type]    += curves.size
    total_steps[experiment_type]     += steps.size
    total_adhesions[experiment_type] += adhesions
    exp_ids[exp_no] = exp_id
    exp_no += 1
for i in range(len(experiment_types) - 1, -1, -1):
  if experiment_types[i] not in data['num_curves']:
    print "[warning] '" + experiment_types[i] + "' is not a valid experiment type"
    experiment_types.pop(i)
if not len(data['num_curves']):
  print 'nothing to do'
  print '%d experiment(s), %d curve(s), %d adhesion curves, %d step(s), execution time = %s' % (total_experiments, sum(total_curves.values()), sum(total_adhesions.values()), sum(total_steps.values()), datetime.datetime.now() - t0)
  exit(0)
if 'data_cum' in tasks:
  for exp_type in experiment_types:
    output_file = open(output_dir + '/cumulated data/experiments - ' + exp_type + '.txt', 'w')
    if output_format == 'igor': output_file.write("'exp. id'\t'# curves'\t'# adhesive curves'\t'# steps'\t'adhesion rate [%]'\t'average # steps'\t'stderr(# steps)'\n")
    else: output_file.write('# exp. id; # curves; # adhesive curves; # steps; adhesion rate [%]; average # steps; stderr(# steps)\n')
    for exp_no in sorted(data['num_curves'][exp_type].keys()): output_file.write('%s\t%d\t%d\t%d\t%f\t%f\t%f\n' % (exp_ids[exp_no], data['num_curves'][exp_type][exp_no], data['adhesions'][exp_type][exp_no], data['num_steps'][exp_type][exp_no], data['adhesion_rate'][exp_type][exp_no], data['steps_ctr_avg'][exp_type][exp_no], data['steps_ctr_stderr'][exp_type][exp_no]))
    output_file.close()
  output_file = open(output_dir + '/cumulated data/experiments.txt', 'w')
  if output_format == 'igor': output_file.write("'exp. type'\t'# experiments'\t'median(adhesion rate) [%]'\t'stderr(adhesion rate)'\t'median(# steps)'\t'stderr(# steps)'\n")
  else: output_file.write('# exp. type; # experiments; median(adhesion rate) [%]; stderr(adhesion rate); median(# steps); stderr(# steps)\n')
  for exp_type in experiment_types: output_file.write('%s\t%d\t%f\t%f\t%f\t%f\n' % (exp_type, len(data['adhesion_rate'][exp_type]), median(np.array(data['adhesion_rate'][exp_type].values())), stderr(np.array(data['adhesion_rate'][exp_type].values())), median(np.array(data['steps_ctr_avg'][exp_type].values())), stderr(np.array(data['steps_ctr_avg'][exp_type].values()))))
  output_file.close()
single_step_keys = {}
first_step_keys = {}
last_step_keys = {}
single_step_curve_keys = {}
for exp_type in experiment_types:
  single_step_keys[exp_type] = {}
  first_step_keys[exp_type]  = {}
  last_step_keys[exp_type]   = {}
  single_step_curve_keys[exp_type] = {}
  for i, v in data['num_steps'][exp_type].iteritems():
    all_keys = range(v)
    single_step_keys[exp_type][i] = [j for j in all_keys if data['steps_ctr'][exp_type][i][data['curve_no'][exp_type][i][j]-1] == 1]
    first_step_keys[exp_type][i]  = [j for j in all_keys if data['step_no'][exp_type][i][j] == 1]
    last_step_keys[exp_type][i]   = [j for j in all_keys if data['step_no'][exp_type][i][j] == data['steps_ctr'][exp_type][i][data['curve_no'][exp_type][i][j]-1]]
  for i, v in data['steps_ctr'][exp_type].iteritems(): single_step_curve_keys[exp_type][i] = [j for j in range(len(v)) if v[j] == 1]
if 'other' in tasks:
  print 'plotting other graphs'
  plot_errorbar(data['steps_ctr_avg'], data['steps_ctr_stderr'], experiment_types, output_dir + '/average # steps', plot_size_errorbar, param_info['steps_ctr_avg'][1], None, exp_ids)
  plot_errorbar_cum(dict((exp_type, median(np.array(data['steps_ctr_avg'][exp_type].values()))) for exp_type in data['steps_ctr_avg']), dict((exp_type, stderr(np.array(data['steps_ctr_avg'][exp_type].values()))) for exp_type in data['steps_ctr_avg']), experiment_types, output_dir + '/average # steps (cumulated)', plot_size_errorbar_cum, param_info['steps_ctr_avg'][1], experiment_type_labels=experiment_type_labels)
  plot_errorbar(data['adhesion_rate'], None, experiment_types, output_dir + '/adhesion rate', plot_size_errorbar, param_info['adhesion_rate'][1], None, exp_ids)
  plot_errorbar_cum(dict((exp_type, median(np.array(data['adhesion_rate'][exp_type].values()))) for exp_type in data['adhesion_rate']), dict((exp_type, stderr(np.array(data['adhesion_rate'][exp_type].values()))) for exp_type in data['adhesion_rate']), experiment_types, output_dir + '/adhesion rate (cumulated)', plot_size_errorbar_cum, param_info['adhesion_rate'][1], experiment_type_labels=experiment_type_labels)
if 'custom' in tasks:
  if custom_script is None: print "[WARNING] 'custom_script' is not defined"
  elif os.path.isfile(custom_script):
    print "executing custom script '%s'" % custom_script
    execfile(custom_script)
  else: print "[ERROR]   custom script '" + custom_script + "' doesn't exist"
if 'averages_data' in tasks:
  path = output_dir + '/data/'
  if not os.path.exists(path): os.mkdir(path)
  write_averages_data(data, curve_params + step_params, param_info, experiment_types, path + 'averages.txt', exp_ids)
if 'medians_data' in tasks:
  path = output_dir + '/data/'
  if not os.path.exists(path): os.mkdir(path)
  write_medians_data(data, curve_params + step_params, param_info, experiment_types, path + 'medians.txt', exp_ids)
if 'modals_data' in tasks:
  path = output_dir + '/data/'
  if not os.path.exists(path): os.mkdir(path)
  write_modals_data(data, curve_params + step_params, param_info, experiment_types, path + 'modals.txt', histogram_bins, histogram_ranges, exp_ids)
if 'averages_data_cum' in tasks:
  path = output_dir + '/cumulated data/'
  if not os.path.exists(path): os.mkdir(path)
  write_averages_data_cum(data, curve_params + step_params, param_info, experiment_types, path + 'averages.txt', cumulation_mode)
if 'medians_data_cum' in tasks:
  path = output_dir + '/cumulated data/'
  if not os.path.exists(path): os.mkdir(path)
  write_medians_data_cum(data, curve_params + step_params, param_info, experiment_types, path + 'medians.txt', cumulation_mode)
if 'modals_data_cum' in tasks:
  path = output_dir + '/cumulated data/'
  if not os.path.exists(path): os.mkdir(path)
  write_modals_data_cum(data, curve_params + step_params, param_info, experiment_types, path + 'modals.txt', histogram_bins, histogram_ranges, cumulation_mode)
if 'CDF_data_cum' in tasks:
  path = output_dir + '/cumulated data/'
  if not os.path.exists(path): os.mkdir(path)
  write_CDF_data_cum(data, curve_params + step_params, param_info, experiment_types, path + 'CDFs.txt', CDF_histogram_bins)
if 'histos_cum' in tasks:
  path = output_dir + '/histograms/'
  if not os.path.exists(path): os.mkdir(path)
  for exp_type in experiment_types:
    print 'plotting cumulated histograms:', exp_type
    if total_curves[exp_type] > 0:
      if   histogram_norm_curves == 2: histogram_norm_multiplier_curves = float(total_steps[exp_type]) / total_curves[exp_type]**2 # normalize to average number of steps
      elif histogram_norm_curves == 3: histogram_norm_multiplier_curves = average(np.array(data['adhesion_rate'][exp_type].values()), weights=data['num_curves'][exp_type].values()) / 100 / total_curves[exp_type] # normalize to adhesion rate
      for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]: plot_histogr_cum(data[param][exp_type],                   path + title + ' - ' + exp_type,              histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, single_step_curve_keys[exp_type], min(histogram_norm_curves, 2), histogram_norm_multiplier_curves)
    if total_curves[exp_type] > 0 and total_steps[exp_type] > 0:
      if   histogram_norm_steps  == 2: histogram_norm_multiplier_steps  = 1. / total_curves[exp_type] # normalize to average number of steps
      elif histogram_norm_steps  == 3: histogram_norm_multiplier_steps  = average(np.array(data['adhesion_rate'][exp_type].values()), weights=data['num_curves'][exp_type].values()) / 100 / total_steps[exp_type] # normalize to adhesion rate
      for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:
	if 'all'   in plot_steps: plot_histogr_cum(data[param][exp_type],                                                                    path + title + ' (all steps) - '  + exp_type, histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, single_step_keys[exp_type],       min(histogram_norm_steps, 2),  histogram_norm_multiplier_steps)
	if 'first' in plot_steps: plot_histogr_cum(dict((i, v[first_step_keys[exp_type][i]]) for i, v in data[param][exp_type].iteritems()), path + title + ' (first step) - ' + exp_type, histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, None,                             min(histogram_norm_steps, 2),  histogram_norm_multiplier_steps)
	if 'last'  in plot_steps: plot_histogr_cum(dict((i, v[last_step_keys[exp_type][i]])  for i, v in data[param][exp_type].iteritems()), path + title + ' (last step) - '  + exp_type, histogram_ranges[param], plot_size_histogram, histogram_bins, label, None, histogram_ylim, None,                             min(histogram_norm_steps, 2),  histogram_norm_multiplier_steps)
if 'CDF' in tasks:
  path = output_dir + '/CDFs/'
  if not os.path.exists(path): os.mkdir(path)
  for exp_type in experiment_types:
    print 'plotting CDFs:', exp_type
    for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]: plot_CDF(data[param][exp_type], path + title + ' - ' + exp_type, histogram_ranges[param], CDF_histogram_bins, plot_size_CDF, label, exp_ids=exp_ids)
    for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:  plot_CDF(data[param][exp_type], path + title + ' - ' + exp_type, histogram_ranges[param], CDF_histogram_bins, plot_size_CDF, label, exp_ids=exp_ids)
if 'CDF_cum' in tasks:
  path = output_dir + '/cumulated CDFs/'
  if not os.path.exists(path): os.mkdir(path)
  print 'plotting cumulated CDFs'
  for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]: plot_CDF_cum(data[param], experiment_types, path + title, histogram_ranges[param], CDF_histogram_bins, plot_size_CDF, label, experiment_type_labels=experiment_type_labels, experiment_type_colors=experiment_type_colors)
  for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:  plot_CDF_cum(data[param], experiment_types, path + title, histogram_ranges[param], CDF_histogram_bins, plot_size_CDF, label, experiment_type_labels=experiment_type_labels, experiment_type_colors=experiment_type_colors)
if 'boxplots' in tasks:
  path = output_dir + '/boxplots/'
  if not os.path.exists(path): os.mkdir(path)
  for exp_type in experiment_types:
    print 'creating boxplots:', exp_type
    for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]: plot_boxplot(data[param][exp_type], path + title + ' - ' + exp_type, plot_size_boxplot, label, boxplot_ranges[param], exp_ids, single_step_curve_keys[exp_type])
    for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:
      if 'all'   in plot_steps: plot_boxplot(data[param][exp_type],                                                                    path + title + ' (all steps) - '  + exp_type, plot_size_boxplot, label, boxplot_ranges[param], exp_ids, single_step_keys[exp_type])
      if 'first' in plot_steps: plot_boxplot(dict((i, v[first_step_keys[exp_type][i]]) for i, v in data[param][exp_type].iteritems()), path + title + ' (first step) - ' + exp_type, plot_size_boxplot, label, boxplot_ranges[param], exp_ids)
      if 'last'  in plot_steps: plot_boxplot(dict((i, v[last_step_keys[exp_type][i]])  for i, v in data[param][exp_type].iteritems()), path + title + ' (last step) - '  + exp_type, plot_size_boxplot, label, boxplot_ranges[param], exp_ids)
if 'boxplots_cum' in tasks:
  path = output_dir + '/cumulated boxplots/'
  if not os.path.exists(path): os.mkdir(path)
  print 'creating cumulated boxplots'
  for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]: plot_boxplot_cum(data[param], experiment_types, path + title, plot_size_boxplot, label, boxplot_ranges[param], single_step_curve_keys)
  for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:
    if 'all'   in plot_steps: plot_boxplot_cum(data[param],                                                                                                                            experiment_types, path + title + ' (all steps)',  plot_size_boxplot, label, boxplot_ranges[param], single_step_keys)
    if 'first' in plot_steps: plot_boxplot_cum(dict((exp_type, dict((i, v[first_step_keys[exp_type][i]]) for i, v in data[param][exp_type].iteritems())) for exp_type in data[param]), experiment_types, path + title + ' (first step)', plot_size_boxplot, label, boxplot_ranges[param])
    if 'last'  in plot_steps: plot_boxplot_cum(dict((exp_type, dict((i, v[last_step_keys[exp_type][i]])  for i, v in data[param][exp_type].iteritems())) for exp_type in data[param]), experiment_types, path + title + ' (last step)',  plot_size_boxplot, label, boxplot_ranges[param])
if 'averages' in tasks or 'averages_cum' in tasks or 'averages_boxplots' in tasks:
  if 'averages'          in tasks and not os.path.exists(output_dir + '/averages'):             os.mkdir(output_dir + '/averages')
  if 'averages_cum'      in tasks and not os.path.exists(output_dir + '/cumulated averages'):   os.mkdir(output_dir + '/cumulated averages')
  if 'averages_boxplots' in tasks and not os.path.exists(output_dir + '/boxplots of averages'): os.mkdir(output_dir + '/boxplots of averages')
  print 'plotting averages'
  for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]:
    plot_averages(data[param], None,                   cumulation_mode, output_dir, title + ' (all steps)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if param == 'steps_ctr': continue
    plot_averages(data[param], single_step_curve_keys, cumulation_mode, output_dir, title + ' (single step)', label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
  for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:
    if 'all'    in plot_steps: plot_averages(data[param], None,             cumulation_mode, output_dir, title + ' (all steps)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'single' in plot_steps: plot_averages(data[param], single_step_keys, cumulation_mode, output_dir, title + ' (single step)', label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'first'  in plot_steps: plot_averages(data[param], first_step_keys,  cumulation_mode, output_dir, title + ' (first step)',  label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'last'   in plot_steps: plot_averages(data[param], last_step_keys,   cumulation_mode, output_dir, title + ' (last step)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
  plot_averages_correlated(data['extension'], data['height'],     None,           None,           cumulation_mode, output_dir, 'step height vs. step position',           param_info['extension'][1], param_info['height'][1],     plot_size_errorbar_corr, plot_ranges['extension'], plot_ranges['height'],     tasks, experiment_types)
  plot_averages_correlated(data['extension'], data['height'],     last_step_keys, last_step_keys, cumulation_mode, output_dir, 'last step height vs. last step position', param_info['extension'][1], param_info['height'][1],     plot_size_errorbar_corr, plot_ranges['extension'], plot_ranges['height'],     tasks, experiment_types)
  plot_averages_correlated(data['extension'], data['peak_force'], last_step_keys, None,           cumulation_mode, output_dir, 'peak force vs. last step position',       param_info['extension'][1], param_info['peak_force'][1], plot_size_errorbar_corr, plot_ranges['extension'], plot_ranges['peak_force'], tasks, experiment_types)
if 'medians' in tasks or 'medians_cum' in tasks or 'medians_boxplots' in tasks:
  if 'medians'          in tasks and not os.path.exists(output_dir + '/medians'):             os.mkdir(output_dir + '/medians')
  if 'medians_cum'      in tasks and not os.path.exists(output_dir + '/cumulated medians'):   os.mkdir(output_dir + '/cumulated medians')
  if 'medians_boxplots' in tasks and not os.path.exists(output_dir + '/boxplots of medians'): os.mkdir(output_dir + '/boxplots of medians')
  print 'plotting medians'
  for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]:
    plot_medians(data[param], None,                   cumulation_mode, output_dir, title + ' (all steps)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if param == 'steps_ctr': continue
    plot_medians(data[param], single_step_curve_keys, cumulation_mode, output_dir, title + ' (single step)', label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
  for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:
    if 'all'    in plot_steps: plot_medians(data[param], None,             cumulation_mode, output_dir, title + ' (all steps)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'single' in plot_steps: plot_medians(data[param], single_step_keys, cumulation_mode, output_dir, title + ' (single step)', label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'first'  in plot_steps: plot_medians(data[param], first_step_keys,  cumulation_mode, output_dir, title + ' (first step)',  label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'last'   in plot_steps: plot_medians(data[param], last_step_keys,   cumulation_mode, output_dir, title + ' (last step)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
  plot_medians_correlated(data['extension'],   data['height'],     None,           None,           cumulation_mode, output_dir, 'step height vs. step position',           param_info['extension'][1],   param_info['height'][1],     plot_size_errorbar_corr, plot_ranges['extension'],   plot_ranges['height'],     tasks, experiment_types)
  plot_medians_correlated(data['extension'],   data['height'],     last_step_keys, last_step_keys, cumulation_mode, output_dir, 'last step height vs. last step position', param_info['extension'][1],   param_info['height'][1],     plot_size_errorbar_corr, plot_ranges['extension'],   plot_ranges['height'],     tasks, experiment_types)
  plot_medians_correlated(data['extension'],   data['peak_force'], last_step_keys, None,           cumulation_mode, output_dir, 'peak force vs. last step position',       param_info['extension'][1],   param_info['peak_force'][1], plot_size_errorbar_corr, plot_ranges['extension'],   plot_ranges['peak_force'], tasks, experiment_types)
  plot_medians_correlated(data['noise_sigma'], data['peak_force'], None,           None,           cumulation_mode, output_dir, 'peak force vs. std dev of noise',         param_info['noise_sigma'][1], param_info['peak_force'][1], plot_size_errorbar_corr, plot_ranges['noise_sigma'], plot_ranges['peak_force'], tasks, experiment_types)
if 'modals' in tasks or 'modals_cum' in tasks or 'modal_boxplots' in tasks:
  if 'modals'         in tasks and not os.path.exists(output_dir + '/modals'):             os.mkdir(output_dir + '/modals')
  if 'modals_cum'     in tasks and not os.path.exists(output_dir + '/cumulated modals'):   os.mkdir(output_dir + '/cumulated modals')
  if 'modal_boxplots' in tasks and not os.path.exists(output_dir + '/boxplots of modals'): os.mkdir(output_dir + '/boxplots of modals')
  print 'plotting modals'
  for param, (title, label) in [(i, param_info[i][:2]) for i in curve_params]:
    plot_modals(data[param], None,                   cumulation_mode, histogram_bins, histogram_ranges[param], output_dir, title + ' (all steps)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if param == 'steps_ctr': continue
    plot_modals(data[param], single_step_curve_keys, cumulation_mode, histogram_bins, histogram_ranges[param], output_dir, title + ' (single step)', label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
  for param, (title, label) in [(i, param_info[i][:2]) for i in step_params]:
    if 'all'    in plot_steps: plot_modals(data[param], None,             cumulation_mode, histogram_bins, histogram_ranges[param], output_dir, title + ' (all steps)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'single' in plot_steps: plot_modals(data[param], single_step_keys, cumulation_mode, histogram_bins, histogram_ranges[param], output_dir, title + ' (single step)', label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'first'  in plot_steps: plot_modals(data[param], first_step_keys,  cumulation_mode, histogram_bins, histogram_ranges[param], output_dir, title + ' (first step)',  label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
    if 'last'   in plot_steps: plot_modals(data[param], last_step_keys,   cumulation_mode, histogram_bins, histogram_ranges[param], output_dir, title + ' (last step)',   label, plot_size_errorbar, plot_size_errorbar_cum, plot_size_boxplot, plot_ranges[param], tasks, experiment_types, exp_ids)
  plot_modals_correlated(data['extension'],   data['height'],     None,           None,           cumulation_mode, histogram_bins, histogram_ranges['extension'],   histogram_ranges['height'],     output_dir, 'step height vs. step position',           param_info['extension'][1],   param_info['height'][1],     plot_size_errorbar_corr, plot_ranges['extension'],   plot_ranges['height'],     tasks, experiment_types)
  plot_modals_correlated(data['extension'],   data['height'],     last_step_keys, last_step_keys, cumulation_mode, histogram_bins, histogram_ranges['extension'],   histogram_ranges['height'],     output_dir, 'last step height vs. last step position', param_info['extension'][1],   param_info['height'][1],     plot_size_errorbar_corr, plot_ranges['extension'],   plot_ranges['height'],     tasks, experiment_types)
  plot_modals_correlated(data['extension'],   data['peak_force'], last_step_keys, None,           cumulation_mode, histogram_bins, histogram_ranges['extension'],   histogram_ranges['peak_force'], output_dir, 'peak force vs. last step position',       param_info['extension'][1],   param_info['peak_force'][1], plot_size_errorbar_corr, plot_ranges['extension'],   plot_ranges['peak_force'], tasks, experiment_types)
  plot_modals_correlated(data['noise_sigma'], data['peak_force'], None,           None,           cumulation_mode, histogram_bins, histogram_ranges['noise_sigma'], histogram_ranges['peak_force'], output_dir, 'peak force vs. std dev of noise',         param_info['noise_sigma'][1], param_info['peak_force'][1], plot_size_errorbar_corr, plot_ranges['noise_sigma'], plot_ranges['peak_force'], tasks, experiment_types)
if 'scatter_cum' in tasks:
  path = output_dir + '/scatter plots/'
  if not os.path.exists(path): os.mkdir(path)
  print 'creating cumulated scatter plots'
  if 'all'    in plot_steps: plot_scatter_cum(data['extension'], data['force'], experiment_types, path + 'rupture points (all steps)',   plot_size_scatter, param_info['extension'][1], param_info['force'][1], scatterplot_ranges['extension'], scatterplot_ranges['force'], exp_ids)
  if 'single' in plot_steps: plot_scatter_cum(data['extension'], data['force'], experiment_types, path + 'rupture points (single step)', plot_size_scatter, param_info['extension'][1], param_info['force'][1], scatterplot_ranges['extension'], scatterplot_ranges['force'], exp_ids, single_step_keys)
  if 'first'  in plot_steps: plot_scatter_cum(data['extension'], data['force'], experiment_types, path + 'rupture points (first step)',  plot_size_scatter, param_info['extension'][1], param_info['force'][1], scatterplot_ranges['extension'], scatterplot_ranges['force'], exp_ids, first_step_keys)
  if 'last'   in plot_steps: plot_scatter_cum(data['extension'], data['force'], experiment_types, path + 'rupture points (last step)',   plot_size_scatter, param_info['extension'][1], param_info['force'][1], scatterplot_ranges['extension'], scatterplot_ranges['force'], exp_ids, last_step_keys)
  if 'all'    in plot_steps: plot_scatter_cum(data['extension'], data['height'], experiment_types, path + 'step heights vs. extensions (all steps)',   plot_size_scatter, param_info['extension'][1], param_info['height'][1], scatterplot_ranges['extension'], scatterplot_ranges['height'], exp_ids)
  if 'single' in plot_steps: plot_scatter_cum(data['extension'], data['height'], experiment_types, path + 'step heights vs. extensions (single step)', plot_size_scatter, param_info['extension'][1], param_info['height'][1], scatterplot_ranges['extension'], scatterplot_ranges['height'], exp_ids, single_step_keys)
  if 'first'  in plot_steps: plot_scatter_cum(data['extension'], data['height'], experiment_types, path + 'step heights vs. extensions (first step)',  plot_size_scatter, param_info['extension'][1], param_info['height'][1], scatterplot_ranges['extension'], scatterplot_ranges['height'], exp_ids, first_step_keys)
  if 'last'   in plot_steps: plot_scatter_cum(data['extension'], data['height'], experiment_types, path + 'step heights vs. extensions (last step)',   plot_size_scatter, param_info['extension'][1], param_info['height'][1], scatterplot_ranges['extension'], scatterplot_ranges['height'], exp_ids, last_step_keys)
if 'tests' in tasks:
  from scipy.stats import mannwhitneyu, chi2
  print 'calculating Mann-Whitney Tests' # see http://statisticslectures.com/mannwhitneyu.php and http://faculty.vassar.edu/lowry/zsamp0.html
  output_file = open(output_dir + '/mann-whitney.txt', 'w')
  output_file.write('# parameter, exp_type1, exp_type2, p, 2*p, U, N1, N2\n')
  for param in sorted(data.keys()):
    if param in ['num_curves', 'num_steps', 'steps_ctr_stderr', 'filename', 'step_no', 'curve_no']: continue
    for i, exp_type1 in enumerate(experiment_types):
      for j in range(i + 1, len(experiment_types)):
        exp_type2 = experiment_types[j]
        if param in ['adhesions', 'adhesion_rate', 'steps_ctr_avg']:
          sample1 = np.array(data[param][exp_type1].values())
          sample2 = np.array(data[param][exp_type2].values())
        else:
          sample1 = np.concatenate(data[param][exp_type1].values())
          sample2 = np.concatenate(data[param][exp_type2].values())
          #sample1 = [median(data[param][exp_type1][i]) for i in data[param][exp_type1]]
          #sample2 = [median(data[param][exp_type2][i]) for i in data[param][exp_type2]]
        if len(sample1) and len(sample2):
	  try:
	    U, p = mannwhitneyu(sample1, sample2)
	  except ValueError:
	    U, p = np.nan, np.nan
	    print '  Mann-Whitney Test for "%s" in experiment types "%s"/"%s" failed: %s (N1 = %d, N2 = %d)' % (param_info[param][0], exp_type1, exp_type2, sys.exc_info()[1], len(sample1), len(sample2))
        else:
          U, p = np.nan, np.nan
          print '  skipped Mann-Whitney Test for "%s" in experiment types "%s"/"%s" (N1 = %d, N2 = %d)' % (param_info[param][0], exp_type1, exp_type2, len(sample1), len(sample2))
        output_file.write('%s\t%s\t%s\t%e\t%e\t%s\t%d\t%d\n' % (param, exp_type1, exp_type2, p, 2*p, U, len(sample1), len(sample2)))
        #print '  %s / %s vs. %s: p = %e, U = %d, N1 = %d, N2 = %d' % (param, exp_type1, exp_type2, p, U, len(sample1), len(sample2))
  output_file.close()
  print 'calculating Kruskal-Wallis Tests' # see http://adorio-research.org/wordpress/?p=237
  output_file = open(output_dir + '/kruskal-wallis.txt', 'w')
  output_file.write('# parameter, exp_type, H0/H1, H, H_critical, p, # groups\n')
  for param in sorted(data.keys()):
    if param in ['num_curves', 'num_steps', 'adhesions', 'adhesion_rate', 'steps_ctr_avg', 'steps_ctr_stderr', 'filename', 'step_no', 'curve_no']: continue
    for exp_type in experiment_types:
      if len(data[param][exp_type].values()) >= 2:
        try: H, p = kruskal_wallis_test(data[param][exp_type].values())
	except ValueError:
	  H, p = np.nan, np.nan
	  print '  Kruskal-Wallis Test for "%s" in experiment type "%s" failed: %s (N = %d)' % (param_info[param][0], exp_type, sys.exc_info()[1], len(data[param][exp_type].values()))
      else:
        H, p = np.nan, np.nan
        print '  skipped Kruskal-Wallis Test for "%s" in experiment type "%s" (N = %d)' % (param_info[param][0], exp_type, len(data[param][exp_type].values()))
      H_critical = chi2.ppf(0.95, len(data[param][exp_type])-1)
      output_file.write('%s\t%s\t%s\t%f\t%f\t%e\t%d\n' % (param, exp_type, 'H1' if H > H_critical else 'H0', H, H_critical, p, len(data[param][exp_type])))
      #print '  %s / %s: H = %f, p = %e' % (param, exp_type, H, p)
  output_file.close()
print '%d experiment(s), %d curve(s), %d adhesion curve(s), %d step(s), execution time = %s' % (total_experiments, sum(total_curves.values()), sum(total_adhesions.values()), sum(total_steps.values()), datetime.datetime.now() - t0)
