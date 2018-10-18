#!/usr/bin/env python
# -*- coding: utf-8 -*-

# history:
# v1.0.3:   support for new CurvalyserLib (v1.7.14)

__version__ = '1.0.3'

import glob
import os.path
import datetime
import numpy as np
import matplotlib.pyplot as pl
from optparse import OptionParser
from CurvalyserLib import load_curvalyser_curve_data, load_curvalyser_step_data, curve_file_fields, step_file_fields, load_curve, prepare_data, denoise, nat2py_index, fmt_float
from ParamalyserLib import get_filter_limits, get_curvalyser_config, filter_steps, filter_curves

def plot_curve(x, y, y_denoised, x_trace, y_trace, steps, filtered_steps, c, filename):
  if c['plot_size_force_curves'] is None: fig_size = None
  else: fig_size = (float(c['plot_size_force_curves'][0])/pl.rcParams['savefig.dpi'], float(c['plot_size_force_curves'][1])/pl.rcParams['savefig.dpi'])
  pl.figure(figsize=fig_size)
  if x_trace is not None and y_trace is not None: pl.plot(x_trace, y_trace, '#7FFF7F')
  if y is not None: pl.plot(x, y, 'b')
  if y_denoised is not None and y_denoised is not y: pl.plot(x, y_denoised, 'k')
  pl.plot([x[0], x[-1]], [0, 0], '--', c='.75')
  if steps.size:
  #  for step in steps:
  #    line_x = x[r['steps']['lt_pos'][i] + 1 - c['fit_drawing_width'] : r['steps']['lt_pos'][i] + 1]
  #    line_y = line_x * r['steps_fits']['lt_slope'][i] + r['steps_fits']['lt_itcpt'][i]
  #    pl.plot(line_x, line_y, 'c', lw=2)
  #    line_x = x[r['steps']['rt_pos'][i] : r['steps']['rt_pos'][i] + c['fit_drawing_width']]
  #    line_y = line_x * r['steps_fits']['rt_slope'][i] + r['steps_fits']['rt_itcpt'][i]
  #    pl.plot(line_x, line_y, 'c', lw=2)
    #pl.plot(steps.extension, steps.force, 'y^', mew=0)
    pl.plot(x[steps.lmax_pos], y_denoised[steps.lmax_pos], 'y^', ms=5, mew=0)
    pl.plot(x[steps.lt_pos],   y_denoised[steps.lt_pos],   'rx', ms=5, mew=1)
    pl.plot(x[steps.rt_pos],   y_denoised[steps.rt_pos],   'g+', ms=5, mew=1)
  if filtered_steps.size:
    pl.plot(x[filtered_steps.lmax_pos], y_denoised[filtered_steps.lmax_pos], 'y^', ms=6,  mew=0)
    pl.plot(x[filtered_steps.lt_pos],   y_denoised[filtered_steps.lt_pos],   'rx', ms=10, mew=2)
    pl.plot(x[filtered_steps.rt_pos],   y_denoised[filtered_steps.rt_pos],   'g+', ms=10, mew=2)
    #pl.plot(x[filtered_steps.lmax_pos], y_denoised[filtered_steps.lmax_pos], '^',  ms=10, mew=1, mfc='None', mec='yellow')
    #pl.plot(x[filtered_steps.lt_pos],   y_denoised[filtered_steps.lt_pos],   'o',  ms=10, mew=1, mfc='None', mec='red')
    #pl.plot(x[filtered_steps.rt_pos],   y_denoised[filtered_steps.rt_pos],   'o',  ms=10, mew=1, mfc='None', mec='green')
  pl.axis(xmin=c['plot_xmin'], xmax=c['plot_xmax'], ymin=c['plot_ymin'], ymax=c['plot_ymax'])
  pl.xlabel(u'z [%s]' % c['unit_x'])
  pl.ylabel('F [%s]' % c['unit_y'])
  if filename is not None: pl.savefig(filename + '.' + c['plot_format'], format=c['plot_format'])
  pl.close()

t0 = datetime.datetime.now()
parser = OptionParser(usage='usage: %prog [options] [config files]')
parser.add_option('-C', '--config_file', dest='config_file',                        metavar='FILE',         help='set Paramalyser config file (default: config-paramalyser)')
parser.add_option('-c', '--cfg_filter',  dest='config_filter',                      metavar='CODE',         help='Python expression to select config files')
parser.add_option('-e', '--exp_types',   dest='exp_types',                          metavar='LIST',         help='only include listed experiment types (comma-separated list)')
parser.add_option('-f', '--files',       dest='file_pattern',                       metavar='FILE_PATTERN', help='only include force curves matching FILE_PATTERN')
parser.add_option('-r', '--range',       dest='file_range',                         metavar='FILE_RANGE',   help='select range of curves (start:stop:step,start:stop:step,...)')
parser.add_option('-o', '--output_dir',  dest='base_output_dir',                    metavar='DIR',          help='Curvalyser output directory')
parser.add_option('-p', '--plots_dir',   dest='plots_dir',                          metavar='DIR',          help='directory for plots (default: plots_filtered)')
parser.add_option('-v', '--verbose',     dest='verbose',       action='store_true',                         help='show verbose messages')
parser.add_option('-V', '--version',     dest='print_version', action='store_true',                         help='show program version')
(options, args) = parser.parse_args()
if options.print_version:
  print 'plot_force_curves v' + __version__
  exit(0)
if options.config_file is not None:
  if os.path.isfile(options.config_file): execfile(options.config_file)
  else:
    print "ERROR: Paramalyser config file '%s' doesn't exist" % options.config_file
    exit(1)
elif os.path.isfile('config-paramalyser'): execfile('config-paramalyser')
else:
  print 'ERROR: Paramalyser config file not found'
  exit(1)
if options.plots_dir is None: plots_dir = 'plots_filtered'
else:                         plots_dir = options.plots_dir
if options.exp_types is None: exp_types = None
else:                         exp_types = [v.strip() for v in options.exp_types.split(',')]
if len(args) >= 1:
  config_files = []
  for arg in args:
    glob_config_dir = glob.glob('config/' + arg)
    if len(glob_config_dir): config_files += glob_config_dir
    else: config_files += glob.glob(arg)
else: config_files = glob.glob('config/*')
config_files = [v for v in config_files if os.path.isfile(v)]
if not len(config_files):
  print 'ERROR: no config files!'
  exit(1)
config_files.sort()
step_filter_curve_limits, step_filter_step_limits, curve_filter_curve_limits, curve_filter_step_limits = get_filter_limits(step_filter_limits, curve_filter_limits)
total_experiments = 0
total_curves             = {}
total_curves_filtered    = {}
total_steps              = {}
total_steps_filtered     = {}
total_adhesions          = {}
total_adhesions_filtered = {}
for config_file_no, config_file in enumerate(config_files):
  t1 = datetime.datetime.now()
  c = get_curvalyser_config(config_file, curvalyser_config)
  if options.file_pattern    is not None: c['file_pattern']    = options.file_pattern
  if options.file_range      is not None: c['file_range']      = options.file_range
  if options.base_output_dir is not None: c['base_output_dir'] = options.base_output_dir
  if c['exp_id'] is None:
    if config_file[:7] == 'config/' or config_file[:7] == 'config\\': exp_id = config_file[7:]
    else: exp_id = config_file
  else: exp_id = c['exp_id']
  if options.config_filter is not None and not eval(options.config_filter):
    print "%s (%.1f%%, exp. %d of %d):\tskipped (excluded by config filter)" % (exp_id, float(config_file_no)/len(config_files)*100, config_file_no+1, len(config_files))
    continue
  if c['file_pattern'] is None: c['file_pattern'] = c['input_dir'] + '/' +  exp_id + '/' + c['file_ext']
  if c['output_dir'] is None: c['output_dir'] = exp_id
  c['output_dir'] = c['base_output_dir'] + '/' + c['output_dir']
  if 'experiment_type' in c and c['experiment_type'] is not None: exp_type = c['experiment_type']
  else: exp_type = None
  if exp_types is not None and exp_type is not None and exp_type not in exp_types:
    print '%s (%.1f%%, exp. %d of %d, %s):\tskipped (excluded by experiment type)' % (exp_id, float(config_file_no)/len(config_files)*100., config_file_no+1, len(config_files), exp_type)
    continue
  files = glob.glob(c['file_pattern'])
  files.sort()
  if c['file_range'] is None:
    start_file, stop_file = nat2py_index(c['file_range_start'], c['file_range_stop'], len(files))
    if c['file_range_step'] is None: c['file_range_step'] = 1 if start_file <= stop_file else -1
    files = files[start_file:stop_file:c['file_range_step']]
  else: files = get_slice(files, c['file_range'])
  if not len(files):
    print '%s (%.1f%%, exp. %d of %d, %s)' % (exp_id, float(config_file_no)/len(config_files)*100., config_file_no+1, len(config_files), exp_type)
    print '[ERROR]   no force curves!'
    continue
  if os.path.exists(c['output_dir']):
    curves = load_curvalyser_curve_data(c['output_dir'] + '/curves.txt')
    steps  = load_curvalyser_step_data(c['output_dir'] + '/steps.txt')
  else: curves = steps = None
  if curves is None or steps is None:
    print '%s (%.1f%%, exp. %d of %d, %s):\t%d file(s)' % (exp_id, float(config_file_no)/len(config_files)*100., config_file_no+1, len(config_files), exp_type, len(files))
    if os.path.exists(c['output_dir']): print "[warning] no Curvalyser data in directory '%s'" % c['output_dir']
    else:                               print "[warning] Curvalyser data directory '" + c['output_dir'] + "' doesn't exist"
    curves = filtered_curves = np.array([], dtype=curve_file_fields).view(np.recarray)
    steps  = filtered_steps  = np.array([], dtype=step_file_fields).view(np.recarray)
  else:
    filtered_curves = curves.copy()
    filtered_steps  = steps.copy()
    if len(step_filter_curve_limits)  or len(step_filter_step_limits):  filtered_curves, filtered_steps = filter_steps(filtered_curves, filtered_steps, step_filter_curve_limits, step_filter_step_limits)
    if len(curve_filter_curve_limits) or len(curve_filter_step_limits): filtered_curves, filtered_steps = filter_curves(filtered_curves, filtered_steps, curve_filter_curve_limits, curve_filter_step_limits)
    print '%s (%.1f%%, exp. %d of %d, %s):\t%d file(s), %d/%d curve(s), %d/%d step(s)' % (exp_id, float(config_file_no)/len(config_files)*100., config_file_no+1, len(config_files), exp_type, len(files), curves.size, filtered_curves.size, steps.size, filtered_steps.size)
  if not os.path.exists(c['output_dir'] + '/' + plots_dir): os.makedirs(c['output_dir'] + '/' + plots_dir)
  for file_no, filename in enumerate(files):
    basename = os.path.basename(filename)
    x, y, x_trace, y_trace, meta, status = load_curve(filename, c)
    if status:
      print '  %s / %s (%5.1f%%, exp. %3d of %3d, file %3d of %3d): skipped' % (exp_id, basename, (float(config_file_no)+float(file_no+1)/len(files))/len(config_files)*100, config_file_no+1, len(config_files), file_no+1, len(files))
      continue
    x, y, x_trace, y_trace = prepare_data(c, x, y, x_trace, y_trace)
    curve_numbers          = np.where(         curves.filename == basename)[0] + 1
    filtered_curve_numbers = np.where(filtered_curves.filename == basename)[0] + 1
    if x.size != x_trace.size or y.size != y_trace.size: x_trace = y_trace = None
    if curve_numbers.size == 1:
      curve = curves[curve_numbers[0]-1]
      selected_steps          =          steps[         steps.curve_no ==          curve_numbers[0]]
      selected_filtered_steps = filtered_steps[filtered_steps.curve_no == filtered_curve_numbers[0]] if filtered_curve_numbers.size == 1 else np.array([])
      print u'  %s / %s (%5.1f%%, exp. %3d of %3d, file %3d of %3d): %2d/%2d step(s), noise sigma: %6.2f %s, denoising: %s, ind thld: %6.2f %s/%s' % (exp_id, basename, (float(config_file_no)+float(file_no+1)/len(files))/len(config_files)*100, config_file_no+1, len(config_files), file_no+1, len(files), selected_steps.size, selected_filtered_steps.size, curve['noise_sigma'], c['unit_y'], fmt_float(curve['denoising_param'], '%6.2f'), curve['indicator_thld'], c['unit_y'], c['unit_x'])
      if options.verbose:
        for i, step in enumerate(selected_steps): print '    step #%2d: extension: %7.3f %s, height: %7.3f %s, avg slope: %7.3f %s/%s, max slope: %7.3f %s/%s%s' % (i+1, step['extension'], c['unit_x'], step['height'], c['unit_y'], step['avg_slope'], c['unit_y'], c['unit_x'], step['max_slope'], c['unit_y'], c['unit_x'], '' if step['lmax_pos'] in filtered_steps.lmax_pos else ' (filtered)')
      if np.isfinite(curve['bl_itcpt']) and np.isfinite(curve['bl_slope']) and np.isfinite(curve['bl_crvtr']):
        if y is not None: y -= curve['bl_itcpt'] + curve['bl_slope'] * x + curve['bl_crvtr'] * x**2
      else:
        if y is not None: y -= np.mean(y[-y.size/10:])
      if y_trace is not None: y_trace -= np.mean(y_trace[-y_trace.size/10:])
      if np.isfinite(curve['contact_pos']):
        if x       is not None: x -= x[curve['contact_pos']]
        if x_trace is not None: x_trace -= x_trace[curve['contact_pos']]
      else:
        if x       is not None: x -= x[0]
        if x_trace is not None: x_trace -= x_trace[0]
      y_denoised = denoise(y, curve['denoising_param'], curve['denoising_param2'], c)
    else:
      selected_steps = selected_filtered_steps = np.array([])
      print '  %s / %s (%5.1f%%, exp. %3d of %3d, file %3d of %3d): no Curvalyser results (%d matches)' % (exp_id, basename, (float(config_file_no)+float(file_no+1)/len(files))/len(config_files)*100, config_file_no+1, len(config_files), file_no+1, len(files), curve_numbers.size)
      if y       is not None: y       -= np.mean(y[-y.size/10:])
      if y_trace is not None: y_trace -= np.mean(y_trace[-y_trace.size/10:])
      if x       is not None: x -= x[0]
      if x_trace is not None: x_trace -= x_trace[0]
      y_denoised = y
    plot_curve(x, y, y_denoised, x_trace, y_trace, selected_steps, selected_filtered_steps, c, c['output_dir'] + '/' + plots_dir + '/' + basename + '-force')
  print '  %d/%d curve(s), %d/%d adhesion curve(s), %d/%d step(s), execution time = %s\n' % (curves.size, filtered_curves.size, np.sum(curves.steps_ctr > 0), np.sum(filtered_curves.steps_ctr > 0), steps.size, filtered_steps.size, datetime.datetime.now() - t1)
  total_experiments += 1
  if exp_type not in total_curves:
    total_curves[exp_type]             = 0
    total_curves_filtered[exp_type]    = 0
    total_steps[exp_type]              = 0
    total_steps_filtered[exp_type]     = 0
    total_adhesions[exp_type]          = 0
    total_adhesions_filtered[exp_type] = 0
  total_curves[exp_type]             += curves.size
  total_curves_filtered[exp_type]    += filtered_curves.size
  total_steps[exp_type]              += steps.size
  total_steps_filtered[exp_type]     += filtered_steps.size
  total_adhesions[exp_type]          += np.sum(curves.steps_ctr > 0)
  total_adhesions_filtered[exp_type] += np.sum(filtered_curves.steps_ctr > 0)
if len(config_files) > 1: print '%d experiment(s), %d/%d curve(s), %d/%d adhesion curve(s), %d/%d step(s), execution time = %s' % (total_experiments, sum(total_curves.values()), sum(total_curves_filtered.values()), sum(total_adhesions.values()), sum(total_adhesions_filtered.values()), sum(total_steps.values()), sum(total_steps_filtered.values()), datetime.datetime.now() - t0)
