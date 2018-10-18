# Embedded file name: C:\Daten\scripts\CurvalyserLib.py
__version__ = '1.9.3'
default_config = {'file_pattern': None,
 'file_range': None,
 'file_range_start': None,
 'file_range_stop': None,
 'file_range_step': None,
 'input_dir': '.',
 'file_ext': '*',
 'file_format': 'crv',
 'columns': None,
 'column_delimiter': None,
 'JPK_segments': (0, 1),
 'JPK_extension_channel': None,
 'JPK_force_channel': None,
 'exp_id': None,
 'base_output_dir': 'output',
 'output_dir': None,
 'nominal_values': {},
 'assert_nominal_values': 1,
 'multiplier_x': None,
 'multiplier_y': None,
 'unit_x': 'a.u.',
 'unit_y': 'a.u.',
 'data_range_start': None,
 'data_range_stop': None,
 'fit_baseline': 1,
 'baseline_fit_params': None,
 'baseline_fit_min_width': 0,
 'baseline_fit_stop_ext': 0,
 'baseline_max_rel_local_RSS': 2.0,
 'baseline_max_avg_RSS': None,
 'baseline_max_local_avg_RSS': None,
 'baseline_step_len': None,
 'baseline_return_to_local_min': True,
 'baseline_verbose': False,
 'denoising_method': 'renoir',
 'denoising_param': None,
 'denoising_param2': float('inf'),
 'denoising_recursions': 1,
 'denoising_wavelet': 'haar',
 'denoising_levels': 0,
 'denoising_thld_mode': 0,
 'denoising_padding_lt': 0,
 'denoising_padding_rt': 0,
 'savgol_order': 2,
 'find_contact_pos': True,
 'max_contact_pos': None,
 'min_contact_force': None,
 'MSF_sigma': 0,
 'MSF_window': 1,
 'MSF_flank_width': 0,
 'MSF_fit_order': 0,
 'MSF_mode': 4,
 'indicator_padding': 0,
 'indicator_margin_lt': 0,
 'indicator_margin_rt': 0,
 'indicator_smoothing_sigma': 0,
 'indicator_threshold': None,
 'indicator_relative_threshold': 5,
 'max_steps': None,
 'step_confinement_lt': 1.0,
 'step_confinement_rt': 1.0,
 'step_fit_clearance_lt': 0,
 'step_fit_clearance_rt': 0,
 'step_fit_min_len': 3,
 'step_fit_max_len': None,
 'step_min_height': None,
 'step_min_width': None,
 'step_max_width': None,
 'step_min_slope': None,
 'indentation_fit': 0,
 'indentation_curve': 0,
 'indentation_lt_fit_width': 0,
 'indentation_rt_fit_width': 0,
 'indentation_fit_avg_window': 0,
 'trace_fit_function': None,
 'trace_fit_init_params': None,
 'retrace_fit_function': None,
 'retrace_fit_init_params': None,
 'single_step_fit_function': None,
 'single_step_init_params': None,
 'plot_force_curves': 1,
 'plot_indicators': 1,
 'show_plots': False,
 'plot_format': 'png',
 'plot_xmin': None,
 'plot_xmax': None,
 'plot_ymin': None,
 'plot_ymax': None,
 'plot_size_force_curves': None,
 'plot_size_indicator_curves': None,
 'fit_drawing_width': None,
 'verbose': 0,
 'spring_constant': None,
 'length_correction': False,
 'simulate_curves': False,
 'paramalyser_curves_range_start': None,
 'paramalyser_curves_range_stop': None,
 'paramalyser_curves_range_step': None}
curve_file_fields = [('steps_ctr', int),
 ('peak_pos', int),
 ('peak_extension', float),
 ('peak_force', float),
 ('indent_force', float),
 ('work', float),
 ('bl_itcpt', float),
 ('bl_slope', float),
 ('bl_crvtr', float),
 ('bl_fit_len', int),
 ('bl_avg_RSS', float),
 ('contact_pos', int),
 ('indent_slope_l', float),
 ('indent_slope_r', float),
 ('trace_fit1', float),
 ('trace_fit2', float),
 ('trace_fit3', float),
 ('trace_fit_RMS', float),
 ('retrace_fit1', float),
 ('retrace_fit2', float),
 ('retrace_fit3', float),
 ('retrace_fit_RMS', float),
 ('noise_sigma', float),
 ('denoising_param', float),
 ('denoising_param2', float),
 ('indicator_thld', float),
 ('filename', str)]
step_file_fields = [('lt_pos', int),
 ('pos', int),
 ('rt_pos', int),
 ('extension', float),
 ('force', float),
 ('height', float),
 ('width', float),
 ('max_slope', float),
 ('slope_l', float),
 ('slope_r', float),
 ('RMS_l', float),
 ('RMS_r', float),
 ('step_no', int),
 ('curve_no', int)]
import glob
import os
import math
import re
import datetime
import collections
import numpy as np
import matplotlib.pyplot as pl
try:
    import SignalLib
except:
    pass

log_entries = []

def log(entry, level = None, log_file = None, display = False):
    global log_entries
    log_entries.append((entry, level))
    if log_file is not None or display:
        output_log([(entry, level)], log_file, display)
    return


def get_log():
    return log_entries


def set_log(entries):
    global log_entries
    log_entries = entries


def del_log():
    global log_entries
    log_entries = []


def output_log(local_log_entries = None, log_file = None, display = True):
    global log_entries
    if local_log_entries is None:
        pass
    else:
        log_entries = local_log_entries
    for entry, level in log_entries:
        if level == 0:
            msg, id = entry
            if log_file is not None:
                log_file.write('[info]    %s:\t%s\n' % (id, msg.encode('Latin-1')))
            if display:
                print '    [info]    %s' % msg
        elif level == 1:
            msg, id = entry
            if log_file is not None:
                log_file.write('[warning] %s:\t%s\n' % (id, msg.encode('Latin-1')))
            if display:
                print '    [warning] %s' % msg
        elif level == 2:
            msg, id = entry
            if log_file is not None:
                log_file.write('[ERROR]   %s:\t%s\n' % (id, msg.encode('Latin-1')))
            if display:
                print '    [ERROR]   %s' % msg
        else:
            if log_file is not None:
                log_file.write(entry.encode('Latin-1') + '\n')
            if display:
                print '  ' + entry

    return


def fmt_float(value, format):
    try:
        if value is None or not np.isfinite(value):
            return str(value)
        float(value)
        return format % value
    except:
        return str(value)

    return


def is_integer(v):
    try:
        int(v)
        return True
    except ValueError:
        return False


def nat2py_index(first = None, last = None, length = None):
    if first is None:
        start = 0
    elif first > 0:
        start = first - 1
    elif first < 0:
        start = first if length is None else length + first
    else:
        start = 0
    if last is None:
        stop = None if length is None else length
    elif last == -1:
        stop = length
    elif last < -1:
        stop = last + 1 if length is None else length + last + 1
    else:
        stop = last
    return (start, stop)


def get_slice(all_values, selection):
    if selection is None or selection.strip() == '':
        return all_values
    else:
        result = []
        for subrange in selection.split(','):
            parts = subrange.split(':', 2)
            if len(parts) == 1:
                if is_integer(parts[0]):
                    start, stop = nat2py_index(int(parts[0]), None, len(all_values))
                    result.append(all_values[start])
            elif len(parts) > 1:
                for i in range(3):
                    if i < len(parts):
                        if is_integer(parts[i]):
                            parts[i] = int(parts[i])
                        else:
                            parts[i] = None
                    else:
                        parts.append(None)

                start, stop = nat2py_index(parts[0], parts[1], len(all_values))
                result.append(all_values[start:stop:parts[2]])

        return result


def read_config_file(config_file = None, config_params = None):
    config = default_config.copy()
    if config_params is not None:
        config.update(config_params)
    if config_file is None:
        config['config_dir'] = None
    else:
        inf = float('inf')
        execfile(config_file, {'config_dir': os.path.dirname(config_file),
         'np': np}, config)
    return config


def read_jpk_old_headers(filename):
    v = {}
    v_trace = {}
    v_retrace = {}
    where = 0
    file = open(filename)
    for line in file:
        if line[0] == '#':
            m = re.search('^#*\\s*([^\\s:=]+)\\s*(=|:)\\s*(.*)$', line.rstrip(' \t\r\n'))
            if m:
                if m.group(1) == 'direction':
                    if m.group(3) == 'trace':
                        where = 1
                    elif m.group(3) == 'retrace':
                        where = 2
                elif where == 1:
                    v_trace[m.group(1)] = m.group(3)
                elif where == 2:
                    v_retrace[m.group(1)] = m.group(3)
                else:
                    v[m.group(1)] = m.group(3)
            else:
                where = 0
        else:
            break

    file.close()
    return (v, v_trace, v_retrace)


def read_jpk_headers(string):
    headers = {}
    for line in string.splitlines():
        line.strip()
        if line[0] == '#':
            continue
        m = re.search('^([^=]+)=(.*)$', line)
        headers[m.group(1)] = m.group(2)

    return headers


def check_nominal_values(nominal_values, meta, basename, assert_nominal_values = True):
    if meta is None:
        log(('meta information missing', basename), 2 if assert_nominal_values else 1)
        return 1
    else:
        status = 0
        for param, restriction in nominal_values.iteritems():
            if param in meta:
                if abs(float(meta[param]) - restriction[0]) > restriction[1]:
                    status |= 2
                    log(("abnormal value for parameter '%s' (%s)" % (param, meta[param]), basename), 2 if assert_nominal_values else 1)
            else:
                status |= 4
                log(("undefined parameter '%s'" % param, basename), 2 if assert_nominal_values else 1)

        return status


def load_curve(filename, c = {}, load = ['y',
 'x',
 'yt',
 'xt',
 'meta']):
    if not os.path.isfile(filename):
        log(("file '%s' does not exist" % filename, os.path.basename(filename)), 2)
        return (None, None, None, None, None, 1)
    else:
        if 'file_format' not in c:
            c['file_format'] = 'crv'
        meta = None
        if c['file_format'] == 'crv':
            import pickle
            file = open(filename, 'rb')
            y = None
            x = None
            y_trace = None
            x_trace = None
            meta = None
            status = 0
            line = file.readline().rstrip()
            if line == 'Curvalyser data file':
                line = file.readline().rstrip()
                if line != '':
                    version = float(line)
                    if version == 1.0:
                        headers = []
                        for i in range(7):
                            line = file.readline().rstrip()
                            if line == '':
                                break
                            headers.append(line)

                        if len(headers) == 7:
                            if 'meta' in load:
                                meta = pickle.loads(file.read(int(headers[0])))
                            if 'nominal_values' in c and c['nominal_values'] is not None:
                                if check_nominal_values(c['nominal_values'], meta, os.path.basename(filename), c['assert_nominal_values']) and c['assert_nominal_values']:
                                    file.close()
                                    return (None,
                                     None,
                                     None,
                                     None,
                                     meta,
                                     7)
                            if 'y' in load:
                                if int(headers[1]) > 0:
                                    y = np.fromstring(file.read(int(headers[1])), np.dtype(headers[2]))
                                else:
                                    status |= 8
                            if 'x' in load:
                                if float(headers[5]) > 0:
                                    x = np.linspace(0, float(headers[5]) * y.size, y.size).astype(headers[2])
                                else:
                                    status |= 16
                            if 'yt' in load:
                                if int(headers[3]) > 0:
                                    y_trace = np.fromstring(file.read(int(headers[3])), np.dtype(headers[4]))
                                else:
                                    status |= 32
                            if 'xt' in load:
                                if float(headers[6]) > 0:
                                    x_trace = np.linspace(0, float(headers[6]) * y_trace.size, y_trace.size).astype(headers[4])
                                else:
                                    status |= 64
                        else:
                            status = 6
                    elif version == 1.1:
                        data = pickle.loads(file.read())
                        if 'y' in load:
                            if 'y_r' in data:
                                y = data['y_r']
                            else:
                                status |= 8
                        if 'yt' in load:
                            if 'y_t' in data:
                                y_trace = data['y_t']
                            else:
                                status |= 16
                        if 'x' in load:
                            if 'dx_r' in data:
                                if data['dx_r'] is None:
                                    x = None
                                else:
                                    x = np.linspace(0, data['dx_r'] * y.size, y.size).astype(type(data['dx_r']))
                            else:
                                status |= 32
                        if 'xt' in load:
                            if 'dx_t' in data:
                                if data['dx_t'] is None:
                                    x_trace = None
                                else:
                                    x_trace = np.linspace(0, data['dx_t'] * y_trace.size, y_trace.size).astype(type(data['dx_t']))
                            else:
                                status |= 64
                        if 'meta' in load:
                            if 'meta' in data:
                                meta = data['meta']
                                if 'nominal_values' in c and c['nominal_values'] is not None:
                                    if check_nominal_values(c['nominal_values'], meta, os.path.basename(filename), c['assert_nominal_values']) and c['assert_nominal_values']:
                                        file.close()
                                        return (None,
                                         None,
                                         None,
                                         None,
                                         meta,
                                         7)
                            else:
                                status |= 128
                    else:
                        status = 5
                else:
                    status = 4
            else:
                status = 3
            file.close()
            if 'multiplier_x' in c and c['multiplier_x'] is not None:
                if x is not None:
                    x *= c['multiplier_x']
                if x_trace is not None:
                    x_trace *= c['multiplier_x']
            if 'multiplier_y' in c and c['multiplier_y'] is not None:
                if y is not None:
                    y *= c['multiplier_y']
                if y_trace is not None:
                    y_trace *= c['multiplier_y']
            if status:
                log(('read error (error code: %d)' % status, os.path.basename(filename)), 2)
            return (x,
             y,
             x_trace,
             y_trace,
             meta,
             status)
        if c['file_format'] == 'text':
            if 'columns' not in c or c['columns'] is None or 'column_delimiter' not in c:
                log(('column numbers and delimiter must be specified', os.path.basename(filename)), 2)
                return (None, None, None, None, None, 3)
            x, y = np.loadtxt(filename, delimiter=c['column_delimiter'], usecols=c['columns'], unpack=True)
            if 'multiplier_x' in c and c['multiplier_x'] is not None:
                x *= c['multiplier_x']
            if 'multiplier_y' in c and c['multiplier_y'] is not None:
                y *= c['multiplier_y']
            return (x,
             y,
             None,
             None,
             meta,
             0)
        if c['file_format'] == 'jpk':
            if 'JPK_segments' not in c or c['JPK_segments'] is None:
                log(('segment numbers must be specified', os.path.basename(filename)), 2)
                return (None, None, None, None, None, 3)
            if 'JPK_extension_channel' not in c or c['JPK_extension_channel'] is None:
                c['JPK_extension_channel'] = 'height'
            if 'JPK_force_channel' not in c or c['JPK_force_channel'] is None:
                c['JPK_force_channel'] = 'vDeflection'
            import zipfile
            if not zipfile.is_zipfile(filename):
                log(("'%s' is not readable" % filename, os.path.basename(filename)), 2)
                return (None, None, None, None, None, 4)
            zf = zipfile.ZipFile(filename)
            if 'segments/' + str(int(c['JPK_segments'][0])) + '/channels/' + c['JPK_extension_channel'] + '.dat' not in zf.namelist():
                log(('extension data for trace segment (# %d) is missing' % c['JPK_segments'][0], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 5)
            if 'segments/' + str(int(c['JPK_segments'][0])) + '/channels/' + c['JPK_force_channel'] + '.dat' not in zf.namelist():
                log(('force data for trace segment (# %d) is missing' % c['JPK_segments'][0], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 6)
            if 'segments/' + str(int(c['JPK_segments'][0])) + '/segment-header.properties' not in zf.namelist():
                log(('meta data for trace segment (# %d) is missing' % c['JPK_segments'][0], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 7)
            if 'segments/' + str(int(c['JPK_segments'][1])) + '/channels/' + c['JPK_extension_channel'] + '.dat' not in zf.namelist():
                log(('extension data for retrace segment (# %d) is missing' % c['JPK_segments'][1], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 8)
            if 'segments/' + str(int(c['JPK_segments'][1])) + '/channels/' + c['JPK_force_channel'] + '.dat' not in zf.namelist():
                log(('force data for retrace segment (# %d) is missing' % c['JPK_segments'][1], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 9)
            if 'segments/' + str(int(c['JPK_segments'][1])) + '/segment-header.properties' not in zf.namelist():
                log(('meta data for retrace segment (# %d) is missing' % c['JPK_segments'][1], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 10)
            headers_trace = read_jpk_headers(zf.read('segments/' + str(int(c['JPK_segments'][0])) + '/segment-header.properties'))
            headers_retrace = read_jpk_headers(zf.read('segments/' + str(int(c['JPK_segments'][1])) + '/segment-header.properties'))
            if 'channel.' + c['JPK_extension_channel'] + '.data.type' not in headers_trace or headers_trace['channel.' + c['JPK_extension_channel'] + '.data.type'] != 'short' and headers_trace['channel.' + c['JPK_extension_channel'] + '.data.type'] != 'float-data':
                log(('extension data type for trace segment (# %d) missing or invalid' % c['JPK_segments'][0], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 11)
            if 'channel.' + c['JPK_force_channel'] + '.data.type' not in headers_trace or headers_trace['channel.' + c['JPK_force_channel'] + '.data.type'] != 'short' and headers_trace['channel.' + c['JPK_force_channel'] + '.data.type'] != 'float-data':
                log(('force data type for trace segment (# %d) missing or invalid' % c['JPK_segments'][0], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 12)
            if 'channel.' + c['JPK_extension_channel'] + '.data.type' not in headers_retrace or headers_retrace['channel.' + c['JPK_extension_channel'] + '.data.type'] != 'short' and headers_retrace['channel.' + c['JPK_extension_channel'] + '.data.type'] != 'float-data':
                log(('extension data type for retrace segment (# %d) missing or invalid' % c['JPK_segments'][1], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 13)
            if 'channel.' + c['JPK_force_channel'] + '.data.type' not in headers_retrace or headers_retrace['channel.' + c['JPK_force_channel'] + '.data.type'] != 'short' and headers_retrace['channel.' + c['JPK_force_channel'] + '.data.type'] != 'float-data':
                log(('force data type for retrace segment (# %d) missing or invalid' % c['JPK_segments'][1], os.path.basename(filename)), 2)
                return (None, None, None, None, None, 14)
            x_trace = zf.read('segments/' + str(int(c['JPK_segments'][0])) + '/channels/' + c['JPK_extension_channel'] + '.dat')
            y_trace = zf.read('segments/' + str(int(c['JPK_segments'][0])) + '/channels/' + c['JPK_force_channel'] + '.dat')
            x = zf.read('segments/' + str(int(c['JPK_segments'][1])) + '/channels/' + c['JPK_extension_channel'] + '.dat')
            y = zf.read('segments/' + str(int(c['JPK_segments'][1])) + '/channels/' + c['JPK_force_channel'] + '.dat')
            zf.close()
            from struct import unpack
            if headers_trace['channel.' + c['JPK_extension_channel'] + '.data.type'] in ('short-data', 'short', 'memory-short-data'):
                x_trace = np.array(unpack('>%dh' % (len(x_trace) / 2), x_trace))
                x_trace = x_trace * float(headers_trace['channel.' + c['JPK_extension_channel'] + '.data.encoder.scaling.multiplier']) + float(headers_trace['channel.' + c['JPK_extension_channel'] + '.data.encoder.scaling.offset'])
            elif headers_trace['channel.' + c['JPK_extension_channel'] + '.data.type'] in ('integer-data', 'memory-integer-data'):
                x_trace = np.array(unpack('>%di' % (len(x_trace) / 4), x_trace))
                x_trace = x_trace * float(headers_trace['channel.' + c['JPK_extension_channel'] + '.data.encoder.scaling.multiplier']) + float(headers_trace['channel.' + c['JPK_extension_channel'] + '.data.encoder.scaling.offset'])
            else:
                x_trace = np.array(unpack('>%df' % (len(x_trace) / 4), x_trace))
            if headers_trace['channel.' + c['JPK_force_channel'] + '.data.type'] in ('short-data', 'short', 'memory-short-data'):
                y_trace = np.array(unpack('>%dh' % (len(y_trace) / 2), y_trace))
                y_trace = y_trace * float(headers_trace['channel.' + c['JPK_force_channel'] + '.data.encoder.scaling.multiplier']) + float(headers_trace['channel.' + c['JPK_force_channel'] + '.data.encoder.scaling.offset'])
            elif headers_trace['channel.' + c['JPK_force_channel'] + '.data.type'] in ('integer-data', 'memory-integer-data'):
                y_trace = np.array(unpack('>%di' % (len(y_trace) / 4), y_trace))
                y_trace = y_trace * float(headers_trace['channel.' + c['JPK_force_channel'] + '.data.encoder.scaling.multiplier']) + float(headers_trace['channel.' + c['JPK_force_channel'] + '.data.encoder.scaling.offset'])
            else:
                y_trace = np.array(unpack('>%df' % (len(y_trace) / 4), y_trace))
            if headers_retrace['channel.' + c['JPK_extension_channel'] + '.data.type'] in ('short-data', 'short', 'memory-short-data'):
                x = np.array(unpack('>%dh' % (len(x) / 2), x))
                x = x * float(headers_retrace['channel.' + c['JPK_extension_channel'] + '.data.encoder.scaling.multiplier']) + float(headers_retrace['channel.' + c['JPK_extension_channel'] + '.data.encoder.scaling.offset'])
            elif headers_retrace['channel.' + c['JPK_extension_channel'] + '.data.type'] in ('integer-data', 'memory-integer-data'):
                x = np.array(unpack('>%di' % (len(x) / 4), x))
                x = x * float(headers_retrace['channel.' + c['JPK_extension_channel'] + '.data.encoder.scaling.multiplier']) + float(headers_retrace['channel.' + c['JPK_extension_channel'] + '.data.encoder.scaling.offset'])
            else:
                x = np.array(unpack('>%df' % (len(x) / 4), x))
            if headers_retrace['channel.' + c['JPK_force_channel'] + '.data.type'] in ('short-data', 'short', 'memory-short-data'):
                y = np.array(unpack('>%dh' % (len(y) / 2), y))
                y = y * float(headers_retrace['channel.' + c['JPK_force_channel'] + '.data.encoder.scaling.multiplier']) + float(headers_retrace['channel.' + c['JPK_force_channel'] + '.data.encoder.scaling.offset'])
            elif headers_retrace['channel.' + c['JPK_force_channel'] + '.data.type'] in ('integer-data', 'memory-integer-data'):
                y = np.array(unpack('>%di' % (len(y) / 4), y))
                y = y * float(headers_retrace['channel.' + c['JPK_force_channel'] + '.data.encoder.scaling.multiplier']) + float(headers_retrace['channel.' + c['JPK_force_channel'] + '.data.encoder.scaling.offset'])
            else:
                y = np.array(unpack('>%df' % (len(y) / 4), y))
            x_trace = np.flipud(x_trace)
            y_trace = np.flipud(y_trace)
            for conversion in headers_trace['channel.' + c['JPK_extension_channel'] + '.conversion-set.conversions.list'].split(' '):
                if conversion == '':
                    continue
                x_trace = x_trace * float(headers_trace['channel.' + c['JPK_extension_channel'] + '.conversion-set.conversion.' + conversion + '.scaling.multiplier']) + float(headers_trace['channel.' + c['JPK_extension_channel'] + '.conversion-set.conversion.' + conversion + '.scaling.offset'])
                x = x * float(headers_retrace['channel.' + c['JPK_extension_channel'] + '.conversion-set.conversion.' + conversion + '.scaling.multiplier']) + float(headers_retrace['channel.' + c['JPK_extension_channel'] + '.conversion-set.conversion.' + conversion + '.scaling.offset'])

            for conversion in headers_trace['channel.' + c['JPK_force_channel'] + '.conversion-set.conversions.list'].split(' '):
                if conversion == '':
                    continue
                y_trace = y_trace * float(headers_trace['channel.' + c['JPK_force_channel'] + '.conversion-set.conversion.' + conversion + '.scaling.multiplier']) + float(headers_trace['channel.' + c['JPK_force_channel'] + '.conversion-set.conversion.' + conversion + '.scaling.offset'])
                y = y * float(headers_retrace['channel.' + c['JPK_force_channel'] + '.conversion-set.conversion.' + conversion + '.scaling.multiplier']) + float(headers_retrace['channel.' + c['JPK_force_channel'] + '.conversion-set.conversion.' + conversion + '.scaling.offset'])

            meta = {}
            for header in headers_trace:
                meta['trace:' + header] = headers_trace[header]

            for header in headers_retrace:
                meta['retrace:' + header] = headers_retrace[header]

            meta['spring_constant'] = float(headers_retrace['channel.' + c['JPK_force_channel'] + '.conversion-set.conversion.force.scaling.multiplier'])
            meta['sensitivity'] = float(headers_retrace['channel.' + c['JPK_force_channel'] + '.conversion-set.conversion.distance.scaling.multiplier'])
            if 'force-segment-header.duration' in headers_trace:
                meta['trace_velocity'] = (x_trace[-1] - x_trace[0]) / float(headers_trace['force-segment-header.duration'])
            if 'force-segment-header.duration' in headers_retrace:
                meta['retrace_velocity'] = (x[-1] - x[0]) / float(headers_retrace['force-segment-header.duration'])
            if 'nominal_values' in c and c['nominal_values'] is not None:
                if check_nominal_values(c['nominal_values'], meta, os.path.basename(filename), c['assert_nominal_values']) and c['assert_nominal_values']:
                    return (None,
                     None,
                     None,
                     None,
                     meta,
                     15)
            if 'multiplier_x' in c and c['multiplier_x'] is not None:
                x *= c['multiplier_x']
                x_trace *= c['multiplier_x']
            if 'multiplier_y' in c and c['multiplier_y'] is not None:
                y *= c['multiplier_y']
                y_trace *= c['multiplier_y']
            return (x,
             y,
             x_trace,
             y_trace,
             meta,
             0)
        if c['file_format'] == 'jpk-old':
            if 'columns' not in c or type(c['columns']) != list and type(c['columns']) != tuple or len(c['columns']) < 2:
                c['columns'] = [None, None]
            direction = 0
            col_x, col_y = c['columns'][0], c['columns'][1]
            headers = {0: {},
             1: {},
             2: {}}
            meta = {}
            x = {1: [],
             2: []}
            y = {1: [],
             2: []}
            checked = False
            file = open(filename)
            for line_no, line in enumerate(file):
                line = line.strip()
                if not len(line):
                    continue
                if line[0] == '#':
                    m = re.search('^#*\\s*([^\\s:=]+)\\s*(=|:)\\s*(.*)$', line)
                    if m:
                        if m.group(1) == 'direction':
                            if m.group(3) == 'trace':
                                direction = 1
                            elif m.group(3) == 'retrace':
                                direction = 2
                            else:
                                direction = 0
                                col_x, col_y = c['columns'][0], c['columns'][1]
                        else:
                            headers[direction][m.group(1)] = m.group(3)
                        if (col_x is None or col_y is None) and m.group(1) == 'columns':
                            col_names = m.group(3).split()
                            if col_x is None:
                                if 'smoothedStrainGaugeHeight' in col_names:
                                    col_x = col_names.index('smoothedStrainGaugeHeight')
                                elif 'strainGaugeHeight' in col_names:
                                    col_x = col_names.index('strainGaugeHeight')
                            if col_y is None and 'vDeflection' in col_names:
                                col_y = col_names.index('vDeflection')
                elif direction == 1 or direction == 2:
                    if col_x is None or col_y is None:
                        file.close()
                        log(('column numbers must be specified', os.path.basename(filename)), 2)
                        return (None, None, None, None, None, 3)
                    if not checked:
                        meta = headers[0]
                        meta.update(headers[1])
                        if 'springConstant' in meta:
                            meta['spring_constant'] = float(meta['springConstant'])
                        if 'zRelativeStart' in meta and 'zRelativeEnd' in meta:
                            if 'traceScanTime' in meta:
                                meta['trace_velocity'] = (float(meta['zRelativeStart']) - float(meta['zRelativeEnd'])) / float(meta['traceScanTime'])
                            if 'retraceScanTime' in meta:
                                meta['retrace_velocity'] = (float(meta['zRelativeStart']) - float(meta['zRelativeEnd'])) / float(meta['retraceScanTime'])
                        if 'nominal_values' in c and c['nominal_values'] is not None:
                            if check_nominal_values(c['nominal_values'], meta, os.path.basename(filename), c['assert_nominal_values']) and c['assert_nominal_values']:
                                file.close()
                                return (None,
                                 None,
                                 None,
                                 None,
                                 meta,
                                 4)
                        checked = True
                    vals = line.split()
                    try:
                        x[direction].append(float(vals[col_x]))
                        y[direction].append(float(vals[col_y]))
                    except IndexError:
                        file.close()
                        log(('read error (not enough columns in line %d)' % (line_no + 1), os.path.basename(filename)), 2)
                        return (None, None, None, None, None, 5)
                    except ValueError:
                        file.close()
                        log(('read error (invalid data in line %d)' % (line_no + 1), os.path.basename(filename)), 2)
                        return (None, None, None, None, None, 6)

            file.close()
            if not len(x[2]):
                log(('read error (no retrace data)', os.path.basename(filename)), 2)
                return (None, None, None, None, None, 7)
            if not len(x[1]):
                log(('no trace data', os.path.basename(filename)), 1)
            if 'traceScanTime' in meta:
                meta['trace_sampling_rate'] = len(x[1]) / float(meta['traceScanTime'])
            if 'retraceScanTime' in meta:
                meta['retrace_sampling_rate'] = len(x[2]) / float(meta['retraceScanTime'])
            if 'multiplier_x' in c and c['multiplier_x'] is not None:
                x[1] = np.array(x[1]) * c['multiplier_x']
                x[2] = np.array(x[2]) * c['multiplier_x']
            if 'multiplier_y' in c and c['multiplier_y'] is not None:
                y[1] = np.array(y[1]) * c['multiplier_y']
                y[2] = np.array(y[2]) * c['multiplier_y']
            return (np.flipud(np.array(x[2])),
             np.flipud(np.array(y[2])),
             np.flipud(np.array(x[1])),
             np.flipud(np.array(y[1])),
             meta,
             0)
        if c['file_format'] == 'jpk-old2':
            meta, trace_headers, retrace_headers = read_jpk_old_headers(filename)
            meta.update(trace_headers)
            if 'springConstant' in meta:
                meta['spring_constant'] = float(meta['springConstant'])
            if 'zRelativeStart' in meta and 'zRelativeEnd' in meta:
                if 'traceScanTime' in meta:
                    meta['trace_velocity'] = (float(meta['zRelativeStart']) - float(meta['zRelativeEnd'])) / float(meta['traceScanTime'])
                if 'retraceScanTime' in meta:
                    meta['retrace_velocity'] = (float(meta['zRelativeStart']) - float(meta['zRelativeEnd'])) / float(meta['retraceScanTime'])
            if 'nominal_values' in c and c['nominal_values'] is not None and check_nominal_values(c['nominal_values'], meta, os.path.basename(filename), c['assert_nominal_values']) and c['assert_nominal_values']:
                return (None,
                 None,
                 None,
                 None,
                 meta,
                 3)
            if 'columns' not in c:
                c['columns'] = None
            x, y = np.loadtxt(filename, dtype=np.float32, usecols=c['columns'], unpack=True)
            if 'kLength' in meta:
                rows = int(meta['kLength'])
            else:
                rows = int(x.size / 2)
            if 'traceScanTime' in meta:
                meta['trace_sampling_rate'] = rows / float(meta['traceScanTime'])
            if 'retraceScanTime' in meta:
                meta['retrace_sampling_rate'] = rows / float(meta['retraceScanTime'])
            if 'multiplier_x' in c and c['multiplier_x'] is not None:
                x *= c['multiplier_x']
            if 'multiplier_y' in c and c['multiplier_y'] is not None:
                y *= c['multiplier_y']
            return (np.flipud(x[-rows:]),
             np.flipud(y[-rows:]),
             np.flipud(x[:rows]),
             np.flipud(y[:rows]),
             meta,
             0)
        log(("unknown file format: '%s'" % c['file_format'], os.path.basename(filename)), 2)
        return (None, None, None, None, None, 2)


def export_crv_file(filename, y_r = None, dx_r = None, y_t = None, dx_t = None, meta = None):
    import pickle
    file = open(filename, 'wb')
    file.write('Curvalyser data file\n')
    file.write('1.1\n')
    file.write(pickle.dumps({'meta': meta,
     'y_r': y_r,
     'dx_r': dx_r,
     'y_t': y_t,
     'dx_t': dx_t}, pickle.HIGHEST_PROTOCOL))
    file.close()


def load_curvalyser_data(filename, dtype):
    data = []
    if not hasattr(dtype, 'names'):
        dtype = np.dtype([ ('', dt) for dt in dtype ])
    number_of_columns = len(dtype.names)
    try:
        f = open(filename)
        for line in f:
            line = line.strip()
            if line == '' or line[0] == '#':
                continue
            values = line.split('\t')
            if len(values) != number_of_columns:
                continue
            data.append(tuple(values))

        f.close()
    except:
        return None

    if len(data):
        new_dtype = [ (v, str(dtype[v])) for v in dtype.names ]
        for i, dt_name in enumerate(dtype.names):
            if dtype[dt_name].type == np.string_ and dtype[dt_name].itemsize == 0:
                new_dtype[i] = (dt_name, dtype[dt_name].type, max([ len(v[i]) for v in data ]))
                dtype = np.dtype(new_dtype)

    return np.array(data, dtype=dtype).view(np.recarray)


def load_curvalyser_curve_data(filename):
    data = load_curvalyser_data(filename, np.dtype(curve_file_fields))
    if data is None:
        return
    else:
        for i in range(data.filename.size):
            data.filename[i] = data.filename[i].strip()[1:].lstrip()

        return data


def load_curvalyser_step_data(filename):
    return load_curvalyser_data(filename, np.dtype(step_file_fields))


def monte_carlo_force_curve(params):
    from math import exp
    import random
    from operator import itemgetter
    params = params.copy()
    kT = 1.3806504e-23 * (params['T'] + 273.15)
    if params['dz'] is None:
        if params['max_z'] is None or params['points'] is None:
            print 'not enough parameters!'
            exit(0)
        else:
            params['dz'] = params['max_z'] / params['points']
    if params['points'] is None:
        if params['max_z'] is None:
            print 'not enough parameters!'
            exit(0)
        else:
            points = int(params['max_z'] / params['dz'])
    else:
        points = params['points']
    if params['max_step_pos'] is None:
        params['max_step_pos'] = points
    x = np.linspace(0, params['max_z'], points).astype(params['dtype'])
    if params['noise_sigma'] > 0:
        y = np.random.normal(0, params['noise_sigma'], points).astype(params['dtype'])
    else:
        y = np.zeros(points, params['dtype'])
    v = params['v']
    k1 = params['k1']
    k2 = params['k2']
    eta = params['eta']
    k_off_0 = params['k_off_0']
    w = params['w']
    delta_t = params['dz'] / v
    if params['variation'] is not None and params['variation'] != 0:
        variation = params['variation'] / 100.0
        k1 *= random.uniform(1 - variation, 1 + variation)
        k2 *= random.uniform(1 - variation, 1 + variation)
        eta *= random.uniform(1 - variation, 1 + variation)
        k_off_0 *= random.uniform(1 - variation, 1 + variation)
        w *= random.uniform(1 - variation, 1 + variation)
    steps = []
    for step in range(random.randrange(params['min_bonds'], params['max_bonds'] + 1)):
        tries = 0
        success = False
        while not success:
            tries += 1
            if tries == params['max_tries']:
                print 'ERROR: unable to generate Monte Carlo curve with more than %d step(s)' % len(steps)
                return False
            i = 0
            flank = np.zeros(points)
            while i < params['max_step_pos']:
                F = k1 * x[i] + eta * v * (1 - exp(-k2 * x[i] / eta / v))
                k_off = k_off_0 * exp(F * w / kT)
                P_rup = k_off * delta_t
                r = random.random()
                flank[i] -= F
                if params['debug'] and not i % 10000.0:
                    print u'%d tries  t=%.3f s  z=%.3f um  F=%.3f pN  k_off=%f s  r=%.1f%%  P_rup=%.1f%%  (%d iterations)' % (tries,
                     x[i] / v,
                     x[i] * 1000000.0,
                     F * 1000000000000.0,
                     k_off,
                     r * 100,
                     P_rup * 100,
                     i)
                if P_rup > r:
                    y += flank
                    if params['multiplier_y'] is None:
                        steps.append((i, F))
                    else:
                        steps.append((i, F * params['multiplier_y']))
                    if params['debug']:
                        print u'%d tries  t=%.3f s  z=%.3f um  F=%.3f pN  k_off=%f s  r=%e%%  P_rup=%e%%  rupture after %d iterations' % (tries,
                         x[i] / v,
                         x[i] * 1000000.0,
                         F * 1000000000000.0,
                         k_off,
                         r * 100,
                         P_rup * 100,
                         i)
                    success = True
                    break
                i += 1

    steps = sorted(steps, key=itemgetter(0))
    if params['multiplier_x'] is not None:
        x *= params['multiplier_x']
    if params['multiplier_y'] is not None:
        y *= params['multiplier_y']
    return (x, y, steps)


def step_curve(params):
    points = params['margin_lt'] + params['margin_rt'] + (params['steps'] + 1) * params['width']
    if 'points' in params and params['points'] is not None:
        points = max(points, params['points'])
    x = np.arange(points, dtype=np.int32)
    y = np.zeros(points, params['dtype'])
    y[0:params['margin_lt']] -= np.ones(params['margin_lt']) * params['height'] * params['steps']
    i = params['margin_lt']
    steps = []
    for step in range(params['steps']):
        y[i:i + params['width']] -= np.ones(params['width']) * params['height'] * (params['steps'] - step)
        i += params['width']
        steps.append((i - 1, params['height']))

    return (x, y, steps)


def RMS(y, free_params = 0):
    if free_params > 0:
        return np.sqrt(np.sum(y ** 2) / (len(y) - free_params))
    return np.sqrt(np.mean(y) ** 2 + np.std(y) ** 2)


def lin_fit(x, y):
    N = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    slope = float(N * np.dot(x, y) - sum_x * sum_y) / (N * np.dot(x, x) - sum_x * sum_x)
    interception = float(sum_y - slope * sum_x) / N
    return (slope, interception)


def lin_fit_opt(x, y, min_fit_len = None):
    if 'SignalLib' in globals():
        return SignalLib.lin_fit_opt(x, y, min_fit_len)
    slope, interception = lin_fit(x[:min_fit_len], y[:min_fit_len])
    return (slope,
     interception,
     len(y[:min_fit_len]),
     RMS(y[:min_fit_len] - slope * x[:min_fit_len] - interception, 2))


def lin_fit_progressive(x, y, max_rel_local_RSS = 2.0, avg_RSS_limit = None, local_avg_RSS_limit = None, step_len = 100, return_to_local_min = False, verbose = False):
    if step_len > 0:
        interval = range(0, y.size - step_len + 1, step_len)
    else:
        interval = range(y.size, -step_len - 1, step_len)
    if not len(interval):
        return ([np.nan, np.nan, np.nan], 0, np.nan)
    else:
        N = 0
        sum_x = 0
        sum_y = 0
        sum_xx = 0
        sum_xy = 0
        last_avg_RSS = np.nan
        previous_N = np.nan
        if verbose:
            all_N = []
            all_avg_RSS = []
            all_local_avg_RSS = []
        for pos in interval:
            if step_len > 0:
                N += step_len
                xp = x[pos:pos + step_len]
                yp = y[pos:pos + step_len]
            else:
                N -= step_len
                xp = x[pos + step_len:pos]
                yp = y[pos + step_len:pos]
            sum_x += sum(xp)
            sum_y += sum(yp)
            sum_xx += np.dot(xp, xp)
            sum_xy += np.dot(xp, yp)
            slope = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x * sum_x)
            interception = (sum_y - slope * sum_x) / N
            if step_len > 0:
                RSS = sum((slope * x[:pos + step_len] + interception - y[:pos + step_len]) ** 2)
                local_RSS = sum((slope * x[pos:pos + step_len] + interception - y[pos:pos + step_len]) ** 2)
            else:
                RSS = sum((slope * x[pos + step_len:] + interception - y[pos + step_len:]) ** 2)
                local_RSS = sum((slope * x[pos + step_len:pos] + interception - y[pos + step_len:pos]) ** 2)
            if verbose:
                all_N.append(N)
                all_avg_RSS.append(RSS / N)
                all_local_avg_RSS.append(local_RSS / abs(step_len))
            if max_rel_local_RSS is not None and local_RSS * N / (RSS * abs(step_len)) > max_rel_local_RSS:
                break
            if avg_RSS_limit is not None and RSS > avg_RSS_limit * N:
                break
            if local_avg_RSS_limit is not None and local_RSS > local_avg_RSS_limit * abs(step_len):
                break
            if not np.isnan(last_avg_RSS) and (not return_to_local_min or RSS / N < last_avg_RSS):
                previous_slope = slope
                previous_interception = interception
                previous_N = N
                previous_RSS = RSS
            last_avg_RSS = RSS / N

        if not np.isnan(previous_N):
            slope = previous_slope
            interception = previous_interception
            N = previous_N
            RSS = previous_RSS
        if verbose:
            pl.figure()
            pl.plot([all_N[0], all_N[-1]], [avg_RSS_limit, avg_RSS_limit], 'b--', label='avg_RSS threshold')
            pl.plot([all_N[0], all_N[-1]], [local_avg_RSS_limit, local_avg_RSS_limit], 'g--', label='local_avg_RSS threshold')
            pl.plot(all_N, all_avg_RSS, 'b.-', label='avg_RSS')
            pl.plot(all_N, all_local_avg_RSS, 'g.-', label='local_avg_RSS')
            pl.legend(loc=0)
            pl.figure()
            pl.plot([all_N[0], all_N[-1]], [max_rel_local_RSS, max_rel_local_RSS], 'b--', label='rel_local_RSS threshold')
            pl.plot(all_N, np.array(all_local_avg_RSS) / np.array(all_avg_RSS), 'b.-', label='rel_local_RSS')
            pl.legend(loc=0)
            pl.show()
        return ([interception, slope, 0], N, RSS / N)


def square_fit_progressive(x, y, max_rel_local_RSS = 2.0, avg_RSS_limit = None, local_avg_RSS_limit = None, step_len = 100, return_to_local_min = False, verbose = False, basename = None):
    x_offset = x[0]
    x -= x_offset
    if step_len > 0:
        interval = range(0, y.size - step_len + 1, step_len)
    elif step_len < 0:
        interval = range(y.size, -step_len - 1, step_len)
    if step_len == 0 or not len(interval):
        return ([np.nan, np.nan, np.nan], 0, np.nan)
    else:
        N = 0
        sum_x = 0
        sum_xx = 0
        sum_xxx = 0
        sum_xxxx = 0
        sum_y = 0
        sum_xy = 0
        sum_xxy = 0
        last_avg_RSS = np.nan
        previous_N = np.nan
        divisor_errors = 0
        a0, a1, a2 = np.nan, np.nan, np.nan
        RSS = np.nan
        if verbose:
            all_N = []
            all_avg_RSS = []
            all_local_avg_RSS = []
        for pos in interval:
            if step_len > 0:
                N += step_len
                xp = x[pos:pos + step_len]
                yp = y[pos:pos + step_len]
            else:
                N -= step_len
                xp = x[pos + step_len:pos]
                yp = y[pos + step_len:pos]
            sum_x += np.sum(xp)
            sum_xx += np.dot(xp, xp)
            sum_xxx += np.sum(xp ** 3)
            sum_xxxx += np.sum(xp ** 4)
            sum_y += np.sum(yp)
            sum_xy += np.dot(xp, yp)
            sum_xxy += np.sum(xp ** 2 * yp)
            divisor = N * sum_xx * sum_xxxx - N * sum_xxx ** 2 - sum_x ** 2 * sum_xxxx + 2 * sum_x * sum_xx * sum_xxx - sum_xx ** 3
            if divisor == 0:
                divisor_errors += 1
                if divisor_errors == 3:
                    log((u'divisor errors in square fit', basename), 2)
                    return ([np.nan, np.nan, np.nan], 0, np.nan)
                continue
            a0 = ((sum_xx * sum_xxxx - sum_xxx ** 2) * sum_y + (sum_xx * sum_xxx - sum_x * sum_xxxx) * sum_xy + (sum_x * sum_xxx - sum_xx ** 2) * sum_xxy) / divisor
            a1 = ((sum_xx * sum_xxx - sum_x * sum_xxxx) * sum_y + (N * sum_xxxx - sum_xx ** 2) * sum_xy + (sum_x * sum_xx - N * sum_xxx) * sum_xxy) / divisor
            a2 = ((sum_x * sum_xxx - sum_xx ** 2) * sum_y + (sum_x * sum_xx - N * sum_xxx) * sum_xy + (N * sum_xx - sum_x ** 2) * sum_xxy) / divisor
            if step_len > 0:
                RSS = sum((a0 + a1 * x[:pos + step_len] + a2 * x[:pos + step_len] ** 2 - y[:pos + step_len]) ** 2)
                local_RSS = sum((a0 + a1 * x[pos:pos + step_len] + a2 * x[pos:pos + step_len] ** 2 - y[pos:pos + step_len]) ** 2)
            else:
                RSS = sum((a0 + a1 * x[pos + step_len:] + a2 * x[pos + step_len:] ** 2 - y[pos + step_len:]) ** 2)
                local_RSS = sum((a0 + a1 * x[pos + step_len:pos] + a2 * x[pos + step_len:pos] ** 2 - y[pos + step_len:pos]) ** 2)
            if verbose:
                all_N.append(N)
                all_avg_RSS.append(RSS / N)
                all_local_avg_RSS.append(local_RSS / abs(step_len))
            if max_rel_local_RSS is not None and local_RSS * N / (RSS * abs(step_len)) > max_rel_local_RSS:
                break
            if avg_RSS_limit is not None and RSS > avg_RSS_limit * N:
                break
            if local_avg_RSS_limit is not None and local_RSS > local_avg_RSS_limit * abs(step_len):
                break
            if not np.isnan(last_avg_RSS) and (not return_to_local_min or RSS / N < last_avg_RSS):
                previous_params = (a0, a1, a2)
                previous_N = N
                previous_RSS = RSS
            last_avg_RSS = RSS / N

        if not np.isnan(previous_N):
            a0, a1, a2 = previous_params
            N = previous_N
            RSS = previous_RSS
        if verbose:
            pl.figure()
            pl.plot([all_N[0], all_N[-1]], [avg_RSS_limit, avg_RSS_limit], 'b--', label='avg_RSS threshold')
            pl.plot([all_N[0], all_N[-1]], [local_avg_RSS_limit, local_avg_RSS_limit], 'g--', label='local_avg_RSS threshold')
            if all_avg_RSS is not None:
                pl.plot(all_N, all_avg_RSS, 'b.-', label='avg_RSS')
            if all_local_avg_RSS is not None:
                pl.plot(all_N, all_local_avg_RSS, 'g.-', label='local_avg_RSS')
            pl.axvline(N, color='k')
            pl.legend(loc=0)
            pl.figure()
            if max_rel_local_RSS is not None:
                pl.plot([all_N[0], all_N[-1]], [max_rel_local_RSS, max_rel_local_RSS], 'b--', label='rel_local_RSS threshold')
            pl.plot(all_N, np.array(all_local_avg_RSS) / np.array(all_avg_RSS), 'b.-', label='rel_local_RSS')
            pl.axvline(N, color='k')
            pl.legend(loc=0)
            pl.show()
        x += x_offset
        a0 -= a1 * x_offset - a2 * x_offset ** 2
        a1 -= 2 * a2 * x_offset
        return ([a0, a1, a2], N, RSS / N)


def denoise(y, c, r, meta):
    if isinstance(c['denoising_param'], collections.Callable):
        r['denoising_param'] = c['denoising_param'](r['noise_sigma'], y.size, meta)
    elif c['denoising_param'] is not None:
        r['denoising_param'] = c['denoising_param']
    else:
        r['denoising_param'] = np.nan
    if isinstance(c['denoising_param2'], collections.Callable):
        r['denoising_param2'] = c['denoising_param2'](r['noise_sigma'], y.size, meta)
    elif c['denoising_param2'] is not None:
        r['denoising_param2'] = c['denoising_param2']
    else:
        r['denoising_param2'] = np.nan
    if not np.isnan(r['denoising_param']) and r['denoising_param'] is not None:
        if c['denoising_padding_lt'] > 0:
            y_padding_lt = 2 * y[0] - y[c['denoising_padding_lt']:0:-1]
            start = c['denoising_padding_lt']
        else:
            y_padding_lt = []
            start = None
        if c['denoising_padding_rt'] > 0:
            y_padding_rt = 2 * y[-1] - y[-2:-c['denoising_padding_rt'] - 2:-1]
            stop = c['denoising_padding_rt']
        else:
            y_padding_rt = []
            stop = None
        y = np.concatenate((y_padding_lt, y, y_padding_rt))
        if c['denoising_method'] == 'rswt' or c['denoising_method'] == 'renoir':
            if 'SignalLib' not in globals():
                print 'ERROR: module "SignalLib" not available'
                exit(0)
        if c['denoising_method'] == 'rswt':
            y_denoised = SignalLib.RSWT(y, threshold=r['denoising_param'], R=c['denoising_recursions'], wavelet_name=c['denoising_wavelet'], levels=c['denoising_levels'], thld_mode=c['denoising_thld_mode'])
        elif c['denoising_method'] == 'renoir':
            y_denoised = SignalLib.ReNoiR_man(y, T0=r['denoising_param'], T1=r['denoising_param2'], R=c['denoising_recursions'], wavelet_name=c['denoising_wavelet'], levels=c['denoising_levels'], thld_mode=c['denoising_thld_mode'])
        elif c['denoising_method'] == 'gauss':
            y_denoised = gauss_filter(y, r['denoising_param'])
        elif c['denoising_method'] == 'savgol':
            y_denoised = savitzky_golay(y, savitzky_golay_coeff(r['denoising_param'], c['savgol_order']))
        elif c['denoising_method'] == 'movavg':
            y_denoised = kernel_smooth(y, np.ones(r['denoising_param'], 'd'))
        else:
            return
        y = y[start:stop]
        y_denoised = y_denoised[start:stop]
    else:
        y_denoised = y
    return y_denoised


def gauss_filter(y, sigma = None, size = None):
    if sigma == 0 or sigma is None and size is None:
        return y
    else:
        if size is None:
            size = int(5 * sigma)
        sigma = size / 10.0 if sigma is None else float(sigma)
        if size > y.size:
            return np.zeros_like(y)
        s = np.empty(size * 2 + 1)
        s[size:] = np.exp(-0.5 * (np.arange(size + 1) / sigma) ** 2)
        s[:size] = s[size + 1:][::-1]
        s /= np.sum(s)
        return np.convolve(y, s, 'same')


def kernel_smooth(y, window):
    window_len = len(window)
    s = np.r_[2 * y[0] - y[window_len:1:-1], y, 2 * y[-1] - y[-1:-window_len:-1]]
    y = np.convolve(window / window.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


def savitzky_golay_coeff(half_window_size, order = 2, deriv = 0):
    try:
        half_window_size = np.abs(np.int(half_window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError('half_window_size and order have to be of type int')

    if half_window_size < 1:
        raise TypeError('half_window_size size must be a positive number')
    if half_window_size * 2 < order + 1:
        raise TypeError('half_window_size is too small for the polynomials order')
    order_range = range(order + 1)
    b = np.mat([ [ k ** i for i in order_range ] for k in range(-half_window_size, half_window_size + 1) ])
    return np.linalg.pinv(b).A[deriv]


def savitzky_golay(y, coeff):
    half_window_size = (len(coeff) - 1) // 2
    firstvals = y[0] - np.abs(y[half_window_size:0:-1] - y[0])
    lastvals = y[-1] + np.abs(y[-2:-half_window_size - 2:-1] - y[-1])
    return np.convolve(coeff, np.concatenate((firstvals, y, lastvals)), mode='valid')


def calculate_distance(x, y, spring_constant):
    return x + y / float(spring_constant)


def fit_baseline(x, y, fit_model, fit_start, max_rel_local_RSS, max_avg_RSS, max_local_avg_RSS, step_len, return_to_local_min, verbose, basename):
    if step_len is None:
        step_len = max(x.size / 200, 1)
    if fit_model == 2:
        fit_params, fit_len, avg_RSS = square_fit_progressive(x[fit_start:], y[fit_start:], max_rel_local_RSS, max_avg_RSS, max_local_avg_RSS, -step_len, return_to_local_min, verbose, basename)
    else:
        fit_params, fit_len, avg_RSS = lin_fit_progressive(x[fit_start:], y[fit_start:], max_rel_local_RSS, max_avg_RSS, max_local_avg_RSS, -step_len, return_to_local_min, verbose)
    return {'fit_params': fit_params,
     'avg_RSS': avg_RSS,
     'fit_len': fit_len}


def find_contact_pos(x, y, min_contact_force = None, start_pos = 0):
    if min_contact_force is None or y[start_pos] > min_contact_force:
        for i in range(start_pos, y.size):
            if y[i] <= 0:
                return i

    return np.nan


def find_contact_pos_trace(x, y, c, r, basename):
    baseline = fit_baseline(x, y, c['fit_baseline'], int(c['baseline_fit_stop_ext'] * r['dst2pts']), c['baseline_max_rel_local_RSS'], c['baseline_max_avg_RSS'], c['baseline_max_local_avg_RSS'], c['baseline_step_len'], c['baseline_return_to_local_min'], c['baseline_verbose'], basename)
    if baseline['fit_len'] < int(c['baseline_fit_min_width'] * r['dst2pts']):
        return np.nan
    else:
        y_corrected = y - baseline['fit_params'][0] - baseline['fit_params'][1] * x - baseline['fit_params'][2] * x ** 2
        if c['min_contact_force'] is None or y_corrected[0] > c['min_contact_force']:
            for i in range(1, y_corrected.size):
                if y_corrected[i] <= 0:
                    return i

        return np.nan


def fit_contact_pos(x, y, contact_pos, lt_fit_len, rt_fit_len):
    if x is not None and y is not None:
        if contact_pos + 1 >= lt_fit_len and lt_fit_len > 0:
            lt_slope, lt_itcpt = lin_fit(x[contact_pos + 1 - lt_fit_len:contact_pos + 1], y[contact_pos + 1 - lt_fit_len:contact_pos + 1])
        else:
            lt_slope, lt_itcpt = np.nan, np.nan
        if contact_pos + rt_fit_len <= x.size and rt_fit_len > 0:
            rt_slope, rt_itcpt = lin_fit(x[contact_pos:contact_pos + rt_fit_len], y[contact_pos:contact_pos + rt_fit_len])
        else:
            rt_slope, rt_itcpt = np.nan, np.nan
        return {'lt_slope': lt_slope,
         'lt_itcpt': lt_itcpt,
         'rt_slope': rt_slope,
         'rt_itcpt': rt_itcpt,
         'pos': contact_pos,
         'x': x[contact_pos],
         'y': y[contact_pos]}
    else:
        return {'lt_slope': np.nan,
         'lt_itcpt': np.nan,
         'rt_slope': np.nan,
         'rt_itcpt': np.nan,
         'pos': np.nan,
         'x': np.nan,
         'y': np.nan}


def fit_indentation(x, y, indentation_force, lt_fit_len, rt_fit_len, averaging_window):
    if x is not None and y is not None:
        start_pos = max(averaging_window, np.argmax(y))
        if y[start_pos] >= indentation_force:
            for fit_pos in range(start_pos, y.size):
                avg_force = np.median(y[fit_pos - averaging_window:fit_pos + averaging_window + 1])
                if avg_force <= indentation_force:
                    lt_slope, lt_itcpt = lin_fit(x[fit_pos + 1 - lt_fit_len:fit_pos + 1], y[fit_pos + 1 - lt_fit_len:fit_pos + 1])
                    rt_slope, rt_itcpt = lin_fit(x[fit_pos:fit_pos + rt_fit_len], y[fit_pos:fit_pos + rt_fit_len])
                    return {'lt_slope': lt_slope,
                     'lt_itcpt': lt_itcpt,
                     'rt_slope': rt_slope,
                     'rt_itcpt': rt_itcpt,
                     'pos': fit_pos,
                     'x': x[fit_pos],
                     'y': avg_force}

    return {'lt_slope': np.nan,
     'lt_itcpt': np.nan,
     'rt_slope': np.nan,
     'rt_itcpt': np.nan,
     'pos': np.nan,
     'x': np.nan,
     'y': np.nan}


def calc_indicator_curve(y, c, r):
    if isinstance(c['MSF_sigma'], collections.Callable):
        r['MSF_sigma'] = c['MSF_sigma'](r['noise_sigma'])
    elif c['MSF_sigma'] > 0:
        r['MSF_sigma'] = c['MSF_sigma']
    else:
        r['MSF_sigma'] = 0
    flank_width = int(c['MSF_flank_width'] / 2)
    if 'SignalLib' in globals():
        result = SignalLib.calc_moving_step_fit(y, sigma=r['MSF_sigma'], window=c['MSF_window'], flank_width=flank_width, fit_order=c['MSF_fit_order'], mode=c['MSF_mode'], padding=c['indicator_padding'])
        indicator = SignalLib.calc_MSF_indicator(result, c['MSF_mode']) * r['dst2pts']
        indicator[np.isnan(indicator)] = 0
    else:
        y = gauss_filter(y, r['MSF_sigma'])
        if c['MSF_window'] > 1 or flank_width > 0:
            indicator = np.zeros(len(y))
            for i in range(flank_width + c['MSF_window'], y.size - flank_width - c['MSF_window'] + 1):
                mean_lt = np.mean(y[i - flank_width - c['MSF_window']:i - flank_width])
                mean_rt = np.mean(y[i + flank_width:i + flank_width + c['MSF_window']])
                indicator[i - 1] = (mean_rt - mean_lt) * r['dst2pts']

        else:
            indicator = np.diff(np.append(y, y[-1])) * r['dst2pts']
    if isinstance(c['indicator_smoothing_sigma'], collections.Callable):
        indicator = gauss_filter(indicator, c['indicator_smoothing_sigma'](r['noise_sigma']))
    elif c['indicator_smoothing_sigma'] > 0:
        indicator = gauss_filter(indicator, c['indicator_smoothing_sigma'])
    r['indicator_sigma'] = calc_noise_level(indicator, r, c, r['dst2pts'], -c['indicator_margin_rt'] if c['indicator_margin_rt'] > 0 else None)
    return indicator


def find_local_maxima(indicator, threshold = None, N = None, margin_lt = 0, margin_rt = 0):
    if margin_lt is None or np.isnan(margin_lt):
        margin_lt = 0
    if margin_rt is None or np.isnan(margin_rt):
        margin_rt = 0
    if 'SignalLib' in globals():
        return SignalLib.find_local_maxima(indicator, num_steps=N, threshold=threshold, margin_lt=margin_lt, margin_rt=margin_rt)
    steps = []
    for i in range(1 + margin_lt, indicator.size - 1 - margin_rt):
        if (threshold is None or indicator[i] >= threshold) and indicator[i - 1] < indicator[i] and indicator[i + 1] < indicator[i]:
            steps.append(i)

    steps = np.array(steps)
    if N is None or not steps.size:
        return steps
    else:
        return np.sort(steps[np.argsort(indicator[steps])][-N:])


def calc_noise_level(y, r, c, dst2pts, stop = None):
    if 'baseline' in r and r['baseline']['fit_len'] > 0:
        start = -r['baseline']['fit_len']
    elif c['baseline_fit_min_width'] > 0:
        start = -int(c['baseline_fit_min_width'] * dst2pts)
    else:
        start = -y.size / 10
    return np.std(y[start:stop])


def find_steps(indicator, x, y, c, r):
    steps = {'pos': [],
     'extension': [],
     'height': [],
     'width': [],
     'max_slope': [],
     'lt_pos': [],
     'rt_pos': [],
     'lt_force': [],
     'rt_force': [],
     'lt_slope': [],
     'lt_itcpt': [],
     'rt_slope': [],
     'rt_itcpt': [],
     'lt_fit_limits': [],
     'rt_fit_limits': [],
     'lt_fit_RMS': [],
     'rt_fit_RMS': []}
    r['indicator_threshold'] = max(c['indicator_threshold'], r['indicator_sigma'] * c['indicator_relative_threshold'])
    step_fit_max_len = len(y) if c['step_fit_max_len'] is None else c['step_fit_max_len']
    local_maxima = find_local_maxima(indicator, r['indicator_threshold'], c['max_steps'], max(r['contact_pos'], c['indicator_margin_lt']), c['indicator_margin_rt'])
    boundaries = [0] + [ pos + 1 for pos in local_maxima ] + [len(y)]
    for i, pos in enumerate(local_maxima):
        lt_pos = pos
        rt_pos = pos
        step_threshold = c['step_confinement_lt'] * indicator[pos]
        while lt_pos > boundaries[i] and indicator[lt_pos - 1] < indicator[lt_pos] and indicator[lt_pos - 1] >= step_threshold:
            lt_pos -= 1

        while lt_pos < pos and indicator[lt_pos + 1] < indicator[lt_pos]:
            lt_pos += 1

        step_threshold = c['step_confinement_rt'] * indicator[pos]
        while rt_pos < boundaries[i + 2] and indicator[rt_pos + 1] < indicator[rt_pos] and indicator[rt_pos + 1] >= step_threshold:
            rt_pos += 1

        while rt_pos > pos and indicator[rt_pos - 1] < indicator[rt_pos]:
            rt_pos -= 1

        fit_limit_lt_lt = max(lt_pos + 1 - step_fit_max_len - c['step_fit_clearance_lt'], boundaries[i])
        fit_limit_lt_rt = max(lt_pos + 1 - c['step_fit_clearance_lt'], boundaries[i])
        fit_limit_rt_lt = min(rt_pos + 1 + c['step_fit_clearance_rt'], boundaries[i + 2])
        fit_limit_rt_rt = min(rt_pos + 1 + step_fit_max_len + c['step_fit_clearance_rt'], boundaries[i + 2])
        if fit_limit_lt_rt - fit_limit_lt_lt >= c['step_fit_min_len']:
            m_lt, t_lt, fit_len_lt, RMS_lt = lin_fit_opt(x[fit_limit_lt_lt:fit_limit_lt_rt][::-1], y[fit_limit_lt_lt:fit_limit_lt_rt][::-1], c['step_fit_min_len'])
            y_lt = m_lt * x[pos] + t_lt
            fit_limit_lt_lt = fit_limit_lt_rt - fit_len_lt
        else:
            m_lt, t_lt, RMS_lt = np.nan, np.nan, np.nan
            y_lt = np.mean(y[fit_limit_lt_lt:fit_limit_lt_rt])
        if fit_limit_rt_rt - fit_limit_rt_lt >= c['step_fit_min_len']:
            m_rt, t_rt, fit_len_rt, RMS_rt = lin_fit_opt(x[fit_limit_rt_lt:fit_limit_rt_rt], y[fit_limit_rt_lt:fit_limit_rt_rt], c['step_fit_min_len'])
            y_rt = m_rt * x[pos] + t_rt
            fit_limit_rt_rt = fit_limit_rt_lt + fit_len_rt
        else:
            m_rt, t_rt, RMS_rt = np.nan, np.nan, np.nan
            y_rt = np.mean(y[fit_limit_rt_lt:fit_limit_rt_rt])
        rt_pos += 1
        step_height = y_rt - y_lt
        step_width = x[rt_pos] - x[lt_pos]
        step_slope = step_height / step_width
        if c['verbose'] >= 2:
            print '  step candidate @ %.3f %s, ind: %.3f (thld: %.3f), scaling factor: %.3f' % (x[pos],
             c['unit_x'],
             indicator[pos],
             r['indicator_threshold'],
             r['indicator_sigma'])
        if c['step_min_height'] is not None and step_height < c['step_min_height']:
            if c['verbose']:
                print '  step @ %.3f %s: step height (%.3f %s) smaller than %.3f %s' % (x[pos],
                 c['unit_x'],
                 step_height,
                 c['unit_y'],
                 c['step_min_height'],
                 c['unit_y'])
            continue
        if c['step_min_width'] is not None and step_width < c['step_min_width']:
            if c['verbose']:
                print u'  step @ %.3f %s: step width (%.3f %s) smaller than %.3f %s' % (x[pos],
                 c['unit_x'],
                 step_width,
                 c['unit_x'],
                 c['step_min_width'],
                 c['unit_x'])
            continue
        if c['step_max_width'] is not None and step_width > c['step_max_width']:
            if c['verbose']:
                print u'  step @ %.3f %s: step width (%.3f %s) bigger than %.3f %s' % (x[pos],
                 c['unit_x'],
                 step_width,
                 c['unit_x'],
                 c['step_max_width'],
                 c['unit_x'])
            continue
        if c['step_min_slope'] is not None and step_slope < c['step_min_slope']:
            if c['verbose']:
                print u'  step @ %.3f %s: average step slope (%.3f) smaller than %.3f' % (x[pos],
                 c['unit_x'],
                 step_slope,
                 c['step_min_slope'])
            continue
        steps['pos'].append(pos)
        steps['extension'].append(x[pos])
        steps['height'].append(step_height)
        steps['width'].append(step_width)
        steps['max_slope'].append(indicator[pos])
        steps['lt_pos'].append(lt_pos)
        steps['rt_pos'].append(rt_pos)
        steps['lt_force'].append(y_lt)
        steps['rt_force'].append(y_rt)
        steps['lt_slope'].append(m_lt)
        steps['rt_slope'].append(m_rt)
        steps['lt_itcpt'].append(t_lt)
        steps['rt_itcpt'].append(t_rt)
        steps['lt_fit_limits'].append((fit_limit_lt_lt, fit_limit_lt_rt))
        steps['rt_fit_limits'].append((fit_limit_rt_lt, fit_limit_rt_rt))
        steps['lt_fit_RMS'].append(RMS_lt)
        steps['rt_fit_RMS'].append(RMS_rt)
        if c['verbose']:
            print u'  STEP FOUND @ %.3f %s:\theight = %.3f %s,\twidth = %.3f %s,\tmax slope = %.3f\t(%.3fx)' % (x[pos],
             c['unit_x'],
             steps['height'][-1],
             c['unit_y'],
             steps['width'][-1],
             c['unit_x'],
             indicator[pos],
             indicator[pos] / r['indicator_sigma'] if r['indicator_sigma'] > 0 else np.nan)

    return steps


def filter_steps(steps, nmbr, indicator):
    if len(steps['pos']) == 0:
        return (steps, max(indicator))
    filtered_steps = {}
    for key in steps.keys():
        if len(steps[key]):
            filtered_steps[key] = []

    if nmbr < 1:
        return (filtered_steps, max(indicator))
    index = min(nmbr, len(steps['pos']))
    threshold = np.sort(indicator[steps['pos']])[-index]
    for i, pos in enumerate(steps['pos']):
        if indicator[pos] >= threshold:
            for key in filtered_steps:
                filtered_steps[key].append(steps[key][i])

    return (filtered_steps, threshold)


def fit_curve(x, y, start, stop, fitfunc, p, meta, r, basename, description):
    from scipy.optimize import leastsq
    errfunc = lambda p, x, y, meta, r: fitfunc(p, x, meta, r) - y
    try:
        p, cov_x, info, msg, result = leastsq(errfunc, p, args=(x[start:stop],
         y[start:stop],
         meta,
         r), full_output=1, maxfev=0)
    except:
        log(('least squares fitting error for %s' % description, basename), 1)
        return ([np.nan] * len(p), np.nan)

    RSS = np.sum(info['fvec'] * info['fvec'])
    dof = len(y) - len(p)
    RMS = np.sqrt(RSS / dof)
    return (p, RMS)


def plot_curve(x, y, y_denoised, x_trace, y_trace, r, c, filename = None, fig_num = None, auto_limits = False, meta = None):
    if c['plot_size_force_curves'] is None:
        fig_size = None
    else:
        fig_size = (float(c['plot_size_force_curves'][0]) / pl.rcParams['savefig.dpi'], float(c['plot_size_force_curves'][1]) / pl.rcParams['savefig.dpi'])
    pl.figure(num=fig_num, figsize=fig_size)
    if x_trace is not None and y_trace is not None:
        pl.plot(x_trace, y_trace, '#7FFF7F')
    if y is not None:
        pl.plot(x, y, 'b')
    if y_denoised is not None and y_denoised is not y:
        pl.plot(x, y_denoised, 'k')
    if 'baseline' in r:
        if r['baseline']['valid']:
            pl.plot([x[0], x[-1]], [0, 0], '--', c='.75')
            if r['baseline']['fit_len'] > 0:
                pl.plot([x[-r['baseline']['fit_len']]], [0], 'o', c='.75')
        else:
            pl.plot(x, r['baseline']['fit_params'][0] + r['baseline']['fit_params'][1] * x + r['baseline']['fit_params'][2] * x ** 2, '--', c='.75')
            if r['baseline']['fit_len'] > 0:
                x_fit_stop = x[-r['baseline']['fit_len']]
                pl.plot([x_fit_stop], [r['baseline']['fit_params'][0] + r['baseline']['fit_params'][1] * x_fit_stop + r['baseline']['fit_params'][2] * x_fit_stop ** 2], 'o', c='.75')
    if 'indentation_fit' in r and not np.isnan(r['indentation_fit']['pos']) and not np.isnan(r['indentation_fit']['x']) and not np.isnan(r['indentation_fit']['y']):
        pl.plot(r['indentation_fit']['x'], r['indentation_fit']['y'], 'mo')
        if not np.isnan(r['indentation_fit']['lt_slope']) and not np.isnan(r['indentation_fit']['lt_itcpt']):
            line_x = x_trace[r['indentation_fit']['pos'] - r['indentation_lt_fit_len']:r['indentation_fit']['pos']] if c['indentation_curve'] == 1 else x[r['indentation_fit']['pos'] - r['indentation_lt_fit_len']:r['indentation_fit']['pos']]
            line_y = line_x * r['indentation_fit']['lt_slope'] + r['indentation_fit']['lt_itcpt']
            pl.plot(line_x, line_y, 'm', lw=2)
        if not np.isnan(r['indentation_fit']['rt_slope']) and not np.isnan(r['indentation_fit']['rt_itcpt']):
            line_x = x_trace[r['indentation_fit']['pos']:r['indentation_fit']['pos'] + r['indentation_rt_fit_len']] if c['indentation_curve'] == 1 else x[r['indentation_fit']['pos']:r['indentation_fit']['pos'] + r['indentation_rt_fit_len']]
            line_y = line_x * r['indentation_fit']['rt_slope'] + r['indentation_fit']['rt_itcpt']
            pl.plot(line_x, line_y, 'm', lw=2)
    if 'contact_pos' in r and not np.isnan(r['contact_pos']):
        pl.plot([x[r['contact_pos']]], [0], 'ko')
        if x_trace is not None and y_trace is not None and x_trace.size == x.size:
            pl.plot([x_trace[r['contact_pos']]], [y_trace[r['contact_pos']]], 'ko')
    if c['trace_fit_function'] is not None and 'trace_fit' in r and not np.isnan(r['contact_pos']):
        pl.plot(x_trace[0:r['contact_pos']], c['trace_fit_function'](r['trace_fit'], x_trace[0:r['contact_pos']], meta, r), 'lime', lw=2, alpha=0.5)
    if c['retrace_fit_function'] is not None:
        if 'retrace_fit' in r and not np.isnan(r['contact_pos']):
            pl.plot(x[0:r['contact_pos']], c['retrace_fit_function'](r['retrace_fit'], x[0:r['contact_pos']], meta, r), 'orange', lw=2, alpha=0.5)
    elif c['single_step_fit_function'] is not None:
        if 'retrace_fit' in r and r['len_steps'] == 1:
            pl.plot(x[r['contact_pos']:r['steps']['pos'][0]], c['single_step_fit_function'](r['retrace_fit'], x[r['contact_pos']:r['steps']['pos'][0]], meta, r), 'orange', lw=2, alpha=0.5)
    if 'steps' in r and 'len_steps' in r and r['len_steps'] > 0:
        fit_drawing_len = None if c['fit_drawing_width'] is None else int(c['fit_drawing_width'] * r['dst2pts'])
        for i in range(r['len_steps']):
            pl.axvline(x[r['steps']['pos'][i]], color='yellow', lw=3, alpha=0.25)
            if fit_drawing_len != 0:
                if fit_drawing_len is None:
                    x_lt = x[r['steps']['lt_fit_limits'][i][0]:r['steps']['lt_fit_limits'][i][1]]
                    x_rt = x[r['steps']['rt_fit_limits'][i][0]:r['steps']['rt_fit_limits'][i][1]]
                else:
                    x_lt = x[r['steps']['pos'][i] + 1 - fit_drawing_len:r['steps']['pos'][i] + 1]
                    x_rt = x[r['steps']['pos'][i] + 1:r['steps']['pos'][i] + 1 + fit_drawing_len]
                if not np.isnan(r['steps']['lt_slope'][i]) and not np.isnan(r['steps']['lt_itcpt'][i]):
                    pl.plot(x_lt, x_lt * r['steps']['lt_slope'][i] + r['steps']['lt_itcpt'][i], 'r', lw=2, alpha=0.75)
                if not np.isnan(r['steps']['rt_slope'][i]) and not np.isnan(r['steps']['rt_itcpt'][i]):
                    pl.plot(x_rt, x_rt * r['steps']['rt_slope'][i] + r['steps']['rt_itcpt'][i], 'g', lw=2, alpha=0.75)
            pl.plot(x[r['steps']['pos']], r['steps']['lt_force'], 'rx', ms=10, mew=2)
            pl.plot(x[r['steps']['pos']], r['steps']['rt_force'], 'g+', ms=10, mew=2)
            pl.plot(x[r['steps']['lt_pos']], r['steps']['lt_force'], 'rx')
            pl.plot(x[r['steps']['rt_pos']], r['steps']['rt_force'], 'g+')

    if not auto_limits:
        pl.axis(xmin=c['plot_xmin'], xmax=c['plot_xmax'], ymin=c['plot_ymin'], ymax=c['plot_ymax'])
    pl.xlabel(u'z [%s]' % c['unit_x'])
    pl.ylabel('F [%s]' % c['unit_y'])
    if filename is not None:
        pl.savefig(filename + '.' + c['plot_format'], format=c['plot_format'])
    if fig_num is None:
        pl.close()
    return


def plot_indicator_curve(x, indicator, r, c, filename = None, fig_num = None):
    if c['plot_size_indicator_curves'] is None:
        fig_size = None
    else:
        fig_size = (float(c['plot_size_indicator_curves'][0]) / pl.rcParams['savefig.dpi'], float(c['plot_size_indicator_curves'][1]) / pl.rcParams['savefig.dpi'])
    pl.figure(num=fig_num, figsize=fig_size)
    pl.plot([x[0], x[-1]], [0, 0], '--', c='.5')
    if c['show_plots']:
        pl.plot(x, indicator, 'b.-', ms=3)
    else:
        pl.plot(x, indicator, 'b')
    if 'len_steps' in r and r['len_steps'] and 'steps' in r and 'pos' in r['steps']:
        for i in range(r['len_steps']):
            pl.axvline(x[r['steps']['pos'][i]], color='yellow', lw=3, alpha=0.5)

        pl.plot(x[r['steps']['lt_pos']], indicator[r['steps']['lt_pos']], 'rx', mew=2, ms=10)
        pl.plot(x[r['steps']['rt_pos']], indicator[r['steps']['rt_pos']], 'g+', mew=2, ms=10)
    pl.plot([x[0], x[-1]], [r['indicator_threshold'], r['indicator_threshold']], 'k')
    pl.xlim(c['plot_xmin'], c['plot_xmax'])
    if r['indicator_threshold'] > 0:
        indicator_stop = -c['indicator_margin_rt'] if c['indicator_margin_rt'] > 0 else None
        pl.ylim(-r['indicator_threshold'], max(max(indicator[c['indicator_margin_lt']:indicator_stop]), r['indicator_threshold']) * 1.05)
    pl.xlabel(u'z [%s]' % c['unit_x'])
    pl.ylabel(u'slope [a.u.]')
    if filename is not None:
        pl.savefig(filename + '.' + c['plot_format'], format=c['plot_format'])
    if fig_num is None:
        pl.close()
    return


def prepare_data(c, x = None, y = None, x_trace = None, y_trace = None):
    if x is not None:
        x = x[c['data_range_start']:c['data_range_stop']]
    if y is not None:
        y = y[c['data_range_start']:c['data_range_stop']]
    if x_trace is not None:
        x_trace = x_trace[c['data_range_start']:c['data_range_stop']]
    if y_trace is not None:
        y_trace = y_trace[c['data_range_start']:c['data_range_stop']]
    return (x,
     y,
     x_trace,
     y_trace)


def analyse(x, y, x_trace, y_trace, c, basename, meta = {}, curves_file = None, steps_file = None, files_ctr = 0, enhanced_results = []):
    r = {'status': 0}
    if not x.size or not y.size:
        r['status'] |= 16
        log(('invalid retrace data', basename), 2)
        return r
    if y.size != x.size:
        r['status'] |= 32
        log(('extension and deflection columns of retrace data have different lengths', basename), 2)
        return r
    r['rows'] = x.size
    if x_trace is not None and y_trace is not None and (x_trace.size != y_trace.size or not x_trace.size):
        x_trace = y_trace = None
        log(('invalid trace data', basename), 1)
    if c['file_format'] == 'jpk-old' and (x_trace is not None and x_trace.size != x.size or y_trace is not None and y_trace.size != y.size):
        log(('lengths of trace and retrace data do not match', basename), 1)
    r['dst2pts'] = float(x.size - 1) / abs(x[-1] - x[0])
    if c['fit_baseline']:
        r['baseline'] = fit_baseline(x, y, c['fit_baseline'], int(c['baseline_fit_stop_ext'] * r['dst2pts']), c['baseline_max_rel_local_RSS'], c['baseline_max_avg_RSS'], c['baseline_max_local_avg_RSS'], c['baseline_step_len'], c['baseline_return_to_local_min'], c['baseline_verbose'], basename)
        if r['baseline']['fit_len'] < int(c['baseline_fit_min_width'] * r['dst2pts']):
            r['baseline']['valid'] = False
            r['status'] |= 1
            log((u'baseline fit too short (%d points, %.3f %s)' % (r['baseline']['fit_len'], r['baseline']['fit_len'] / r['dst2pts'], c['unit_x']), basename), 2)
            if c['length_correction']:
                log(('length correction requires successful baseline fit', basename), 2)
        else:
            r['baseline']['valid'] = True
    elif c['baseline_fit_params'] is not None:
        r['baseline'] = {'fit_params': c['baseline_fit_params'],
         'avg_RSS': np.nan,
         'fit_len': 0,
         'valid': True}
    else:
        r['baseline'] = {'fit_params': (np.nan, np.nan, np.nan),
         'avg_RSS': np.nan,
         'fit_len': 0,
         'valid': False}
    if r['status']:
        r['noise_sigma'] = np.nan
    else:
        if r['baseline']['valid']:
            baseline = r['baseline']['fit_params'][0] + r['baseline']['fit_params'][1] * x + r['baseline']['fit_params'][2] * x ** 2
            y -= baseline
            if y_trace is not None:
                y_trace -= np.mean(y_trace[-y_trace.size / 10:])
        r['noise_sigma'] = calc_noise_level(y, r, c, r['dst2pts'])
        if c['length_correction']:
            if c['spring_constant'] is not None:
                spring_constant = c['spring_constant']
            elif 'spring_constant' in meta:
                spring_constant = meta['spring_constant']
            else:
                spring_constant = None
            if spring_constant is not None:
                x = calculate_distance(x, y, spring_constant)
                if x_trace is not None:
                    x_trace = calculate_distance(x_trace, y, spring_constant)
            else:
                r['status'] |= 2
                log(('length correction requires spring constant', basename), 2)
    if r['status']:
        y_denoised = None
        r['peak_pos'] = np.nan
        r['peak_force'] = np.nan
        if c['find_contact_pos']:
            r['contact_pos'] = np.nan
        else:
            r['contact_pos'] = 0
        if 'baseline' in r:
            r['baseline']['fit_params'][0] += r['baseline']['fit_params'][1] * x[0] + r['baseline']['fit_params'][2] * x[0] ** 2
            r['baseline']['fit_params'][1] += 2 * r['baseline']['fit_params'][2] * x[0]
        x -= x[0]
        if x_trace is not None:
            x_trace -= x_trace[0]
    else:
        y_denoised = denoise(y, c, r, meta)
        if y_denoised is None:
            r['status'] |= 4
            log(('invalid denoising method', basename), 2)
            return r
        if 'y_denoised' in enhanced_results:
            r['y_denoised'] = y_denoised
        r['peak_pos'] = y_denoised.argmin()
        r['peak_force'] = -y_denoised[r['peak_pos']]
        if c['find_contact_pos']:
            indicator_stop = -c['indicator_margin_rt'] if c['indicator_margin_rt'] > 0 else None
            r['contact_pos'] = find_contact_pos(x, y_denoised, c['min_contact_force'])
            if np.isnan(r['contact_pos']):
                log(('contact point not detected in denoised retrace curve; trying to detect in original curve', basename), 0)
                r['contact_pos'] = find_contact_pos(x, y, c['min_contact_force'])
        else:
            r['contact_pos'] = 0
        if np.isnan(r['contact_pos']) or c['max_contact_pos'] is not None and x[r['contact_pos']] - x[0] > c['max_contact_pos']:
            r['status'] |= 8
            x_offset = x[0]
            x -= x_offset
            if x_trace is not None:
                x_trace -= x_offset
            if np.isnan(r['contact_pos']):
                log(('contact point calibration failed', basename), 2)
            else:
                log((u'contact point found, but too far right (row: %d, extension: %.3f %s)' % (r['contact_pos'], x[r['contact_pos']], c['unit_x']), basename), 2)
        else:
            x_offset = x[r['contact_pos']]
            x -= x_offset
            if x_trace is not None:
                if r['contact_pos'] < x_trace.size:
                    x_trace -= x_offset
                else:
                    x_trace = y_trace = None
                    log(('trace data too short', basename), 1)
    if r['status']:
        r['peak_ext'] = np.nan
        r['indentation_fit'] = {'lt_slope': np.nan,
         'lt_itcpt': np.nan,
         'rt_slope': np.nan,
         'rt_itcpt': np.nan,
         'pos': np.nan,
         'x': np.nan,
         'y': np.nan}
        r['work'] = np.nan
        r['len_steps'] = 0
        if c['retrace_fit_function'] is not None:
            r['retrace_fit'], r['retrace_fit_RMS'] = ([np.nan, np.nan, np.nan], np.nan)
            log(('retrace fit requires successful baseline fit and contact point calibration', basename), 1)
        if c['trace_fit_function'] is not None:
            r['trace_fit'], r['trace_fit_RMS'] = ([np.nan, np.nan, np.nan], np.nan)
            log(('trace fit requires successful baseline fit and contact point calibration', basename), 1)
        if c['single_step_fit_function'] is not None:
            r['retrace_fit'], r['retrace_fit_RMS'] = ([np.nan, np.nan, np.nan], np.nan)
            log(('single-step fit requires successful baseline fit and contact point calibration', basename), 1)
        if c['plot_force_curves']:
            if not os.path.exists(c['output_dir'] + '/plots'):
                os.makedirs(c['output_dir'] + '/plots')
            plot_curve(x, y, y_denoised, x_trace, y_trace, r, c, c['output_dir'] + '/plots/' + basename + '-force (failed)', fig_num=1, auto_limits=True, meta=meta)
            if c['show_plots']:
                pl.show()
            pl.close(1)
        return r
    r['peak_ext'] = x[r['peak_pos']]
    r['indentation_lt_fit_len'] = int(c['indentation_lt_fit_width'] * r['dst2pts'])
    r['indentation_rt_fit_len'] = int(c['indentation_rt_fit_width'] * r['dst2pts'])
    if r['indentation_lt_fit_len'] == 0:
        r['indentation_lt_fit_len'] = r['contact_pos']
    if c['indentation_fit'] == 0:
        r['indentation_fit'] = fit_contact_pos(x_trace if c['indentation_curve'] == 1 else x, y_trace if c['indentation_curve'] == 1 else y_denoised, r['contact_pos'], r['indentation_lt_fit_len'], r['indentation_rt_fit_len'])
    else:
        r['indentation_fit'] = fit_indentation(x_trace if c['indentation_curve'] == 1 else x, y_trace if c['indentation_curve'] == 1 else y, c['indentation_fit'], r['indentation_lt_fit_len'], r['indentation_rt_fit_len'], c['indentation_fit_avg_window'])
    r['work'] = -sum(y[r['contact_pos']:]) * x[-1] / y[r['contact_pos']:].size
    r['indent_force'] = y[0]
    indicator = calc_indicator_curve(y, c, r)
    if 'indicator' in enhanced_results:
        r['indicator'] = indicator
    r['steps'] = find_steps(indicator, x, y, c, r)
    r['len_steps'] = len(r['steps']['pos'])
    if c['retrace_fit_function'] is not None:
        r['retrace_fit'], r['retrace_fit_RMS'] = fit_curve(x, y, 0, r['contact_pos'], c['retrace_fit_function'], c['retrace_fit_init_params'], meta, r, basename, 'retrace fit function')
    elif c['single_step_fit_function'] is not None and r['len_steps'] == 1:
        r['retrace_fit'], r['retrace_fit_RMS'] = fit_curve(x, y, r['contact_pos'], r['steps']['pos'][0], c['single_step_fit_function'], c['single_step_fit_init_params'], meta, r, basename, 'single step fit function')
    else:
        r['retrace_fit'], r['retrace_fit_RMS'] = ([np.nan, np.nan, np.nan], np.nan)
    if c['trace_fit_function'] is not None:
        if x_trace is not None and y_trace is not None:
            r['trace_fit'], r['trace_fit_RMS'] = fit_curve(x_trace, y_trace, 0, r['contact_pos'], c['trace_fit_function'], c['trace_fit_init_params'], meta, r, basename, 'trace fit function')
        else:
            r['trace_fit'], r['trace_fit_RMS'] = ([np.nan, np.nan, np.nan], np.nan)
            log(('trace fit requires trace data', basename), 1)
    else:
        r['trace_fit'], r['trace_fit_RMS'] = ([np.nan, np.nan, np.nan], np.nan)
    if len(r['trace_fit']) < 3:
        r['trace_fit'] = np.append(r['trace_fit'], [np.nan] * (3 - len(r['trace_fit'])))
    if len(r['retrace_fit']) < 3:
        r['retrace_fit'] = np.append(r['retrace_fit'], [np.nan] * (3 - len(r['retrace_fit'])))
    if curves_file is not None:
        curves_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t# %s\n' % (r['len_steps'],
         r['peak_pos'],
         r['peak_ext'],
         r['peak_force'],
         r['indent_force'],
         r['work'],
         r['baseline']['fit_params'][0],
         r['baseline']['fit_params'][1],
         r['baseline']['fit_params'][2],
         r['baseline']['fit_len'],
         r['baseline']['avg_RSS'],
         r['contact_pos'],
         r['indentation_fit']['lt_slope'],
         r['indentation_fit']['rt_slope'],
         r['trace_fit'][0],
         r['trace_fit'][1],
         r['trace_fit'][2],
         r['trace_fit_RMS'],
         r['retrace_fit'][0],
         r['retrace_fit'][1],
         r['retrace_fit'][2],
         r['retrace_fit_RMS'],
         r['noise_sigma'],
         r['denoising_param'],
         r['denoising_param2'],
         r['indicator_threshold'],
         basename))
    if steps_file is not None:
        steps_file.write('# %s\n' % basename)
        for i in range(r['len_steps']):
            steps_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (r['steps']['lt_pos'][i],
             r['steps']['pos'][i],
             r['steps']['rt_pos'][i],
             r['steps']['extension'][i],
             -r['steps']['lt_force'][i],
             r['steps']['height'][i],
             r['steps']['width'][i],
             r['steps']['max_slope'][i],
             r['steps']['lt_slope'][i],
             r['steps']['rt_slope'][i],
             r['steps']['lt_fit_RMS'][i],
             r['steps']['rt_fit_RMS'][i],
             i + 1,
             files_ctr + 1))

    if c['plot_force_curves'] or c['plot_indicators']:
        if not os.path.exists(c['output_dir'] + '/plots'):
            os.makedirs(c['output_dir'] + '/plots')
        plot_curve(x, y, y_denoised, x_trace, y_trace, r, c, c['output_dir'] + '/plots/' + basename + '-force', 1, meta=meta)
        if c['plot_indicators']:
            plot_indicator_curve(x, indicator, r, c, c['output_dir'] + '/plots/' + basename + '-indicator', 2)
        if c['show_plots']:
            pl.show()
        pl.close(1)
        if c['plot_indicators']:
            pl.close(2)
    return r


def CurvalyserCLI():
    from optparse import OptionParser
    from time import strftime
    import pprint
    t0 = datetime.datetime.now()
    parser = OptionParser(usage='usage: %prog [options] [config files]')
    parser.add_option('-l', '--logfile', dest='logfile', metavar='FILE', help='set logfile to FILE')
    parser.add_option('-c', '--cfg_filter', dest='config_filter', metavar='CODE', help='Python expression to select config files')
    parser.add_option('-e', '--exp_types', dest='exp_types', metavar='LIST', help='only include listed experiment types (comma-separated list)')
    parser.add_option('-O', '--overwrite', dest='overwrite', action='store_true', help='overwrite existing experiments')
    parser.add_option('-f', '--files', dest='file_pattern', metavar='FILE_PATTERN', help='only include force curves matching FILE_PATTERN')
    parser.add_option('-r', '--range', dest='file_range', metavar='FILE_RANGE', help='select range of curves (start:stop:step,start:stop:step,...)')
    parser.add_option('-o', '--output_dir', dest='base_output_dir', metavar='DIR', help='base output directory')
    parser.add_option('-v', '--verbose', dest='verbose', action='count', help='level of verbosity (-v or -vv)')
    parser.add_option('-d', '--denoise', dest='denoising_param', type='float', metavar='VALUE', help='set primary denoising parameter')
    parser.add_option('-i', '--ind_thld', dest='ind_thld', type='float', metavar='VALUE', help='set relative indicator threshold')
    parser.add_option('-p', '--plot', dest='plot', metavar='LIST', help='select plots (f = force curves, i = indicators)')
    parser.add_option('-s', '--show_plots', dest='show_plots', action='store_true', help='show plots in GUI')
    parser.add_option('-V', '--version', dest='print_version', action='store_true', help='show program version')
    options, args = parser.parse_args()
    if options.verbose:
        np.seterr(all='warn')
    if options.print_version:
        print 'Curvalyser v' + __version__
        exit(0)
    if len(args) >= 1:
        config_files = []
        for arg in args:
            glob_config_dir = glob.glob('config/' + arg)
            if len(glob_config_dir):
                config_files += glob_config_dir
            else:
                config_files += glob.glob(arg)

    else:
        config_files = glob.glob('config/*')
    config_files = [ v for v in config_files if os.path.isfile(v) ]
    if not len(config_files):
        print 'ERROR: no config files!'
        exit(1)
    config_files.sort()
    if options.logfile is not None and options.logfile != '':
        summarised_log_file = open(options.logfile, 'a')
        summarised_log_file.write(strftime('%d.%m.%Y %H:%M:%S') + '   Curvalyser v%s\n' % __version__)
    else:
        summarised_log_file = None
    if options.exp_types is None:
        exp_types = None
    else:
        exp_types = [ v.strip() for v in options.exp_types.split(',') ]
    try:
        SignalLib.use_c_libraries(True, options.verbose)
    except:
        pass

    total_files = 0
    total_adhesions = 0
    total_errors = 0
    total_warnings = 0
    total_steps = 0
    for config_file_no, config_file in enumerate(config_files):
        t1 = datetime.datetime.now()
        config = read_config_file(config_file)
        if options.file_pattern is not None:
            config['file_pattern'] = options.file_pattern
        if options.file_range is not None:
            config['file_range'] = options.file_range
        if options.base_output_dir is not None:
            config['base_output_dir'] = options.base_output_dir
        if options.verbose is not None:
            config['verbose'] = options.verbose
        if options.denoising_param is not None:
            config['denoising_param'] = options.denoising_param
        if options.ind_thld is not None:
            config['indicator_relative_threshold'] = options.ind_thld
        if options.show_plots is not None:
            config['show_plots'] = options.show_plots
        if options.plot is not None:
            config['plot_force_curves'] = 'f' in options.plot
            config['plot_indicators'] = 'i' in options.plot
        if config['exp_id'] is None:
            if config_file[:7] == 'config/' or config_file[:7] == 'config\\':
                exp_id = config_file[7:]
            else:
                exp_id = config_file
        else:
            exp_id = config['exp_id']
        if config['file_pattern'] is None:
            config['file_pattern'] = config['input_dir'] + '/' + exp_id + '/' + config['file_ext']
        if config['output_dir'] is None:
            config['output_dir'] = exp_id
        config['output_dir'] = config['base_output_dir'] + '/' + config['output_dir']
        if options.config_filter is not None and not eval(options.config_filter):
            print '%s (%4.1f%%, exp. %d of %d):\tskipped (excluded by config filter)' % (exp_id,
             float(config_file_no) / len(config_files) * 100,
             config_file_no + 1,
             len(config_files))
            continue
        if exp_types is not None and 'experiment_type' in config and config['experiment_type'] is not None and config['experiment_type'] not in exp_types:
            print "%s (%4.1f%%, exp. %d of %d, '%s'):\tskipped (excluded by experiment type)" % (exp_id,
             float(config_file_no) / len(config_files) * 100,
             config_file_no + 1,
             len(config_files),
             config['file_pattern'])
            continue
        if not options.overwrite and os.path.isfile(config['output_dir'] + '/curves.txt') and os.path.isfile(config['output_dir'] + '/steps.txt'):
            print "%s (%4.1f%%, exp. %d of %d, '%s'):\tskipped (output files already exist)" % (exp_id,
             float(config_file_no) / len(config_files) * 100,
             config_file_no + 1,
             len(config_files),
             config['file_pattern'])
            continue
        print "%s (%4.1f%%, exp. %d of %d, '%s')" % (exp_id,
         float(config_file_no) / len(config_files) * 100,
         config_file_no + 1,
         len(config_files),
         config['file_pattern'])
        if config['simulate_curves']:
            files = [ 'simulated curve #' + str(i + 1) for i in range(config['simulate_curves']) ]
        else:
            if type(config['file_pattern']) == list:
                files = []
                for file_pattern in config['file_pattern']:
                    files += glob.glob(file_pattern)

            else:
                files = glob.glob(config['file_pattern'])
            files.sort()
        if config['file_range'] is None:
            start_file, stop_file = nat2py_index(config['file_range_start'], config['file_range_stop'], len(files))
            if config['file_range_step'] is None:
                config['file_range_step'] = 1 if start_file <= stop_file else -1
            files = files[start_file:stop_file:config['file_range_step']]
        else:
            files = get_slice(files, config['file_range'])
        if not len(files):
            if summarised_log_file is not None:
                summarised_log_file.write('%s:\t[ERROR]   no input files!\n' % exp_id)
            print '[ERROR]   no input files!'
            continue
        if config['indicator_threshold'] is None and config['indicator_relative_threshold'] is None:
            if summarised_log_file is not None:
                summarised_log_file.write('%s:\t[warning] indicator_threshold not specified, skipping step detection\n' % exp_id)
            print '[warning] indicator_threshold not specified, skipping step detection'
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
        if not os.path.exists(config['output_dir'] + '/plots') and (config['plot_force_curves'] or config['plot_indicators']):
            os.mkdir(config['output_dir'] + '/plots')
        file(config['output_dir'] + '/config.txt', 'w').write(pprint.pformat(config))
        curves_file = open(config['output_dir'] + '/curves.txt', 'w')
        steps_file = open(config['output_dir'] + '/steps.txt', 'w')
        curves_file.write('# steps, peak position, peak extension, peak force, indentation force, work, baseline interception, baseline slope, baseline curvature, baseline fit length, baseline avg RSS, contact point, left contact slope, right contact slope, trace fit param 1/2/3, trace fit RMS, retrace fit param 1/2/3, retrace fit RMS, noise sigma, denoising parameter 1, denoising parameter 2, indicator threshold, file name\n')
        steps_file.write('# left flank pos., step pos., right flank pos., extension, rupture force, height, width, max slope, left slope, right slope, left slope RMS, right slope RMS, step #, curve #\n')
        log_file = open(config['output_dir'] + '/log.txt', 'w')
        log_file.write(strftime('%d.%m.%Y %H:%M:%S') + '   Curvalyser v%s\n' % __version__)
        files_ctr = 0
        adhesions = 0
        errors = 0
        warnings = 0
        steps_ctr = 0
        for file_no, filename in enumerate(files):
            del_log()
            basename = os.path.basename(filename)
            if config['simulate_curves']:
                x, y, mcsteps = monte_carlo_force_curve(config['simulate_params'])
                x_trace = None
                y_trace = None
                meta = None
                status = 0
                y[0] = 1e-09
                if config['multiplier_x'] is not None:
                    x *= config['multiplier_x']
                if config['multiplier_y'] is not None:
                    y *= config['multiplier_y']
            else:
                x, y, x_trace, y_trace, meta, status = load_curve(filename, config)
            if status:
                print '  %s / %s (%5.1f%%, exp. %3d of %3d, file %3d of %3d): skipped' % (exp_id,
                 basename,
                 (float(config_file_no) + float(file_no + 1) / len(files)) / len(config_files) * 100,
                 config_file_no + 1,
                 len(config_files),
                 file_no + 1,
                 len(files))
            else:
                x, y, x_trace, y_trace = prepare_data(config, x, y, x_trace, y_trace)
                results = analyse(x, y, x_trace, y_trace, config, basename, meta, curves_file, steps_file, files_ctr)
                if results['status']:
                    print '  %s / %s (%5.1f%%, exp. %3d of %3d, file %3d of %3d): skipped' % (exp_id,
                     basename,
                     (float(config_file_no) + float(file_no + 1) / len(files)) / len(config_files) * 100,
                     config_file_no + 1,
                     len(config_files),
                     file_no + 1,
                     len(files))
                else:
                    files_ctr += 1
                    steps_ctr += results['len_steps']
                    if results['len_steps'] > 0:
                        adhesions += 1
                    print u'  %s / %s (%5.1f%%, exp. %3d of %3d, file %3d of %3d): %2d step(s), %d row(s), noise: %6.2f %s, denoising: %s, sigma: %5.2f, ind thld: %6.2f' % (exp_id,
                     basename,
                     (float(config_file_no) + float(file_no + 1) / len(files)) / len(config_files) * 100,
                     config_file_no + 1,
                     len(config_files),
                     file_no + 1,
                     len(files),
                     results['len_steps'],
                     results['rows'],
                     results['noise_sigma'],
                     config['unit_y'],
                     fmt_float(results['denoising_param'], '%6.2f'),
                     results['MSF_sigma'],
                     results['indicator_threshold'])
            log_entries = get_log()
            output_log(log_entries, log_file, True)
            warnings += len([ entry for entry, level in log_entries if level == 1 ])
            errors += len([ entry for entry, level in log_entries if level == 2 ])

        curves_file.close()
        steps_file.write('# %d step(s)\n' % steps_ctr)
        steps_file.close()
        execution_time = datetime.datetime.now() - t1
        log('%d curve(s), %d adhesion curve(s), %d error(s), %d warning(s), %d step(s), execution time = %s' % (len(files),
         adhesions,
         errors,
         warnings,
         steps_ctr,
         execution_time), None, log_file, True)
        if len(config_files) > 1:
            print
        log_file.close()
        if summarised_log_file is not None:
            summarised_log_file.write('%s:\t%d curve(s), %d error(s), %d warning(s), %d step(s), execution time = %s\n' % (exp_id,
             len(files),
             errors,
             warnings,
             steps_ctr,
             execution_time))
        total_files += len(files)
        total_adhesions += adhesions
        total_errors += errors
        total_warnings += warnings
        total_steps += steps_ctr

    if summarised_log_file is not None:
        summarised_log_file.close()
    if len(config_files) > 1:
        print 'total: %d curve(s), %d adhesion curve(s), %d error(s), %d warning(s), %d step(s), execution time = %s' % (total_files,
         total_adhesions,
         total_errors,
         total_warnings,
         total_steps,
         datetime.datetime.now() - t0)
    return