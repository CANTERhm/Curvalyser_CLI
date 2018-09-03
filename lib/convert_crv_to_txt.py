#!/usr/bin/env python
# -*- coding: utf-8 -*-

save_curves = 1
save_steps = 1

import glob
import os
import sys

import numpy as np

from CurvalyserLib import load_curve

if len(sys.argv) < 3:
    print 'syntax: ' + sys.argv[0] + ' output_dir input_files'
    exit(1)
output_dir = sys.argv[1]
input_files = []
for arg in sys.argv[2:]: input_files += glob.glob(arg)
if len(input_files):
    if save_curves and not os.path.exists(output_dir):            os.makedirs(output_dir)
    if save_steps and not os.path.exists(output_dir + '/steps'): os.makedirs(output_dir + '/steps')
for input_file in input_files:
    x, y, x_trace, y_trace, meta, status = load_curve(input_file, load=['y', 'x', 'meta'])
    if status:
        print 'file:', input_file, 'error:', status
    else:
        print 'converting', input_file
        basename = os.path.basename(input_file)
        if basename[-4:] == '.crv': basename = basename[:-4]
        if save_curves:
            if x is None:
                np.savetxt(output_dir + '/' + basename + '.txt', y, fmt='%s', delimiter='\t')
            else:
                np.savetxt(output_dir + '/' + basename + '.txt', np.transpose(np.array([x, y])), fmt='%s',
                           delimiter='\t')
        if save_steps: np.savetxt(output_dir + '/steps/' + basename + '.txt', meta['steps'], fmt=('%d', '%s'),
                                  delimiter='\t')
