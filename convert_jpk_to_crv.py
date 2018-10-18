#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import glob
import numpy as np
from CurvalyserLib import load_curve, export_crv_file

if len(sys.argv) < 3:
  print 'syntax: ' + sys.argv[0] + ' output_dir input_files'
  exit(1)
output_dir = sys.argv[1]
input_files = []
for arg in sys.argv[2:]: input_files += glob.glob(arg)
if not os.path.exists(output_dir) and len(input_files): os.makedirs(output_dir)
for input_file in input_files:
  x, y, x_trace, y_trace, meta, status = load_curve(input_file, 'jpk')
  if status: print 'file:', input_file, 'error:', status
  else:
    print 'converting', input_file
    basename = os.path.basename(input_file)
    if basename[-4:] == '.out': basename = basename[:-4]
    export_crv_file(output_dir + '/' + basename + '.crv', y, (x[-1] - x[0]) / float(len(x)), y_trace, (x_trace[-1] - x_trace[0]) / float(len(x)), meta)
