#!/Library/Frameworks/Python.framework/Versions/2.6/bin/python
# -*- coding: utf-8 -*-

import glob
import os
import os.path
import numpy as np

input_file_pattern = 'C:\Users\Steffie\Desktop\Curvalyser\ForceCurves/*'
ignore_ids = []
experiment_types = { # id -> experiment type


'TEST' : 'TEST',



}

files = glob.glob(input_file_pattern)
files.sort()
if not os.path.exists('config/'): os.mkdir('config/')
counter = 0
for file in files:
  id = os.path.basename(file)
  if id in ignore_ids:
    print id + '\t(ignored)'
    continue
  if id not in experiment_types:
    print id + '\t(skipped)'
    continue
  print '%s:\t%s' % (id, experiment_types[id])
  counter += 1
  config_file = open('config/' + id, 'w')
  config_file.write("""\
execfile('config-curvalyser')
experiment_type = '""" + experiment_types[id] + """'
""")
  config_file.close()
print '\n', counter, 'config file(s) created:'
