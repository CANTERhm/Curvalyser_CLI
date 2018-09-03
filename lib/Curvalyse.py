#!/usr/bin/env python
# -*- coding: utf-8 -*-

# runs generate_config_files.py and intercepts the base_output_dir setting
# from the general config file and command line to set the Curvalyser logfile

general_config_file = 'config-curvalyser'
python_cmd = 'python'

import os
import re
import shutil
import sys

from CurvalyserLib import read_config_file

c = read_config_file(general_config_file)
base_output_dir = c['base_output_dir']
for i, arg in enumerate(sys.argv[1:]):
    m = re.match('(-o|--output_dir=?)(.*)', arg)
    if m:
        if m.end(2) > m.start(2):
            base_output_dir = arg[m.start(2):m.end(2)]
        elif i < len(sys.argv) - 2:
            base_output_dir = sys.argv[i + 2]
scripts_dir = os.path.dirname(__file__)
if not os.path.exists(base_output_dir): os.makedirs(base_output_dir)
os.system(python_cmd + ' generate_config_files.py')
if os.path.exists(general_config_file) and os.path.abspath('.') != os.path.abspath(base_output_dir): shutil.copy2(
    general_config_file, base_output_dir)
print "running Curvalyser with base output directory '%s'" % base_output_dir
os.system(python_cmd + ' %s/Curvalyser.py -l"%s/log.txt" %s' % (
scripts_dir, base_output_dir, ' '.join(['"' + v + '"' for v in sys.argv[1:]])))
