#!/usr/bin/env python

input_filename = '/Daten/beta1/2011-08-05/CBF_test_01_01sec/force-save-2011.08.05-14.52.01.out'
output_dir = '/Daten/Auswertung/CMA-Plots-Denoised/'
denoising_param = 2.0

import os

import numpy as np
import pylab as pl

from CurvalyserLib import load_curve, output_log
from SignalLib import ReNoiR, noise_profiles

x, y, x_trace, y_trace, meta, status = load_curve(input_filename, {'file_format': 'jpk-old'})
if status: output_log()
y_denoised = ReNoiR(y, threshold=denoising_param, thld_calibr_params=noise_profiles['AFM'])
if output_dir is not None and output_dir != '': np.savetxt(output_dir + '/' + os.path.basename(input_filename) + '.txt',
                                                           np.transpose(np.array([x, y])), fmt='%s', delimiter='\t')
pl.plot(x, y)
pl.plot(x, y_denoised)
pl.show()
