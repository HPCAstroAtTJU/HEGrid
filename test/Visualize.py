# --------------------------------------------------------------------
#
# title                  :Visualize.py
# description            :Visualize output map
# author                 :
#
# --------------------------------------------------------------------

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from astropy.io import fits
from astropy import wcs
import numpy as np
import os
import sys, getopt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from astropy.visualization import astropy_mpl_style

# get input / output fits file from arguments
path = ""
ofile = ""
num = ""
try:
	opts, args = getopt.getopt(sys.argv[1:], 'hp:o:n:', ['path=', 'output=', 'num='])
except getopt.GetoptError:
	print('python Visualize.py -p <absolute path> -i <inputfile> -o <outputfile> -n <number>')
	sys.exit()
for opt, arg in opts:
	if opt in ["-h", "--help"]:
		print("usage: python Visualize.py -p <absolute path> -i <inputfile> -o <outputfile> -n <number>'")
		sys.exit()
	elif opt in ["-p", "--path"]:
		path = arg
	elif opt in ["-o", "--output"]:
		ofile = arg
	elif opt in ["-n", "--num"]:
		num = arg
print("output file is ", path+ofile+num+'.fits')
outfile = os.path.join(wcs.__path__[0], path+ofile+num+'.fits')

# get output data
hdulist = fits.open(outfile)
out_header = hdulist[0].header
out_data = hdulist[0].data
out_data = np.nan_to_num(out_data)
out_data = np.reshape(out_data, (out_header['NAXIS1'], out_header['NAXIS2']))
out_min = np.min(out_data)
out_max = np.max(out_data)
out_cabs = np.max(np.abs(out_data))
out_wcs = wcs.WCS(header=out_header, naxis=[wcs.WCSSUB_CELESTIAL])
hdulist.close()

# show output image
plt.style.use(astropy_mpl_style)
fig = plt.figure(figsize=(20, 20))
# ax = fig.add_subplot(111, projection=out_wcs.celestial)
ax = fig.add_subplot(projection=out_wcs.celestial)
# im = ax.imshow(out_data, cmap='viridis', vmin=out_min, vmax=out_max)
im = ax.imshow(out_data, norm=colors.LogNorm(vmin=1e-5, vmax=out_max), cmap='viridis')

lon, lat = ax.coords
lon.set_major_formatter('dd:mm')
lat.set_major_formatter('dd:mm')
lon.set_axislabel('R.A. [deg]')
lat.set_axislabel('Dec [deg]')

ax = plt.gca()
# ax.invert_xaxis()
cbarr = plt.colorbar(im)
plt.show()
fig.savefig(path+'output'+num+'.png')

