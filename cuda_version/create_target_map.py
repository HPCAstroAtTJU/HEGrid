# --------------------------------------------------------------------
#
# title                  :target_map.py
# description            :Produce random data for test
# author                 :
#
# --------------------------------------------------------------------

# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

from astropy.io import fits
# from astropy.coordinates import SkyCoord
from astropy import wcs
import numpy as np
import sys, getopt
import os
import math

# get input / target files path from arguments
path = ""
tfile = ""
num = ""
beam_size = 0
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hp:t:b:',
                               ['path=', 'target=', 'beam='])
except getopt.GetoptError:
    print(
        'python target_map.py -p <absolute path> -t <targetfile> -b <beam size>')
    print(
        'For example\n\tpython target_map.py -p /home/summit/Project/HPDC_2022/ target -b 180')
    sys.exit(2)
for opt, arg in opts:
    if opt in ["-h", "--help"]:
        print(
            "usage: python target_map.py -p <absolute path> -i <inputfile> -t <targetfile> -b <beam size>")
        print(
            'For example\n\tpython target_map.py -p /home/summit/Project/HPDC_2022/ -t target -b 180')
        sys.exit()
    elif opt in ["-p", "--path"]:
        path = arg
    elif opt in ["-t", "--target"]:
        tfile = arg
    elif opt in ["-b", "--beamarg"]:
        beam_size = int(arg)
        print(beam_size)

print("target file is ", path + tfile + '.fits')
targetfile = os.path.join(wcs.__path__[0], path + tfile + '.fits')


# Produce a FITS header that contains the target field.
def setup_header(mapcenter, mapsize, pixsize, beamsize_fwhm):
    # define target grid (via fits header according to WCS convention)
    dnaxis1 = int(mapsize[0] / pixsize)
    dnaxis2 = int(mapsize[1] / pixsize)
    print(pixsize, dnaxis1, dnaxis2)
    print("map_center:",(mapcenter[0],mapcenter[1]))
    print("map_size:",(mapsize[0],mapsize[1]))

    header = {
        'NAXIS': 3,
        'NAXIS1': dnaxis1,
        'NAXIS2': dnaxis2,
        'NAXIS3': 1,  # need dummy spectral axis
        'CTYPE1': 'RA---SIN',
        'CTYPE2': 'DEC--SIN',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CDELT1': -pixsize,
        'CDELT2': pixsize,
        'CRPIX1': dnaxis1 / 2.,
        'CRPIX2': dnaxis2 / 2.,
        'CRVAL1': mapcenter[0],
        'CRVAL2': mapcenter[1],
    }
    return header


# initialize target data
def setup_target_data(my_header):
    yzx_shape = (my_header['NAXIS3'], my_header['NAXIS2'], my_header['NAXIS1'])
    datacube = np.zeros(yzx_shape, dtype=np.float32)
    return datacube

# Parameter setting
mapcenter = 70.04, 36.29  # all in degrees (ra,dec)
# map_size = 5.
mapsize = 55, 10  # (ra,dec)
beamsize_fwhm = 2 * beam_size / 3600.
# a good pixel size is a third of the FWHM of the PSF (avoids aliasing)
pixsize = beamsize_fwhm / 3.
# pixelsize = 0.01

# get target header
my_header = setup_header(mapcenter, mapsize, pixsize, beamsize_fwhm)
my_wcs = wcs.WCS(my_header, naxis=[wcs.WCSSUB_CELESTIAL])
my_wcs = wcs.WCS(my_header)
target_header = my_wcs.to_header()

# get target data
datacube = setup_target_data(my_header)

# write target fits file
fits.writeto(targetfile, data=datacube, header=target_header, overwrite=True, checksum=True)
