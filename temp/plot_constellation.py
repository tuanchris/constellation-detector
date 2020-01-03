import photutils
import numpy as np
from photutils import datasets
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from photutils import aperture_photometry, CircularAperture
%matplotlib inline

hdu = datasets.load_star_image()
image = hdu.data[500:700, 500:700].astype(float)
image -= np.median(image)

from photutils import DAOStarFinder
from astropy.stats import mad_std

bkg_sigma = mad_std(image)
daofind = DAOStarFinder(fwhm=.4, threshold = 3.*bkg_sigma)
sources = daofind(image)

positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
phot_table = aperture_photometry(image, apertures)


plt.figure(figsize=(40,40))
plt.imshow(img, cmap='gray_r')
apertures.plot(color='b', lw=1.5, alpha=0.5)


img = cv2.imread('./data/pixel-4-astrophotography-2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.float32(img) - np.median(img)
bkg_sigma = mad_std(img)
daofind = DAOStarFinder(fwhm=.4, threshold = 3.*bkg_sigma)
sources = daofind(img)
sources = sources.to_pandas()
top_n = sources[sources.peak.isin(sources.peak.nlargest(50))]
# top_n = sources

positions = np.transpose((top_n['xcentroid'], top_n['ycentroid']))
apertures = CircularAperture(positions, r=4.)

plt.figure(figsize=(40,40))
plt.imshow(img, cmap='gray_r')
apertures.plot(color='b', lw=1.5, alpha=0.5)
