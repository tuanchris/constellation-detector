import cv2
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch
from photutils import DAOStarFinder, CircularAperture
from photutils import aperture_photometry
import pandas as pd
import numpy as np

img = cv2.imread('./data/sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')

mean, median, std = sigma_clipped_stats(img)
print((mean, median, std))

daofind = DAOStarFinder(fwhm=max(img.shape)*.01, threshold = 5. * std, brightest=20)
sources = daofind(img-median)

df = sources.to_pandas()
df_plot = df
# df_plot = df[df.mag.isin(df.mag.nsmallest(20))]

positions = np.transpose((df_plot.xcentroid, df_plot.ycentroid))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.figure(figsize=(20,20))
plt.imshow(img, cmap='gray_r', origin = 'lower',norm=norm)
apertures.plot(color='blue', lw=1.5, alpha=.5)

phot_table = aperture_photometry(img, apertures, method='subpixel',subpixels=5).to_pandas()
df_plot = phot_table[phot_table.aperture_sum.isin(phot_table.aperture_sum.nlargest(20))]

positions = np.transpose((df_plot.xcenter, df_plot.ycenter))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.figure(figsize=(20,20))
plt.imshow(img, cmap='gray_r', origin = 'lower',norm=norm)
apertures.plot(color='blue', lw=1.5, alpha=.5)
