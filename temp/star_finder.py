import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from photutils import aperture_photometry, CircularAperture
from photutils import DAOStarFinder
from astropy.stats import mad_std

class StarFinder:
    '''
    A class to find n_brightest stars in an image.
    Example useage:
        dir = './data/sample.jpg'
        stars = StarFinder(dir)
        stars.find()
        stars.plot(30)
    '''
    def __init__(self, img_url):
        self.img_url = img_url

    def _load_image(self):
        '''
        Helper function to load and preprocess image
        '''
        # Load img
        img = cv2.imread(self.img_url)
        # Cvt to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # substract rough esitmation of the background
        img = np.float32(img) - np.median(img)

        self.img = img


    def find(self, fwhm=.4, threshold=3.):
        '''
        Search for stars in an image using DAOFIND algorithm.

        Parameters:
            - threshold (float): The absolute image value above which to select
                sources.
            - fwhmfloat (float): The full-width half-maximum (FWHM) of the major
                axis of the Gaussian kernel in units of pixels.
        '''
        self._load_image()
        img = self.img
        bg_sigma = mad_std(img)
        daofind = DAOStarFinder(fwhm=.4, threshold = threshold*bg_sigma)
        sources = daofind(img).to_pandas()
        self.sources = sources
        return sources

    def plot(self, top_n=0, figsize=(20,10), cmap='gray_r'):
        '''
        Plot the found stars in the image. If top_n is set to a value > 0,
        plot only top_n brightest stars.

        Parameters:
            - top_n (int): number of brightest stars to plot
            - figsize (tuple): size of the image
            - cmap (str): color map to plotted, default is inverted gray
        '''
        if top_n > 0:
            sources = self.top_n(top_n)
        else:
            sources = self.sources

        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=4.)

        plt.figure(figsize=figsize)
        plt.imshow(self.img, cmap=cmap)
        for i in sources.index:
            label = sources.peak[i]
            plt.annotate(label, (sources.xcentroid[i], sources.ycentroid[i]))

        apertures.plot(color='b', lw=1.5, alpha=0.5)


    def top_n(self, n):
        '''
        Return top_n brightest stars as a Pandas Data Frame

        Parameters:
            - top_n: top_n brightest stars to return
        '''
        top_n = self.sources[self.sources.peak.isin(self.sources.peak.nlargest(n))]
        return top_n

dir = './data/sample.jpg'
stars = StarFinder(dir)
stars.find()
stars.plot(20)

half_max = (np.max(stars.img) - np.min(stars.img))/2
nearest = np.abs(stars.img - half_max).argmin()
fwhm = (stars.img[nearest] -  np.min(stars.img))*2
fwhm
