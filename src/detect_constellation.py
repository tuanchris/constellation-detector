import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from photutils import aperture_photometry, CircularAperture
from photutils import DAOStarFinder
from astropy.stats import mad_std

class StarImage():
    def __init__(self, img_url):
        '''
        A class to find n_brightest stars in an image.
        Example useage:
            dir = './data/sample.jpg'
            stars = StarImage(dir)
        '''
        self.img_url = img_url
        self._load_image()
        self.mean, self.median, self.std = sigma_clipped_stats(self.img)
        self._preprocess_image()



    def _load_image(self):
        '''
        Helper function to load and preprocess image
        '''
        # Load img
        img = cv2.imread(self.img_url)

        self.img = img
        print(f'Loaded image of size {img.shape}')

    def _preprocess_image(self):

        # substract rough esitmation of the background
        processed_img = np.float32(self.img) - self.median
        # Cvt to gray scale
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        self.processed_img = processed_img


    def find_stars(self, max_dim_per = 0.005, times_std = 5, n_brightest = 20):
        max_dim = max(self.img.shape)
        daofind = DAOStarFinder(
            fwhm= max_dim * max_dim_per,
            threshold = times_std * self.std,
            brightest=n_brightest)
        sources = daofind(self.processed_img)
        self.sources = sources.to_pandas()

path = '../data/sample.png'
stars = StarImage(path)
stars.find_stars(n_brightest=50)
stars.sources
