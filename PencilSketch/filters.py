#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A module containing a number of interesting image filter effects,
    such as:
    * Black-and-white pencil sketch
    * Warming/cooling filters
    * Cartoonizer
"""

import numpy as np
import cv2

from scipy.interpolate import UnivariateSpline

__author__ = "Michael Beyeler"
__license__ = "GNU GPL 3.0 or later"


class PencilSketch:
    """Pencil sketch effect
        A class that applies a pencil sketch effect to an image.
        The processed image is overlayed over a background image for visual
        effect.
    """

    def __init__(self, width, height, bg_gray='pencilsketch_bg.jpg'):
        """Initialize parameters
            :param (width, height): Image size.
            :param bg_gray: Optional background image to improve the illusion
                            that the pencil sketch was drawn on a canvas.
        """
        self.width = width
        self.height = height

        # try to open background canvas (if it exists)
        self.canvas = cv2.imread(bg_gray, cv2.CV_8UC1)
        if self.canvas is not None:
            self.canvas = cv2.resize(self.canvas, (self.width, self.height))

    def render(self, img_rgb):
        """Applies pencil sketch effect to an RGB image
            :param img_rgb: RGB image to be processed
            :returns: Processed RGB image
        """
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
        img_blend = cv2.divide(img_gray, img_blur, scale=256)

        # if available, blend with background canvas
        if self.canvas is not None:
            img_blend = cv2.multiply(img_blend, self.canvas, scale=1./256)

        return cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)