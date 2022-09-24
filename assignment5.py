"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random

from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

from functionUtils import AbstractShape
from sklearn.cluster import MiniBatchKMeans,DBSCAN
import matplotlib.pyplot as plt

class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        dots = contour(10000)
        area = 0
        i = 0
        while i < 10000-1:
            curr_area = (dots[i][1]+dots[i+1][1])*(dots[i+1][0]-dots[i][0])
            area = area + curr_area
            i = i+1
        area = area / 2
        return np.float32(np.abs(area))

    def sample_dots(self, sample, n):
        sampled_xs = []
        sampled_ys = []
        for i in range(n):
            x, y = sample()
            sampled_xs.append(x)
            sampled_ys.append(y)
        sampled_xs = np.array(sampled_xs)
        sampled_ys = np.array(sampled_ys)
        return sampled_xs, sampled_ys
    
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        sampled_xs, sampled_ys = self.sample_dots(sample, 10000)
        x_mean = (np.sum(sampled_xs))/10000
        y_mean = (np.sum(sampled_ys))/10000
        angles = np.arctan2(sampled_xs - x_mean , sampled_ys - y_mean)
        sorted_indexes = np.argsort(angles)
        sorted_xs = sampled_xs[sorted_indexes]
        fitted_xs = savgol_filter(sorted_xs, 10000-1001, 13)
        sorted_ys = sampled_ys[sorted_indexes]
        fitted_ys = savgol_filter(sorted_ys , 10000-1001, 13)
        dots = []
        for i in range(len(fitted_xs)):
            dots.append([fitted_xs[i] , fitted_ys[i]])

        class MyShape(AbstractShape):
            def __init__(self):
                pass

            def sample(self):
                return sample

            def area(self) -> np.float32:
                area = 0
                i = 0
                while i < 10000 - 1:
                    curr_area = (dots[i][1] + dots[i + 1][1]) * (dots[i + 1][0] - dots[i][0])
                    area = area + curr_area
                    i = i + 1
                area = area / 2
                return np.float32(np.abs(area))

        return MyShape()


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #
    #     def sample():
    #         time.sleep(7)
    #         return circ()
    #
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=sample, maxtime=5)
    #     T = time.time() - T
    #     self.assertTrue(isinstance(shape, AbstractShape))
    #     self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
