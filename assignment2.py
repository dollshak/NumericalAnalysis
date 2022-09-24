"""
In this assignment you should find the intersection points for two functions.
"""
import numpy
import numpy as np
import time
import random
from collections.abc import Iterable


def find_der(f, x):
    h=1e-10
    return (f(x+h)-f(x))/h


def findRootWithNewton(g, initialGuess, maxerr):
    x_i = initialGuess
    for j in range(1000):
        try:
            if np.abs(g(x_i)) <= maxerr:
                return round(x_i, 4)
            der_g = find_der(g, x_i)
            if not der_g == 0:
                x_i = x_i - g(x_i) / der_g

        except:
            break
    return None


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
            
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        def g(x):
            return np.abs(f1(x) - f2(x))

        roots = []
        i = a
        while i <= b:
            j = i + (b - a) / 100
            potential_root = findRootWithNewton(g, i, maxerr / 2)
            if potential_root is not None:
                if (potential_root not in roots) & (potential_root >= a) & (potential_root <= b):
                    roots.append(potential_root)
            i = j
        return roots



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):
    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
