"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random

from mpmath.libmp.backend import xrange

import assignment1
from assignment1 import Assignment1

def LU_partial_decomposition(matrix):
    n, m = matrix.shape
    P = np.identity(n)
    L = np.identity(n)
    U = matrix.copy()
    PF = np.identity(n)
    LF = np.zeros((n,n))
    for k in range(0, n - 1):
        index = np.argmax(abs(U[k:,k]))
        index = index + k
        if index != k:
            P = np.identity(n)
            P[[index,k],k:n] = P[[k,index],k:n]
            U[[index,k],k:n] = U[[k,index],k:n]
            PF = np.dot(P,PF)
            LF = np.dot(P,LF)
        L = np.identity(n)
        for j in range(k+1,n):
            L[j,k] = -(U[j,k] / U[k,k])
            LF[j,k] = (U[j,k] / U[k,k])
        U = np.dot(L,U)
    np.fill_diagonal(LF, 1)
    return PF, LF, U

def solve_low(l, b, d):
    y = np.array([b[0]])
    i = 1
    while i < (d+1):
        l_to_mult = l[i][:i]
        mult_array = np.multiply(l_to_mult , y)
        sum = np.sum(mult_array)
        y = np.append(y, b[i] - sum)
        i = i+1
    return y



def solve_up(u, y, d):
    elem = (y[d]) / (u[d][d])
    x = np.array([elem])
    i = d-1
    while i >=0:
        u_to_mult = u[i][i+1:]
        mult_array = np.multiply(u_to_mult , x)
        sum = np.sum(mult_array)
        yi = y[i]
        udd = u[i][i]
        almost_to_add = yi - sum
        to_add = almost_to_add / udd
        x = np.append((y[i] - sum)/(u[i][i]) , x)
        i = i-1
    return x


def lu_solve(left_part_matrix, right_part_matrix, d):
    p, l, u = LU_partial_decomposition(left_part_matrix)
    b = np.matmul(p, right_part_matrix)
    y = solve_low(l, b, d)
    x = solve_up(u, y, d)
    return x

class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass



    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        t0 = time.time()
        if d < 3:
            n = 6000
        else:
            n = 2000
        x_dots=np.linspace(a,b,n)
        y_dots=[]

        i = 0
        while i < len(x_dots):
            y_dots.append(f(x_dots[i]))
            if time.time() - t0 > maxtime:
                return None
            i = i+1
        left_part_matrix = np.array([])
        right_part_matrix = np.array([])

        for i in range(d+1):
            for j in range(d+1):
                element = np.sum(np.power(x_dots, (i+j)))
                left_part_matrix = np.append(left_part_matrix, element)
            right_part_element = sum(y_dots * np.power(x_dots, i))
            right_part_matrix = np.append(right_part_matrix, right_part_element)

        left_part_matrix = np.reshape(left_part_matrix, ((d + 1), (d + 1)))
        right_part_matrix = np.reshape(right_part_matrix, ((d+1) , 1))

        coefficients = lu_solve(left_part_matrix,right_part_matrix, d)

        def f(x):
            i = 0
            y = 0
            while i <= d:
                y = y + coefficients[i]*np.power(x, i)
                i = i+1
            return y


        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # replace these lines with your solution
        # result = lambda x: x
        # y = f(1)

        return f


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):            
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

        
        



if __name__ == "__main__":
    unittest.main()
