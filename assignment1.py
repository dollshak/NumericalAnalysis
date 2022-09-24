"""
In this assignment you should interpolate the given function.
"""
import math

import numpy
import numpy as np
import time
import random
import sympy as sp


def get_line(x_1, y_1, x_2, y_2):
    def f(x):
        m = (y_2-y_1)/(x_2-x_1)
        b = y_1 - m*x_1
        y = m * x + b
        return y
    return f


def get_length(x_1, y_1, x_2, y_2):
    delta_x = abs(x_2-x_1)
    delta_y = abs(y_2-y_1)
    length = pow(pow(delta_x, 2)+pow(delta_y, 2), 0.5)  # pitagoras
    return length


def find_t_help(x_1, y_1, x_2, y_2, x):
    f = get_line(x_1, y_1, x_2, y_2) # create a line between the two dots and
    y = f(x)  # find the y value of x on the line
    part = get_length(x_1, y_1, x, y)
    full_length = get_length(x_1, y_1, x_2, y_2) # get the length of the line
    t = part/full_length
    return t


def find_t(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x):
    t = 0
    if (x >= x_1) & (x < x_2):
        t = find_t_help(x_1, y_1, x_2, y_2, x)
    elif (x >= x_2) & (x < x_3):
        t = find_t_help(x_2, y_2, x_3, y_3, x)
    elif (x >= x_3) & (x < x_4):
        t = find_t_help(x_3, y_3, x_4, y_4, x)
    else:
        print("x is not in this domain")
    return t


def biz(y_1, y_2, y_3, y_4, t):
    y = y_4*pow(t, 3)+3*y_3*pow(t, 2)*(1-t)+3*y_2*t*pow(1-t, 2)+y_1*pow(1-t, 3)
    return y


def get_d_s(dots):
    dots_len = len(dots)
    num_of_control_points = dots_len - 1
    d_s_dots = [[dots[0][0] + 2 * dots[1][0], dots[0][1] + 2 * dots[1][1]]]
    i = 1
    while i <= num_of_control_points - 2:
        d_s_dots.append([2 * (2 * dots[i][0] + dots[i + 1][0]), 2 * (2 * dots[i][1] + dots[i + 1][1])])
        i = i + 1

    d_s_dots.append(
        [8 * dots[dots_len - 2][0] + dots[dots_len - 1][0], 8 * dots[dots_len - 2][1] + dots[dots_len - 1][1]])
    return d_s_dots


def get_new_c_s(c_s, b_s, a_s, num_of_control_points):
    j=1
    new_c_s = [((c_s[0])/(b_s[0]))]
    while j < num_of_control_points:
        new_c_s.append(c_s[j] / (b_s[j] - new_c_s[j - 1] * a_s[j]))
        j = j + 1
    return new_c_s


def get_new_d_s(d_s_dots, new_c_s, c_s , b_s, a_s, num_of_control_points):
    j = 1
    new_d_s_points = [([d_s_dots[0][0]/b_s[0] , d_s_dots[0][1]/b_s[0]])]
    while j < num_of_control_points:
        d_s_dots[j][0]
        new_d_s_points[j - 1][0]
        a_s[j]
        b_s[j]
        new_c_s[j - 1] * a_s[j]
        new_d_s_points.append([(d_s_dots[j][0]-new_d_s_points[j-1][0]*a_s[j]) / (b_s[j]-new_c_s[j-1]*a_s[j]) , (d_s_dots[j][1]-new_d_s_points[j-1][1]*a_s[j]) / (b_s[j]-new_c_s[j-1]*a_s[j])])
        j=j+1
    return new_d_s_points


def calculate_a_s(dots):  # using thomas algorithm
    dots_len = len(dots)
    num_of_control_points = dots_len - 1
    # setup the arrays a_2 b_s c_s
    c_s = []
    b_s = []
    a_s = [0]
    for i in range(0, num_of_control_points - 1):
        c_s.append(1)
        b_s.append(4)
        a_s.append(1)
    c_s.append(0)
    b_s.append(7)
    b_s[0] = 2
    a_s[len(a_s)-1] = 2

    # setup the array d_s
    d_s_dots = get_d_s(dots)
    # the algorithm
    # first phase - update c_s and d_s
    new_c_s = get_new_c_s (c_s, b_s, a_s, num_of_control_points)
    new_d_s_points = get_new_d_s(d_s_dots, new_c_s, c_s, b_s, a_s, num_of_control_points)

    # second phase - find x_s
    x_s = np.ones((num_of_control_points,2))
    k = num_of_control_points - 1
    x_s[k][0] = new_d_s_points[k][0]
    x_s[k][1] = new_d_s_points[k][1]
    k=k-1

    while k >= 0:
        x_s[k][0] = new_d_s_points[k][0] - new_c_s[k] * x_s[k+1][0]
        x_s[k][1] = new_d_s_points[k][1] - new_c_s[k] * x_s[k+1][1]
        k=k-1
    return x_s


def calculate_b_s(a_s_control_points, dots):
    num_of_c_points = len(a_s_control_points)
    b_s = np.ones((num_of_c_points, 2))
    i = 0
    while i < num_of_c_points - 1:
        b_s[i][0] = 2*dots[i+1][0] - a_s_control_points[i+1][0]
        b_s[i][1] = 2*dots[i+1][1] - a_s_control_points[i+1][1]
        i=i+1
    b_s[num_of_c_points-1][0] = (a_s_control_points[num_of_c_points-1][0]+dots[len(dots)-1][0])/2
    b_s[num_of_c_points-1][1] = (a_s_control_points[num_of_c_points-1][1]+dots[len(dots)-1][1])/2
    return b_s


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        dots = []
        x = np.linspace(a, b, n)
        for i in range(n):
            dots.append([x[i], f(x[i])])
        a_s_control_points = calculate_a_s(dots)
        b_s_control_points = calculate_b_s(a_s_control_points, dots)

        def g(x):
            i = 0
            in_domain = False
            while i < len(dots) - 1:
                if (x >= dots[i][0]) & (x <= dots[i + 1][0]):
                    in_domain = True
                    # x_1 = dots[i][0]
                    # x_2 = a_s_control_points[i][0]
                    # x_3 = b_s_control_points[i][0]
                    # x_4 = dots[i + 1][0]
                    y_1 = dots[i][1]
                    y_2 = a_s_control_points[i][1]
                    y_3 = b_s_control_points[i][1]
                    y_4 = dots[i+1][1]
                    t = (x-dots[i][0])/(dots[i+1][0]-dots[i][0])
                    y = biz(y_1, y_2, y_3, y_4, t)
                    return y
                if x == b:
                    return f(x)
                i = i + 1
            if not in_domain:
                print("x is not in the domain")
        return g

    def p(self, x ):
        return -1

    def c_k (x, y):
        return -1


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
