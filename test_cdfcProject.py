import pytest
import numpy as np
from cdfcProject import normalize, discretization, __mapInstanceToClass, __dealToBuckets, fillBuckets, __getPermutation
from sklearn.preprocessing import StandardScaler
from collections import namedtuple


def test_discretization():
    # * This test checks that the discrete function we are using has the expected outcomes
    # the values should become       -1   -1   1   1    0     0   0  1  -1
    assert discretization(np.array((-10, -5.7, 2, 9.5, 0.5, -0.5, 0, 1, -1))) == np.array((-1, -1, 1, 1, 0, 0, 0, 1, -1))
    

# * These tests will check that the permutation that happens in __getPermutation() doesn't change based on the
# * data type of the array of arrays. The __getPermutation function takes an array of instances, which will all
# * be in the same class, and reorders them (randomly in production, but using seed 498 during testing). This
# * random ordering will then be looped over sequentially in order to fill (or "deal") to our buckets.
# * Essentially this is the function that introduces randomness into our K-Fold Validation.
# *
# * Notes:
# * > we are using seed 498. This will create a default_rng that when given a list of 3 items, it will move the
# *   1st item to the last position. So:
# *   [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  ---becomes-->  [[4, 5, 6], [7, 8, 9], [1, 2, 3]]
# *   [[a, b, c], [d, e, f], [g, h, i]]  ---becomes-->  [[d, e, f], [g, h, i], [a, b, c]]

def test__getPermutationInt():
    assert __getPermutation([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 498) == [[4, 5, 6], [7, 8, 9], [1, 2, 3]]


def test__getPermutationFloat():
    assert __getPermutation([[1.5, 2.6, 3.1], [4.001, 5.778, 6.345], [7.000, 8.123, 9.912]], 498) == [[4.001, 5.778, 6.345], [7.000, 8.123, 9.912], [1.5, 2.6, 3.1]]


def test__getPermutationNegInt():
    assert __getPermutation([[-1, -2, 3], [-4, -5, -6], [-7, -8, -9]], 498) == [[-4, -5, -6], [-7, -8, -9], [-1, -2, -3]]
    

def test__getPermutationNegFloat():
    assert __getPermutation([[-1.5, -2.6, -3.1], [-4.001, -5.778, -6.345], [-7.000, -8.123, -9.912]], 498) == [[-4.001, -5.778, -6.345], [-7.000, -8.123, -9.912], [-1.5, -2.6, -3.1]]
    
    
def test__getPermutationMixedPosNeg():
    assert __getPermutation([[1, 2, 3], [-4, 5, -6], [-7, -8, -9]], 498) == [[-4, 5, -6], [-7, -8, -9], [1, 2, 3]]
    
    
def test__getPermutationMixedAll():
    assert __getPermutation([[-1.5, 2.6, -3.1], [4.001, 5, 6], [-7.000, -8, -9.912]], 498) == [[4.001, 5, 6], [-7.000, -8, -9.912], [-1.5, 2.6, -3.1]]


def test_normalize(data):
    assert False


def test__mapInstanceToClass():
    assert False


def test__dealToBuckets():
    assert False


def test_fill_buckets():
    assert False
