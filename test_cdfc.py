import random as rand
from unittest import TestCase
from cdfc import Hypothesis, ConstructedFeature


# *********************** Setup ************************ #
rand.seed(498)  # seed the random library

# ********************** Objects *********************** #


# Tree object tests
class TestTree(TestCase):
    # Run Tree Test
    def test_run_tree(self):
        self.fail()


# Constructed Feature object tests
class TestConstructedFeature(TestCase):
    # Get Used Features test
    def test_get_used_features(self):
        self.fail()
    
    # Transform Test
    def test_transform(self):
        self.fail()


# Hypothesis object tests
class TestHypothesis(TestCase):
    # Get Fitness test
    def test_get_fitness(self):
        self.fail()
    
    # Transform test
    def test_transform(self):
        self.fail()
        
    def test_for_references(self):
        self.fail()


# ******************* End of Objects ********************* #

def test_crossover():  # requires 2 hypoths
    assert False


def test_mutate():  # requires 1 hypoth
    assert False


def test_evolve():  # requires 1 population, 1 hypothesis
    assert False


def test__tournament():  # requires 1 population
    assert False

    
def test__crossoverTournament():  # requires 1 population
    assert False


def test_create_hypothesis():
    assert False


def test_create_initial_population():
    assert False


def test_cdfc():
    assert False


def test_constructed_feature():
    assert False


def test_hypothesis():
    assert False


def test_population():
    assert False
