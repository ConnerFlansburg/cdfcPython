import numpy as np

rand = np.random.default_rng(498)  # create a numpy random generator with/without seed
instances = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(instances)
permutation = rand.permutation(instances)  # shuffle the instances
print(permutation)
