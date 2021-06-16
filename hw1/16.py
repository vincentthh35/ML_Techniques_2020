from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('Agg')
import math

def rbf_kernel(x1, x2, gamma):
    return np.exp(- gamma * np.square(x1 - x2).sum())

target_number = 0

# convert data
f = open('features.train', 'r')
lines = f.readlines()

# if y is target_number, y = 1, else -1
all_data = np.array([ [ int(line.split()[0][0]), *map(float, line.split()[1:]) ] for line in lines ])
train_y = np.where(all_data[:,0] == target_number, 1, -1)
train_x = all_data[:,1:]

# test_data = np.array([ [ int(line.split()[0][0]), *map(float, line.split()[1:]) ] for line in lines ])
# test_y = np.where(test_data[:,0] == target_number, 1, -1)
# test_x = test_data[:,1:]

# print(train_y, train_x)
# print(f'{len(train_y)}, {len(train_x)}')
# print(train_y, train_x)
C_list = [-3, -2, -1, 0, 1]
# log_10 C = 0.1 does better
C = -1
gamma_list = [-1, 0, 1, 2, 3]
gamma = 80
all_Eout = []
best_gamma_count = [0, 0, 0, 0, 0]
exp_time = 100
sample_n = 1000
N = len(train_x)
# all_w = []
# all_Ein = []
# all_distance = []
# n_support_vectors = []
for i in range(exp_time):
    sample_set = random.sample(range(N), sample_n)
    sample_set.sort()
    mask = np.full(N, True, dtype=bool)
    mask[sample_set] = False
    test_x = train_x[~mask]
    test_y = train_y[~mask]
    train_x_prime = train_x[mask]
    train_y_prime = train_y[mask]
    Ein_list = []
    for gamma in gamma_list:
        classifier = svm.SVC(kernel = 'rbf', gamma = 10 ** gamma, C = 10 ** C)
        classifier.fit(train_x_prime, train_y_prime)
        # print(f'*************** log_10 gamma is {gamma} ****************')
        # print(f'support vectors are: {classifier.support_vectors_}\ncoefficients are: {classifier.dual_coef_}\nsupport vectors\'s indices are {classifier.support_}\n')
        result = classifier.predict(test_x)
        error_array = np.not_equal(result, test_y)
        Ein = np.sum(error_array) / len(error_array)
        Ein_list.append(Ein)
        # w = classifier.coef_
        # all_w.append(np.linalg.norm(w))
        # n_support_vectors.append(classifier.n_support_)
    best_gamma_count[np.argmin(Ein_list)] += 1
    print(f'its the {i}-th exp:')

plt.bar(gamma_list, best_gamma_count, color = '#838BC5')
plt.xlabel('$\log_{10} \gamma$')
plt.ylabel('number of time that log_10 gamma is chosen')
plt.savefig('16.jpg')
f.close()
