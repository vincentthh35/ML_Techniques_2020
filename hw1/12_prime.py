from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import math

target_number = 8

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
C_list = [-5, -3, -1, 1, 3]
all_w = []
all_Ein = []
color_list = ['red', 'green', 'yellow', 'blue', 'black']
# n_support_vectors = []
for C in C_list[:2]:
    classifier = svm.SVC(kernel = 'poly', degree = 2, gamma = 1, coef0 = 1, C = 10 ** C)
    classifier.fit(train_x, train_y)
    print(f'*************** C is {C} ****************')
    print(f'support vectors are: {classifier.support_vectors_}\ncoefficients are: {classifier.dual_coef_}\nsupport vectors\'s indices are {classifier.support_}\n')
    result = classifier.predict(train_x)
    error_array = np.not_equal(result, train_y)
    # count = 0
    # for i in error_array:
    #     if i == True:
    #         count  = count + 1
    # print(f'np.sum: {np.sum(error_array)}, my calculation: {count}, mean():{np.mean(result != train_y)}')
    a0 = float(classifier.intercept_)
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    a5 = 0
    for i in range(len(classifier.support_)):
        a0 = a0 + classifier.dual_coef_[0][i]
        a1 = a1 + classifier.support_vectors_[i][0] * 2 * classifier.dual_coef_[0][i]
        a2 = a2 + classifier.support_vectors_[i][1] * 2 * classifier.dual_coef_[0][i]
        a3 = a3 + (classifier.support_vectors_[i][0] ** 2) * classifier.dual_coef_[0][i]
        a4 = a4 + (classifier.support_vectors_[i][0] * classifier.support_vectors_[i][1] * 2) * classifier.dual_coef_[0][i]
        a5 = a5 + (classifier.support_vectors_[i][1] ** 2) * classifier.dual_coef_[0][i]
    a0, a1, a2, a3, a4, a5 = a0 * 1e5, a1 * 1e5, a2 * 1e5, a3 * 1e5, a4 * 1e5, a5 * 1e5
    print(f'the nonlinear curve is:')
    print(f'{a0} + {a1}x_1 + {a2}x_2 + {a3}(x_1)^2 + {a4}x_1x_2 + {a5}(x_2)^2')
    delta = 0.002
    mrange = np.arange(-10, 1, delta)
    nrange = np.arange(-10, 10, delta)
    M, N = np.meshgrid(mrange, nrange)
    plt.contour(M, N, a0 + a1 * M + a2 * N + a3 * (M ** 2) + a4 * M * N + a5 * (N ** 2), [0, 1, 2, 3, 4], colors = color_list[C_list.index(C)])
    # w = classifier.coef_
    # all_w.append(np.linalg.norm(w))
    # n_support_vectors.append(classifier.n_support_)


plt.savefig('12_line_prime.jpg')
f.close()
