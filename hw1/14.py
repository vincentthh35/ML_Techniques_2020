from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
gamma = 80
# all_w = []
# all_Ein = []
all_distance = []
# n_support_vectors = []
for C in C_list:
    classifier = svm.SVC(kernel = 'rbf', gamma = gamma, C = 10 ** C)
    classifier.fit(train_x, train_y)
    print(f'*************** log_10 C is {C} ****************')
    # print(f'support vectors are: {classifier.support_vectors_}\ncoefficients are: {classifier.dual_coef_}\nsupport vectors\'s indices are {classifier.support_}\n')
    X = train_x[classifier.support_]
    N = len(X)
    print(f'N = {N}')
    alpha_y = classifier.dual_coef_[0] * train_y[classifier.support_]
    K = np.array([ [ np.square(X[i] - X[j]).sum() for j in range(N) ] for i in range(N) ])
    K = np.exp(- gamma * K)
    Y = classifier.dual_coef_[0]
    w_square = Y.dot(K).dot(Y.T)
    w_norm = np.sqrt(w_square)
    print('start of exp:')
    # experiment : take the first support to calculate SUM(alpha * y * K(x, x')) + b
    g_svm = classifier.intercept_[0]
    print(f'the type of g_svm is {type(g_svm)}, and g_svm is {g_svm}')
    # for i in range(N):
    #     g_svm = g_svm + classifier.dual_coef_[0][i] * rbf_kernel(X[i], X[0], gamma)
    g_svm = 1
    print(f'wx + b = {g_svm}')
    print(f'w.norm() = {w_norm}')
    all_distance.append(np.absolute(g_svm) / w_norm)
    print(f'distance = {np.absolute(g_svm) / w_norm}')
    # w = classifier.coef_
    # all_w.append(np.linalg.norm(w))
    # n_support_vectors.append(classifier.n_support_)


plt.plot(C_list, all_distance, color = '#997FD7')
plt.xlabel('$\log_{10} C$')
plt.ylabel('Distance of free support vector to hyperplane')
plt.savefig('14.jpg')
f.close()
