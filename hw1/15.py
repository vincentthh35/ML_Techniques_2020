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

f.close()
f = open('features.test', 'r')
lines = f.readlines()
test_data = np.array([ [ int(line.split()[0][0]), *map(float, line.split()[1:]) ] for line in lines ])
test_y = np.where(test_data[:,0] == target_number, 1, -1)
test_x = test_data[:,1:]

# print(train_y, train_x)
# print(f'{len(train_y)}, {len(train_x)}')
# print(train_y, train_x)
C_list = [-3, -2, -1, 0, 1]
# log_10 C = 0.1 does better
C = -1
gamma_list = [0, 1, 2, 3, 4]
gamma = 80
all_Eout = []
# all_w = []
# all_Ein = []
# all_distance = []
# n_support_vectors = []
for gamma in gamma_list:
    classifier = svm.SVC(kernel = 'rbf', gamma = 10 ** gamma, C = 10 ** C)
    classifier.fit(train_x, train_y)
    print(f'*************** log_10 gamma is {gamma} ****************')
    # print(f'support vectors are: {classifier.support_vectors_}\ncoefficients are: {classifier.dual_coef_}\nsupport vectors\'s indices are {classifier.support_}\n')
    result = classifier.predict(test_x)
    error_array = np.not_equal(result, test_y)
    Eout = np.sum(error_array) / len(error_array)
    all_Eout.append(Eout)
    # w = classifier.coef_
    # all_w.append(np.linalg.norm(w))
    # n_support_vectors.append(classifier.n_support_)


plt.plot(gamma_list, all_Eout, color = 'brown')
plt.xlabel('$\log_{10} \gamma$')
plt.ylabel('Eout')
plt.savefig('15.jpg')
f.close()
