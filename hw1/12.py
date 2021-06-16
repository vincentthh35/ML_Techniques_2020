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
# n_support_vectors = []
for C in C_list:
    classifier = svm.SVC(kernel = 'poly', degree = 2, gamma = 1, coef0 = 1, C = 10 ** C, shrinking = False)
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
    Ein = np.sum(error_array) / len(error_array)
    all_Ein.append(Ein)
    # w = classifier.coef_
    # all_w.append(np.linalg.norm(w))
    # n_support_vectors.append(classifier.n_support_)


plt.plot(C_list, all_Ein, color = '#4CA967')
plt.xlabel('$\log_{10} C$')
plt.ylabel('Ein')
plt.savefig('12.jpg')
f.close()
