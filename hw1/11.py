from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import math

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
C_list = [-5, -3, -1, 1, 3]
all_w = []
for C in C_list:
    classifier = svm.SVC(kernel = 'linear', C = 10 ** C, shrinking = False)
    classifier.fit(train_x, train_y)
    print(f'*************** C is {C} ****************')
    print(f'support vectors are: {classifier.support_vectors_}\ncoefficients are: {classifier.dual_coef_}\nsupport vectors\'s indices are {classifier.support_}\n')
    result = classifier.predict(test_x)
    error_array = np.not_equal(result, test_y)
    Eout = np.sum(error_array) / len(error_array)
    w = classifier.coef_
    print(f'Eout is {Eout}')
    print(f'b is: {classifier.intercept_}, w is {w}')
    all_w.append(np.linalg.norm(w))

plt.plot(C_list, all_w, color = 'orange')
plt.xlabel('$\log_{10} C$')
plt.ylabel('||w||')
plt.savefig('11.jpg')
f.close()
