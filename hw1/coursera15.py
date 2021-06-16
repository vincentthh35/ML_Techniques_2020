from svm import *
from svmutil import *
import math

# train_y, train_x = svm_read_problem('features.train')

# convert data
target_number = 0
f = open('features.train', 'r')
train_y = []
train_x = []
lines = f.readlines()
# i = 1
for line in lines:
    y, x_1, x_2 = map(float, line.split())
    train_x.append({1: x_1, 2: x_2})
    if int(y) == target_number:
        y = 1
    else:
        y = -1
    train_y.append(y)
    f.readline()
    # print(str(y))
    # print(f'{i}success')
    # i += 1
f.close()
print('success with reading training data')
prob = svm_problem(train_y, train_x)

# test_y, test_x = svm_read_problem('features.test')
parameters = svm_parameter('-t 0 -c 0.01 -b 0 ')
model = svm_train(prob, parameters)
svm_save_model("train15.model", model)
# w = model.SVs * model.sv_coef
# print(w)
# calculate w
f = open('train15.model')
lines = f.readlines()
lines = lines[8:]
axis_1 , axis_2 = 0.0, 0.0
for line in lines:
    alpha, label_1, label_2 = line.split()
    alpha = float(alpha)
    label_1 = float(label_1.split(':')[1])
    label_2 = float(label_2.split(':')[1])
    axis_1 += alpha * (label_1)
    axis_2 += alpha * (label_2)

# get the length of w
print(f'label_1 = {label_1}, label_2 = {label_2}, ||w|| = {math.sqrt(label_1 ** 2 + label_2 ** 2)}')
