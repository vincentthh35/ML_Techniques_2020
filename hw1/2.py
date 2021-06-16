from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def kernel(x_1, x_2):
    ans = 0
    for i in range(len(x_1)):
        ans += x_1[i] * x_2[i]
    return (1 + ans) ** 2

X_1 = [1, 0, 0, -1, 0, 0, -2]
X_2 = [0, 1, -1, 0, 2, -2, 0]
Y_0 = [-1, -1, -1, 1, 1, 1, 1]
n = len(X_1)

X = np.array([X_1, X_2])
X = np.transpose(X)
classifier = svm.SVC(kernel = 'poly', degree = 2, gamma = 1, coef0 = 1, C = 1e10, shrinking = False)
classifier.fit(X, Y_0)
print(f'support vectors are: {classifier.support_vectors_}\ncoefficients are: {classifier.dual_coef_}\nsupport vectors\'s indices are {classifier.support_}\n')
# calculate coefficients
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

print(f'the nonlinear curve is:')
print(f'{a0} + {a1}x_1 + {a2}x_2 + {a3}(x_1)^2 + {a4}x_1x_2 + {a5}(x_2)^2')

# plot
delta = 0.01
xrange = np.arange(-4, 4, delta)
yrange = np.arange(-4, 4, delta)
X, Y = np.meshgrid(xrange, yrange)
plt.contour(X, Y, -1.666 -1.777 * X + 0.8887 * (X ** 2) + 0.6665 * (Y ** 2), [0])

mrange = np.arange(-4, 4, delta)
nrange = np.arange(-4, 4, delta)
M, N = np.meshgrid(mrange, nrange)
plt.contour(M, N, N ** 2 - 2 * M - 1.5, [0], colors = 'blue')

# gen color
scatter_color = []
for i in range(len(X_1)):
    if Y_0[i] == -1:
        scatter_color.append('red')
    else:
        scatter_color.append('green')
plt.scatter(X_1, X_2, c = scatter_color)

plt.savefig('5.jpeg')
