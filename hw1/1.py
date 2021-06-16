import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X_1 = [1, 0, 0, -1, 0, 0, -2]
X_2 = [0, 1, -1, 0, 2, -2, 0]
Y = [-1, -1, -1, 1, 1, 1, 1]

def phi_1(i):
    return X_2[i] ** 2 - 2 * X_1[i] - 2

def phi_2(i):
    return X_1[i] ** 2 - 2 * X_2[i] - 1



X_1prime = []
X_2prime = []

for i in range(len(X_1)):
    X_1prime.append(phi_1(i))
    X_2prime.append(phi_2(i))

for i in range(len(X_1)):
    print(f'({X_1[i]}, {X_2[i]}) -> ({X_1prime[i]}, {X_2prime[i]}) , y = {Y[i]}')

scatter_color = []
for i in range(len(Y)):
    if Y[i] == -1:
        scatter_color.append('red')
    else:
        scatter_color.append('green')


plt.scatter(X_1prime, X_2prime, c = scatter_color)
plt.plot([-0.5, -0.5], [5, -6], color = 'black')
plt.savefig('1.jpg')
