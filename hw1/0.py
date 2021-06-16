# environment setup:
#   `pip3 install numpy scikit-learn matplotlib`
# keyword:
#   soft-margin SVM

# reference: https://stackoverflow.com/questions/33843981/under-what-parameters-are-svc-and-linearsvc-in-scikit-learn-equivalent
# conclsion: LinearSVC is not linear SVM, do not use it if do not have to.

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from os import mkdir, listdir

X_train, Y_train = None, None
X_test, Y_test = None, None


def load_data():
    global X_train, Y_train, X_test, Y_test
    for name in ('train', 'test'):
        content = np.array([ [ int(line.split()[0][0]), float(line.split()[1]), float(line.split()[2]) ] for line in open(f'features.{name}', 'r').readlines() ])
        globals()[f'Y_{name}'], globals()[f'X_{name}'] = np.split(content, [1], axis=1)
        globals()[f'Y_{name}'].shape = globals()[f'Y_{name}'].shape[0]


def problem_11():
    log_C = (-5, -3, -1, 1, 3)
    print('# problem 11')

    # obtain data
    if 'cache' not in listdir(): mkdir('cache')
    if 'clfs_11.pickle' not in listdir('./cache'):
        print('Cache not found.')
        clfs = {}
        for _log_C in log_C:
            print(f'Generating the SVM where log10(C) = {_log_C} ... ', end='', flush=True)
            clfs[_log_C] = svm.SVC(
                C=10**_log_C, kernel='linear', shrinking=False, cache_size=1024
            ).fit(X_train, Y_train == 0)
            print('Done!')
        with open('./cache/clfs_11.pickle', 'wb') as f:
            pickle.dump(clfs, f)
        print('Cache created.')
    else:
        print('Found cache. Using cache.')
        with open('./cache/clfs_11.pickle', 'rb') as f:
            clfs = pickle.load(f)

    w_norm = [ np.linalg.norm(_clf.coef_[0]) for _, _clf in clfs.items() ]

    # plot
    plt.plot( log_C, w_norm )
    plt.title( 'Problem 11' )
    plt.xlabel( r'$\log_{10} C$' )
    plt.ylabel( r'$||w||$', rotation=0 )
    if 'out_images' not in listdir(): mkdir('out_images')
    plt.savefig('./out_images/11.png')
    print('output image: ./out_images/11.png')
    print()


def problem_12_13():
    log_C = (-5, -3, -1, 1, 3)
    print('# problem 12')

    # obtain data
    if 'cache' not in listdir(): mkdir('cache')
    if 'clfs_12.pickle' not in listdir('./cache'):
        print('Cache not found.')
        clfs = {}
        for _log_C in log_C:
            print(f'Generating the SVM where log10(C) = {_log_C} ... ', end='', flush=True)
            clfs[_log_C] = svm.SVC(
                C=10**_log_C, kernel='poly', degree=2, gamma=1, coef0=1, shrinking=False, cache_size=1024
            ).fit( X_train, Y_train == 8 )
            print('Done!')
        with open('./cache/clfs_12.pickle', 'wb') as f:
            pickle.dump(clfs, f)
        print('Cache created.')
    else:
        print('Found cache. Using cache.')
        with open('./cache/clfs_12.pickle', 'rb') as f:
            clfs = pickle.load(f)

    Ein = [ _clf.score(X_train, Y_train == 8) for _, _clf in clfs.items() ]

    # plot
    plt.plot( log_C, Ein )
    plt.title( 'Problem 12' )
    plt.xlabel( r'$\log_{10} C$' )
    plt.ylabel( r'$E_{in}$', rotation=0 )
    if 'out_images' not in listdir(): mkdir('out_images')
    plt.savefig('./out_images/12.png')
    print('output image: ./out_images/12.png')
    print()

    # problem 13
    print('# problem 13')
    NoSV = [ _clf.support_.shape[0] for _, _clf in clfs.items() ]

    # plot
    plt.plot( NoSV, Ein )
    plt.title( 'Problem 13' )
    plt.xlabel( r'$\log_{10} C$' )
    plt.ylabel( r'$\#SV$', rotation=0 )
    if 'out_images' not in listdir(): mkdir('out_images')
    plt.savefig('./out_images/13.png')
    print('output image: ./out_images/13.png')
    print()


if __name__ == "__main__":
    load_data()
    # problem_11()
    problem_12_13()
