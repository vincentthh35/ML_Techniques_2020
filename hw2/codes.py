import sys
import os
import math
import pickle
import numpy as np
from numpy import linalg as LA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

lr = 0.1
T = 5000
log_2d_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
INPUT_DIM = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu' # expected device is cuda
# def error_function():

class MyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, input_layer=None):
        super(MyLinear, self).__init__(in_features=in_features, out_features=out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # constrained
        if input_layer is not None:
            self.my_transpose(input_layer)
            if self.bias is not None:
                bound = math.sqrt(6) / math.sqrt(1 + in_features + out_features)
                nn.init.uniform_(self.bias, -bound, bound)
        # not constrained
        else:
            self.my_reset_parameters(d_1=in_features, d_2=out_features)

    def my_reset_parameters(self, d_1, d_2):
        bound = math.sqrt(6) / math.sqrt(1 + d_1 + d_2)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def my_transpose(self, input_layer):
        self.weight = nn.Parameter(torch.transpose(input_layer.weight, 0, 1))

class MyAutoEncoder(nn.Module):
    def __init__(self, constrained, d_tilde, d=INPUT_DIM):
        super(MyAutoEncoder, self).__init__()

        # encoder
        my_linear = MyLinear(in_features=d, out_features=d_tilde)
        self.encoder = nn.Sequential(
            my_linear,
            nn.Tanh(),
        ).to(device)

        # decoder
        self.decoder = nn.Sequential(
            MyLinear(
                in_features=d_tilde,
                out_features=d,
                input_layer=my_linear
            ) if constrained else MyLinear(
                in_features=d_tilde,
                out_features=d
            ),
        ).to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def to_tensor(x, y):
    print(f'to_tensor: device={device}')
    return torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)

def go_a_step(model):
    model.encoder[0].weight -= lr * model.encoder[0].weight.grad
    model.encoder[0].bias -= lr * model.encoder[0].bias.grad
    model.decoder[0].weight -= lr * model.decoder[0].weight.grad
    model.decoder[0].bias -= lr * model.decoder[0].bias.grad

def save_model_as(model, prob, constrained, d_tilde): # for cuda
    if not os.path.isdir( './models' ):
        os.makedirs( './models' )
    torch.save(model.state_dict(), f'./models/prob{prob}_{constrained}_{d_tilde}.pt')
    return

def load_model(prob, constrained, d_tilde): # load to gpu
    model = MyAutoEncoder(constrained=constrained, d_tilde=(2 ** d_tilde), d=INPUT_DIM)
    model.load_state_dict( torch.load( f'./models/prob{prob}_{constrained}_{d_tilde}.pt' ) )

    model.to(device)

    return model

def mean_shift(x):
    x_mean = np.mean(x, axis=0)
    new_x = np.subtract(x, x_mean)
    return new_x, x_mean

def g(x, x_mean, w):
    wwt = np.matmul( np.transpose(w), w )
    return np.transpose(np.add( np.matmul( wwt, np.transpose(np.subtract( x, x_mean )) ), np.transpose(x_mean) ))

def input_test():
    f = open('./zip.test')
    lines = f.readlines()
    all_data = np.array([ [ int(line.split()[0][0]), *map(float, line.split()[1:]) ] for line in lines ])
    x = all_data[:,1:]
    y = all_data[:,0]
    f.close()
    return x, y

def input_train():
    f = open('./zip.train')
    lines = f.readlines()
    all_data = np.array([ [ int(line.split()[0][0]), *map(float, line.split()[1:]) ] for line in lines ])
    x = all_data[:,1:]
    y = all_data[:,0]
    f.close()
    return x, y

def prob_11():
    train_x, train_y = input_train()
    tensor_x, tensor_y = to_tensor(train_x, train_y)
    Ein_list = []


    for d_tilde in log_2d_list:
        my_auto_encoder = MyAutoEncoder(constrained=False, d=INPUT_DIM, d_tilde=(2 ** d_tilde))
        loss_func = nn.MSELoss()

        for step in range(T):
            encoded, decoded = my_auto_encoder(tensor_x)
            loss = loss_func(decoded, tensor_x)
            my_auto_encoder.zero_grad()
            loss.backward()

            # one step
            with torch.no_grad():
                go_a_step(my_auto_encoder)

            if step % 500 == 0:
                print(f'step: {step}')

        # save model
        save_model_as(my_auto_encoder, 11, False, d_tilde)

        # prepare list
        Ein_list.append( nn.functional.mse_loss(decoded, tensor_x).item() )

    print(f'Ein_list: {Ein_list}')
    # plot
    plt.plot(log_2d_list, Ein_list, color='#30C758')
    plt.xlabel('$\log_2 (tilde) d$')
    plt.ylabel('avg err$_{x}$ on zip.train')
    plt.savefig('./11.jpg')
    return

def prob_12():
    test_x, test_y = input_test()
    tensor_test_x, tensor_test_y = to_tensor(test_x, test_y)
    Eout_list = []

    for d_tilde in log_2d_list:
        my_auto_encoder = load_model(11, False, d_tilde)

        encoded, decoded = my_auto_encoder(tensor_test_x)

        Eout_list.append( nn.functional.mse_loss(decoded, tensor_test_x).item() )

    print(f'Eout_list: {Eout_list}')
    # plot
    plt.plot(log_2d_list, Eout_list, color='#6f549E')
    plt.xlabel('$\log_2 (tilde) d$')
    plt.ylabel('avg err$_{x}$ on zip.test')
    plt.savefig('./12.jpg')
    return

def prob_13():
    train_x, train_y = input_train()
    tensor_x, tensor_y = to_tensor(train_x, train_y)
    Ein_list = []

    for d_tilde in log_2d_list:
        my_auto_encoder = MyAutoEncoder(constrained=True, d=INPUT_DIM, d_tilde=(2 ** d_tilde))
        loss_func = nn.MSELoss()

        for step in range(T):
            encoded, decoded = my_auto_encoder(tensor_x)
            loss = loss_func(decoded, tensor_x)
            my_auto_encoder.zero_grad()
            loss.backward()

            # one step
            with torch.no_grad():
                go_a_step(my_auto_encoder)

            if step % 500 == 0:
                print(f'step: {step}')

        # save model
        save_model_as(my_auto_encoder, 13, True, d_tilde)

        # prepare list
        Ein_list.append( nn.functional.mse_loss(decoded, tensor_x).item() )

    print(f'Ein_list: {Ein_list}')
    # plot
    plt.plot(log_2d_list, Ein_list, color='#FAC150')
    plt.xlabel('$\log_2 (tilde) d$')
    plt.ylabel('avg err$_{x}$ on zip.train with constraint')
    plt.savefig('./13.jpg')
    return

def prob_14():
    test_x, test_y = input_test()
    tensor_test_x, tensor_test_y = to_tensor(test_x, test_y)
    Eout_list = []

    for d_tilde in log_2d_list:
        my_auto_encoder = load_model(13, True, d_tilde)

        encoded, decoded = my_auto_encoder(tensor_test_x)

        Eout_list.append( nn.functional.mse_loss(decoded, tensor_test_x).item() )

    print(f'Eout_list: {Eout_list}')
    # plot
    plt.plot(log_2d_list, Eout_list, color='#F58792')
    plt.xlabel('$\log_2 (tilde) d$')
    plt.ylabel('avg err$_{x}$ on zip.test with constraint')
    plt.savefig('./14.jpg')
    return

def prob_15():
    global log_2d_list
    log_2d_list = log_2d_list[:7]
    train_x, train_y = input_train()
    Ein_list = []

    for d_tilde in log_2d_list:
        print(f'd_tilde: {d_tilde}')
        copy_x = train_x.copy()
        x, x_mean = mean_shift(copy_x)
        xtx = np.matmul( np.transpose(x), x )
        eigvalue, eigvector = LA.eig(xtx)
        target = eigvalue.argsort()[::-1]
        # print(target)
        target = target[:(2 ** d_tilde)]
        # print(target)
        # print(eigvector[target[0]])
        w = np.array([ eigvector[:,i] for i in target ])
        # print(w)
        # w = np.transpose(w)
        err_x = []
        for temp_x in train_x:
            err_vector = np.subtract( g(temp_x, x_mean, w), temp_x )
            err_vector = np.square(err_vector)
            err_x.append(np.mean(err_vector))
            # print(np.mean(err_vector))

        outfile = open(f'./models/PCA_{d_tilde}', 'wb')
        pickle.dump(w, outfile)
        outfile.close()

        err_x_mean = np.mean(err_x)
        Ein_list.append(err_x_mean)

    print(f'Ein_list: {Ein_list}')
    # plot
    plt.plot(log_2d_list, Ein_list, color='#86B0BD')
    plt.xlabel('$\log_2 (tilde)d$')
    plt.ylabel('avg err$_{x}$ on zip.train of PCA')
    plt.savefig('./15.jpg')
    return

def prob_16():
    global log_2d_list
    log_2d_list = log_2d_list[:7]
    test_x, test_y = input_test()
    Eout_list = []

    for d_tilde in log_2d_list:
        print(f'd_tilde: {d_tilde}')
        copy_x = test_x.copy()
        x, x_mean = mean_shift(copy_x)
        # load w
        infile = open(f'./models/PCA_{d_tilde}', 'rb')
        w = pickle.load(infile)
        infile.close()
        err_x = []
        for temp_x in test_x:
            err_vector = np.subtract( g(temp_x, x_mean, w), temp_x )
            err_vector = np.square(err_vector)
            err_x.append(np.mean(err_vector))
            # print(np.mean(err_vector))

        err_x_mean = np.mean(err_x)
        Eout_list.append(err_x_mean)

    print(f'Eout_list: {Eout_list}')
    # plot
    plt.plot(log_2d_list, Eout_list, color='#F2BB5C')
    plt.xlabel('$\log_2 (tilde)d$')
    plt.ylabel('avg err$_{x}$ on zip.test of PCA')
    plt.savefig('./16.jpg')
    return

def main():
    if len(sys.argv) == 2:
        try:
            prob = int(sys.argv[1])
        except:
            print('something wrong in argument')
            sys.exit(1)
        if prob == 11:
            prob_11()
        elif prob == 12:
            prob_12()
        elif prob == 13:
            prob_13()
        elif prob == 14:
            prob_14()
        elif prob == 15:
            prob_15()
        elif prob == 16:
            prob_16()
        else:
            print('wrong problem number')
            exit(1)
    else:
        print('wrong number of argument')
        sys.exit(1)

if __name__ == '__main__':
    main()
