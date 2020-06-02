import pandas as pd
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rbf(X, X_train, sigma=0.3):
    # Assuming values passed are numpy arrays
    X = torch.from_numpy(X)
    X = X.to(device)
    m, n = list(X.shape)
    K = torch.zeros((m, n), device=device)
    for i in range(0, m):
        for j in range(0, n):
            temp = X[i] - X_train[j]
            dist = torch.dot(temp, temp.t())
            K[i, j] = torch.exp(-1 * dist / (2 * (sigma ** 2)))
    K = torch.cat(torch.ones((m, 1), device=device), K, 1)
    return K


def rbf_train(X_train, y_train, reg, L_rate, sigma=0.3, iters=500):
    global X_train
    X_train = torch.from_numpy(X_train)
    m, n = list(X_train.shape)
    y_train = torch.from_numpy(y_train)
    y_train = y_train.view(-1, 1)
    y_train[y_train == 0] = -1
    K_train = rbf(X_train, X_train, sigma)  # rbf kernel
    theta = torch.zeros((m + 1, 1), device=device)
    # cost = 1000
    for i in range(iters):
        der = torch.zeros((m + 1, 1), device=device)

        # Train Theta
        for j in range(m):
            k = K_train[j]
            k = k.view(1, -1)
            z = torch.dot(k, theta)
            if (y_train[j] * z < 1):
                der = der + theta - reg * y_train[j] * k.T
            else:
                der = der + theta
        # print ('der ={}'.format(der))
        der[0] = 0
        # Update Theta
        theta = theta - L_rate * der / m
        # print ('theta={}'.format(theta))

        # Find cost for updated Theta
        # Z = torch.dot(K_train, theta)
        # cost = 0
        # cost = cost + reg * torch.sum(torch.maximum(torch.zeros((m, 1)), 1 - torch.mul(y_train.t(), Z)))
        # cost = cost + torch.dot(theta.t(), theta) - theta[0] ** 2
        # if i % 50 == 0:
        #     print("Cost after iter", i, "=", cost)

    # Return trained Theta
    theta_trained = theta.cpu().detach().clone().numpy()
    return theta_trained


def rbf_predict(X_test, theta, sigma=0.3):
    X_test = torch.from_numpy(X_test)
    m, n = list(X_test.shape)
    K_test = rbf(X_test, X_train, sigma)
    y_test = torch.dot(K_test, theta)

    for i in range(m):
        if y_test[i] >= 0:
            y_test[i] = 1
        else:
            y_test[i] = 0

    return y_test
