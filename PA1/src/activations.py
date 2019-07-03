import numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x,axis=0)
    return np.exp(x)/np.sum(np.exp(x),axis=0,keepdims=True)

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def grad_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

def grad_tanh(x):
    return (1 -tanh(x))*(1 + tanh(x))

def grad_relu(x):
    x[x>0] =1
    x[x<0] = 0
    return x

