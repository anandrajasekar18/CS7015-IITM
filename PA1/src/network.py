import numpy as np 
from activations import *
import copy
import pickle

def create_mini_batches(X,y,batch_size):
    comb_X_y = np.hstack([X,y])
    np.random.shuffle(comb_X_y)
    num_batches = int(X.shape[0]/batch_size)

    batches = []
    for i in range(num_batches):
        batch = comb_X_y[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = batch[:, :-10] 
        y_mini = batch[:, -10:]
        batches.append((X_mini, y_mini)) 
    return batches

def save_weights(weights_dict, epoch, save_dir):
    with open(save_dir +'weights_{}.pkl'.format(epoch), 'wb') as f:
        pickle.dump(weights_dict, f)


class build_network:
    def __init__(self,lr = 0.001,gamma = 0.9,n_hidden=2, sz=[100,100], activation ='sigmoid', save_dir ='./save_dir/', loss ='ce', opt = 'adam', weights = None,batch_sz = 256, epoch = 20, anneal=True, input_dim = 784, output_dim = 10, output_activation ='softmax',lambd = 0.001):
        self.lr = lr
        self.n_hidden = n_hidden
        self.sz = sz
        self.loss = loss
        self.opt = opt
        self.batch_sz = batch_sz
        self.epoch = epoch
        self.anneal =  anneal
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.opt_dict = {'gd':self.gd, 'momentum': self.momentum, 'nag' : self.nag, 'adam' : self.adam}
        self.step = 0
        self.lambd = lambd
        dim_list = [self.input_dim] + self.sz +[self.output_dim]
        self.param = {}
        self.train_log = []
        self.val_log = []
        self.save_dir = save_dir
        if (weights == None):
            self.param = {}
            for i in range(self.n_hidden+1):
                self.param['W_{}'.format(i+1)] = np.random.normal(0,np.sqrt(2/(dim_list[i]+dim_list[i+1])),(dim_list[i+1],dim_list[i]))
                self.param['b_{}'.format(i+1)] = np.random.normal(0,np.sqrt(1/(dim_list[i+1])),(dim_list[i+1],1))
        else:
            self.param = weights

        if (self.opt in ['momentum','nag']):
            for i in range(self.n_hidden+1):
                self.param['history_W_{}'.format(i+1)] = np.zeros((dim_list[i+1],dim_list[i]))
                self.param['history_b_{}'.format(i+1)] = np.zeros((dim_list[i+1],1))
                self.gamma = gamma
        
        elif (self.opt == 'adam'):
            for i in range(self.n_hidden+1):
                self.param['history_W_{}'.format(i+1)] = np.zeros((dim_list[i+1],dim_list[i]))
                self.param['history_b_{}'.format(i+1)] = np.zeros((dim_list[i+1],1))
                self.param['rms_W_{}'.format(i+1)] = np.zeros((dim_list[i+1],dim_list[i]))
                self.param['rms_b_{}'.format(i+1)] = np.zeros((dim_list[i+1],1))
                self.beta1 = 0.9
                self.beta2 = 0.999
                self.epsilon = 1e-8
            


    def compute_metrics(self, y_true, y_pred):
        m = y_pred.shape[1]
        norm = 0
        for i in range(self.n_hidden+1):
            norm+=np.linalg.norm(self.param['W_{}'.format(i+1)])**2

        if self.loss == 'sq':
            loss =  1/(2*m) * np.sum((y_true-y_pred)**2)
        else :
            loss = -1/m*np.sum(np.log10(y_pred[y_true.astype(bool)]))

        error = 1-(np.mean(np.argmax(y_pred,axis=0) == np.argmax(y_true,axis=0)))

        return (loss+ self.lambd/(2*m) * norm,error*100)

        
    def forward_prop(self,X):
        l_act = {}
        nl_act = {}

        nl_act['layer_{}'.format(0)] = X
        for i in range(self.n_hidden):
            l_act['layer_{}'.format(i+1)] = np.matmul(self.param['W_{}'.format(i+1)],nl_act['layer_{}'.format(i)]) + self.param['b_{}'.format(i+1)]
            nl_act['layer_{}'.format(i+1)] = eval(self.activation)(l_act['layer_{}'.format(i+1)])
        l_act['layer_{}'.format(self.n_hidden+1)] = np.matmul(self.param['W_{}'.format(self.n_hidden+1)],nl_act['layer_{}'.format(self.n_hidden)]) + self.param['b_{}'.format(self.n_hidden+1)]
        nl_act['layer_{}'.format(self.n_hidden+1)] = eval(self.output_activation)(l_act['layer_{}'.format(self.n_hidden+1)])

        return (l_act, nl_act)

    def backward_prop(self,l_act, nl_act,y_true):
        k = self.output_dim
        m = y_true.shape[1]
        if self.loss == 'sq':
            y_pred = nl_act['layer_{}'.format(self.n_hidden+1)]
            inn_nl_prod = ((y_pred-y_true) * y_pred).reshape(1,k,-1)
            tiled = np.repeat(np.identity(k), m, axis=1).reshape((k,k,-1)) - np.repeat(y_pred.reshape(k,1,-1), k, axis=1)
            grad_y = np.sum(inn_nl_prod * tiled,axis=1)
        else :
            grad_y = -(y_true - nl_act['layer_{}'.format(self.n_hidden+1)])

        
        
        grad_l_act = {}
        grad_nl_act = {}
        grads = {}

        grad_l_act['layer_{}'.format(self.n_hidden+1)] = 1/m*grad_y
        for i in range(self.n_hidden,-1,-1):
            grads['W_{}'.format(i+1)] = np.matmul(grad_l_act['layer_{}'.format(i+1)], np.transpose(nl_act['layer_{}'.format(i)]))
            grads['b_{}'.format(i+1)] = np.sum(grad_l_act['layer_{}'.format(i+1)],axis=1,keepdims=True)
            grad_nl_act['layer_{}'.format(i)] = np.matmul(np.transpose(self.param['W_{}'.format(i+1)]),grad_l_act['layer_{}'.format(i+1)])
            if (i==0):
                break
            grad_l_act['layer_{}'.format(i)] = eval('grad_' + self.activation)(l_act['layer_{}'.format(i)]) * grad_nl_act['layer_{}'.format(i)]
        return grads

    def gd(self,X,y):
        self.step += 1
        l_act, nl_act = self.forward_prop(X)
        grads = self.backward_prop(l_act,nl_act,y)
        for i in range(self.n_hidden+1):
            self.param['W_{}'.format(i+1)] = self.param['W_{}'.format(i+1)]- self.lr * grads['W_{}'.format(i+1)]- self.lambd/self.batch_sz * self.param['W_{}'.format(i+1)]
            self.param['b_{}'.format(i+1)] = self.param['b_{}'.format(i+1)]- self.lr * grads['b_{}'.format(i+1)]

        loss,error = self.compute_metrics(y,nl_act['layer_{}'.format(self.n_hidden+1)])
        return loss,error
    
    def momentum(self,X,y):
        self.step += 1
        l_act, nl_act = self.forward_prop(X)
        grads = self.backward_prop(l_act,nl_act,y)
        for i in range(self.n_hidden+1):
            self.param['history_W_{}'.format(i+1)] = self.lr * grads['W_{}'.format(i+1)] + self.gamma * self.param['history_W_{}'.format(i+1)]
            self.param['W_{}'.format(i+1)] = self.param['W_{}'.format(i+1)]- (self.param['history_W_{}'.format(i+1)]) - self.lambd/self.batch_sz * self.param['W_{}'.format(i+1)]
            self.param['history_b_{}'.format(i+1)] = self.lr * grads['b_{}'.format(i+1)] + self.gamma * self.param['history_b_{}'.format(i+1)]
            self.param['b_{}'.format(i+1)] = self.param['b_{}'.format(i+1)]- (self.param['history_b_{}'.format(i+1)])
        loss,error = self.compute_metrics(y,nl_act['layer_{}'.format(self.n_hidden+1)])
        return loss,error

    def nag(self,X,y):
        self.step += 1
        old_param = {}
        for i in range(self.n_hidden+1):
            old_param['W_{}'.format(i+1)] = copy.deepcopy(self.param['W_{}'.format(i+1)])
            old_param['b_{}'.format(i+1)] = copy.deepcopy(self.param['b_{}'.format(i+1)])
            self.param['W_{}'.format(i+1)] = self.param['W_{}'.format(i+1)]- (self.gamma * self.param['history_W_{}'.format(i+1)]) 
            self.param['b_{}'.format(i+1)] = self.param['b_{}'.format(i+1)]- (self.gamma * self.param['history_b_{}'.format(i+1)])

        l_act, nl_act = self.forward_prop(X)
        grads = self.backward_prop(l_act,nl_act,y)

        for i in range(self.n_hidden+1):
            self.param['history_W_{}'.format(i+1)] = self.lr * grads['W_{}'.format(i+1)] + self.gamma * self.param['history_W_{}'.format(i+1)]
            self.param['W_{}'.format(i+1)] = old_param['W_{}'.format(i+1)]- (self.param['history_W_{}'.format(i+1)]) - self.lambd/self.batch_sz * self.param['W_{}'.format(i+1)]
            self.param['history_b_{}'.format(i+1)] = self.lr * grads['b_{}'.format(i+1)] + self.gamma * self.param['history_b_{}'.format(i+1)]
            self.param['b_{}'.format(i+1)] = old_param['b_{}'.format(i+1)]- (self.param['history_b_{}'.format(i+1)])

        loss,error = self.compute_metrics(y,nl_act['layer_{}'.format(self.n_hidden+1)])
        return loss,error

    def adam(self,X,y):
        self.step += 1
        l_act, nl_act = self.forward_prop(X)
        grads = self.backward_prop(l_act,nl_act,y)
        for i in range(self.n_hidden+1):
            self.param['history_W_{}'.format(i+1)] = (1-self.beta1) * grads['W_{}'.format(i+1)] + self.beta1 * self.param['history_W_{}'.format(i+1)]
            self.param['rms_W_{}'.format(i+1)] = (1-self.beta2) * grads['W_{}'.format(i+1)]**2 + self.beta2 * self.param['rms_W_{}'.format(i+1)]
            corrected_history = self.param['history_W_{}'.format(i+1)]/(1 - self.beta1**(self.step))
            corrected_rms = self.param['rms_W_{}'.format(i+1)]/(1 - self.beta2**(self.step))
            self.param['W_{}'.format(i+1)] = self.param['W_{}'.format(i+1)] - (self.lr/(np.sqrt(corrected_rms + self.epsilon ))*corrected_history) - self.lambd/self.batch_sz * self.param['W_{}'.format(i+1)]

            self.param['history_b_{}'.format(i+1)] = (1-self.beta1) * grads['b_{}'.format(i+1)] + self.beta1 * self.param['history_b_{}'.format(i+1)]
            self.param['rms_b_{}'.format(i+1)] = (1-self.beta2) * grads['b_{}'.format(i+1)]**2 + self.beta2 * self.param['rms_b_{}'.format(i+1)]
            corrected_history = self.param['history_b_{}'.format(i+1)]/(1 - self.beta1**(self.step))
            corrected_rms = self.param['rms_b_{}'.format(i+1)]/(1 - self.beta2**(self.step))
            self.param['b_{}'.format(i+1)] = self.param['b_{}'.format(i+1)] - (self.lr/(np.sqrt(corrected_rms + self.epsilon ))*corrected_history)

        loss,error = self.compute_metrics(y,nl_act['layer_{}'.format(self.n_hidden+1)])
        return loss,error
    
    def predict(self,X,y):
        _,nl_act = self.forward_prop(X)
        loss,error = self.compute_metrics(y,nl_act['layer_{}'.format(self.n_hidden+1)])
        return loss,error
    
    def train(self,X,y,X_val,y_val,state = None):
        prev_loss = self.predict(np.transpose(X_val),np.transpose(y_val))[0]
        n_steps = X.shape[0]/self.batch_sz
        if state == None:
            j = 1
        else:
            j = state+1
        while j < self.epoch+1:
            epoch_train_log =[]
            epoch_val_log = []
            batches = create_mini_batches(X,y,self.batch_sz)
            param = copy.deepcopy(self.param)
            for k,mini_batch in enumerate(batches):
                mini_X, mini_y = mini_batch
                self.opt_dict[self.opt](np.transpose(mini_X),np.transpose(mini_y))
                if (k!=0 and k%100 == 0):
                    epoch_train_log+= [[j,k] + list(self.predict(np.transpose(X),np.transpose(y)))+ [self.lr]]
                    epoch_val_log+= [[j,k] + list(self.predict(np.transpose(X_val),np.transpose(y_val)))+ [self.lr]]
            nloss,error = self.predict(np.transpose(X_val),np.transpose(y_val))
            if (nloss > prev_loss and self.anneal == True):
                self.param = param
                self.step -= n_steps
                self.lr = self.lr/2
                j = j-1
            else :
                prev_loss = nloss
                self.train_log.extend(epoch_train_log)
                self.val_log.extend(epoch_val_log)
                weights_list = []
                for i in range(self.n_hidden+1):
                    weights_list.append(self.param['W_{}'.format(i+1)])
                biases_list = []
                for i in range(self.n_hidden+1):
                    biases_list.append(self.param['b_{}'.format(i+1)])
                save_weights(weights_list+biases_list,j,self.save_dir)
            j = j+1
            


