import pandas as pd
import numpy as np 
from network import build_network
from sklearn.decomposition import PCA
import pickle
import argparse

def load_weights(state,save_dir):
    with open(save_dir +'weights_{}.pkl'.format(state),'rb') as f:
              weight_dict = pickle.load(f)
    return weight_dict


parser = argparse.ArgumentParser(description='Classification using neural networks')

parser.add_argument('--lr', type=float,default= 0.001, help='Learning rate for gradient descent')
parser.add_argument('--momentum', type=float, default=0.9, help='coefficient for momentum(gamma)')
parser.add_argument('--num_hidden', type=int, default=2 ,help='number of hidden layers')
parser.add_argument('--sizes', type=lambda x: x.split(','), default=[100,100], help='Size of hidden layers- Provide as comma separated list')
parser.add_argument('--activation', default='sigmoid',help='activation function (tanh or sigmoid)')
parser.add_argument('--loss', default='ce', help='loss function (sq for Squared error loss or ce for cross entropy')
parser.add_argument('--opt', default='adam',help='Optimization/Learning algorithm (gd, momentum, nag, or adam)')
parser.add_argument('--batch_size', type=int,default=20,help='batch size (1 or multiples of 5)')
parser.add_argument('--epochs',type= int ,default=5, help='number of passes over the data')
parser.add_argument('--anneal', default='True', help='If true,learning rate is halved if at any epoch the validation loss decreases and then restarts that epoch')
parser.add_argument('--save_dir', default='./save_dir/',help='the directory in which the pickled model should be saved')
parser.add_argument('--expt_dir', default='./expt_dir/', help='(the directory in which the log files will be saved')
parser.add_argument('--train', default='test.csv',help='path to training set')
parser.add_argument('--val', default='valid.csv',help='path to validation set')
parser.add_argument('--test', default='test.csv',help='path to test set')
parser.add_argument('--pretrain', default = 'False', help='If true it will load the weights from 12th epoch and start running')
parser.add_argument('--testing', default = 'True', help='If true, the code will evaluate the test set')
parser.add_argument('--state', type=int, default = 1, help='state')
parser.add_argument('--input_dim', type=int, default =75, help='number of compnents for PCA')
parser.add_argument('--lambd', type=float, default =0, help='Regularisation parameter')

args = parser.parse_args()


def cvt_bool(a):
	if a=='False' or a=='false':
		return False
	if a=='True' or a=='true':
		return True

args.sizes = list(map(int,args.sizes))
args.testing=cvt_bool(args.testing)
args.pretrain=cvt_bool(args.pretrain)
args.anneal=cvt_bool(args.anneal)

args.sizes = list(map(int, args.sizes))



train_dir=args.train
val_dir=args.val
test_dir=args.test


np.random.seed(1234)
train = pd.read_csv(train_dir,header = 0,index_col = 0).values
val = pd.read_csv(val_dir,header = 0,index_col = 0).values
test = pd.read_csv(test_dir,header = 0,index_col = 0)


X_train = train[:,:-1]/255
X_val = val[:,:-1]/255
X_test = test.values/255


pca = PCA(n_components=args.input_dim, whiten=True)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

y_train = train[:,-1].reshape((-1))
y_val = val[:,-1].reshape((-1))


y_hat = np.zeros((55000,10))
y_hat[range(55000),y_train.astype(int)] = 1

y_valn = np.zeros((5000,10))
y_valn[range(5000),y_val.astype(int)] = 1


if args.pretrain == True:
    weights = load_weights(args.state, args.save_dir)
    shift = int(len(weights)/2)
    param = {}
    for i in range(args.num_hidden+1):
        param['W_{}'.format(i+1)] = weights[i]
        param['b_{}'.format(i+1)] = weights[i+shift]

    mlp = build_network(lr = args.lr ,gamma=args.momentum ,n_hidden=args.num_hidden, sz = args.sizes, weights = param,loss =args.loss,activation=args.activation,opt =args.opt,batch_sz=args.batch_size,epoch=args.epochs,lambd =  args.lambd,anneal=args.anneal,save_dir=args.save_dir ,input_dim=args.input_dim)
    if args.testing == False:
        mlp.train(X_train,y_hat,X_val,y_valn,args.state)
else:
    mlp = build_network(lr = args.lr,gamma=args.momentum ,n_hidden=args.num_hidden, sz = args.sizes,loss =args.loss,activation=args.activation,opt =args.opt,batch_sz=args.batch_size,epoch=args.epochs,lambd = args.lambd, anneal=args.anneal,save_dir=args.save_dir,input_dim=args.input_dim)
    mlp.train(X_train,y_hat,X_val,y_valn)

if args.testing == True:
    pred = mlp.forward_prop(np.transpose(X_test))[1]['layer_{}'.format(mlp.n_hidden+1)].argmax(axis=0)
    pd.DataFrame([test.index,pred], index = ['id','label']).T.to_csv(args.expt_dir+'predictions_{}.csv'.format(args.state),index=False)

train = open(args.expt_dir+'logs_train.txt','w')
val = open(args.expt_dir+'logs_val.txt','w')
for i in range(len(mlp.train_log)):
    train.write("Epoch %d, Step %d, Loss: %.5f, Error: %.2f, lr: %f\n" %tuple(mlp.train_log[i])) 
    val.write("Epoch %d, Step %d, Loss: %.5f, Error: %.2f, lr: %f\n" %tuple(mlp.val_log[i]))

train.close()
val.close()


    
