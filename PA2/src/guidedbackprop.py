import numpy as np 
np.random.seed(0) 
import pandas as pd
import tensorflow as tf
tf.set_random_seed(1)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import scipy
import scipy.ndimage
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

def create_model(X,is_train):
    conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = [5,5], padding='same',activation = tf.nn.relu, strides = [1,1])
    conv2 = tf.layers.conv2d(inputs = conv1, filters = 32, kernel_size = [5,5], padding='same',activation = tf.nn.relu, strides = [1,1])
    pool1 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)

    conv3 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = [3,3], padding='same',activation = tf.nn.relu, strides = [1,1])
    conv4 = tf.layers.conv2d(inputs = conv3, filters = 64, kernel_size = [3,3], padding='same',activation = tf.nn.relu, strides = [1,1])
    pool2 = tf.layers.max_pooling2d(inputs = conv4, pool_size = [2,2], strides = 2)

    conv5 = tf.layers.conv2d(inputs = pool2, filters = 64, kernel_size = [3,3], padding='same',activation = tf.nn.relu, strides = [1,1])
    conv6 = tf.layers.conv2d(inputs = conv5, filters = 128, kernel_size = [3,3], padding='valid',activation = tf.nn.relu, strides = [1,1])
    pool3 = tf.layers.max_pooling2d(inputs = conv6, pool_size = [2,2], strides = 2)
    flatten = tf.reshape(pool3, [-1, 6272])

    dense1 = tf.layers.dense(inputs = flatten, units = 256, activation = tf.nn.relu)
    dense2 = tf.layers.dense(inputs = dense1, units = 20, activation = None)
    bn1 = tf.layers.batch_normalization(inputs = dense2, training = is_train)
    logits = tf.nn.softmax(logits = bn1,axis = 1)
    return logits

X = tf.placeholder(tf.float32, shape = (None,64,64,3))
y = tf.placeholder(tf.float32, shape = (None,20))
is_train = tf.placeholder(tf.bool)

logits = create_model(X,is_train)
predictions = tf.argmax(logits, axis=1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits))
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.5
learning_rate = tf.train.inverse_time_decay(learning_rate, global_step,
decay_steps, decay_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                            
with tf.control_dependencies(update_ops):                                                            
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss,global_step=global_step)

epochs = 100
batch_size = 32
patience = 10

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
saver = tf.train.Saver()


train = pd.read_csv('../data/train.csv',header = 0,index_col = 0).values
X_train = train[:,:-1]/255
train_size = X_train.shape[0]
X_train = X_train.reshape((-1,64,64,3))
ind = np.random.permutation(X_train.shape[0])
X_train = X_train[ind]



import numpy as np
saver = tf.train.Saver()
with tf.Session() as sess: 
    saver.restore(sess,'./save_dir/model.ckpt')
    tensor = tf.get_default_graph().get_tensor_by_name("conv2d_5/Conv2D:0")
    rand_ind = np.random.randint(0,8*8*128, size = 10)
    tensors,indices = [],[]
    for ind in rand_ind:
        k = ind%128
        j = 3+ind//128%8
        i = 3+ind//128//8%8
        tensors.append(tensor[:,i,j,k])
        indices.append([i,j,k])
    
    out = sess.run(tensors, feed_dict = {X : X_train[:1000]})
    imgs = []
    for i in range(len(out)):
        imgs.append(np.argsort(out[i])[::-1][:10])
    

@ops.RegisterGradient("gRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

import matplotlib as mpl
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape = (1,64,64,3))
is_train = tf.placeholder(tf.bool)
with tf.Session() as sess:
    g = sess.graph
    with g.gradient_override_map({'Relu': 'gRelu'}):
        logits = create_model(X,is_train)
        saver = tf.train.Saver()
        saver.restore(sess,'./save_dir/model.ckpt')
        tensor = tf.get_default_graph().get_tensor_by_name("conv2d_5/Conv2D:0")
        for i in range(10):
            fig, ax = plt.subplots(nrows=1, ncols=2)
            for j,img in enumerate(imgs[i]):
                w,h,c = indices[i]
                grad = tf.gradients(tensor[:,w,h,c],X)
                ax[0].imshow(X_train[img])
                guided = sess.run(grad,feed_dict={X:X_train[img].reshape(1,64,64,3)})[0].squeeze()
                mini = np.amin(guided,keepdims=True)
                maxi = np.amax(guided,keepdims=True)
                guided = (guided-mini)/(maxi-mini)
                ax[1].imshow(guided,cmap=mpl.cm.gray)
                plt.savefig('./guided/neuron_{}_{}.png'.format(i,j))