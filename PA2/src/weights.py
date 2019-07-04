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

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,'./save_dir/model.ckpt')
    with tf.variable_scope('conv2d',reuse=True) as scope_conv:
      weights = tf.get_variable('kernel')
      vals = weights.eval()




weights = np.zeros((5,5,3,32))
for i in range (32): 
    mini = np.amin(vals[...,i],keepdims=True)
    maxi = np.amax(vals[...,i],keepdims=True)
    weights[...,i] = (vals[...,i]-mini)/(maxi-mini)



for i in range(1, 33):
    ax = plt.subplot(4, 8, i)
    ax.imshow(weights[:,:,:,i-1],interpolation='nearest')
    ax.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)

# plt.show()
plt.savefig('weights.png')