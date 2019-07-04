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

def create_model(X,is_train,initializer = tf.contrib.layers.xavier_initializer()):
    conv1 = tf.layers.conv2d(inputs = X, filters = 64, kernel_size = [5,5], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    bn1 = tf.layers.batch_normalization(inputs = conv1, training = is_train)
    act1 = tf.nn.relu(bn1)
    dp3 = tf.layers.dropout(act1, rate = 0.2, training = is_train )
    
    conv2 = tf.layers.conv2d(inputs = dp3, filters = 64, kernel_size = [5,5], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    bn2 = tf.layers.batch_normalization(inputs = conv2, training = is_train)
    act2 = tf.nn.relu(bn2)
    
    
    pool1 = tf.layers.max_pooling2d(inputs = act2, pool_size = [2,2], strides = 2)
    dp4 = tf.layers.dropout(pool1, rate = 0.2, training = is_train )
    

    conv3 = tf.layers.conv2d(inputs = dp4, filters = 128, kernel_size = [3,3], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    bn3 = tf.layers.batch_normalization(inputs = conv3, training = is_train)
    act3 = tf.nn.relu(bn3)
    dp5 = tf.layers.dropout(act3, rate = 0.2, training = is_train )
    
    conv4_1 = tf.layers.conv2d(inputs = dp5, filters = 64, kernel_size = [3,3], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    conv4_2 = tf.layers.conv2d(inputs = dp5, filters = 64, kernel_size = [5,5], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    conv4 = tf.concat([conv4_1,conv4_2],-1)
    bn4 = tf.layers.batch_normalization(inputs = conv4, training = is_train)
    act4 = tf.nn.relu(bn4)
    
    pool2 = tf.layers.max_pooling2d(inputs = act4, pool_size = [2,2], strides = 2)
    dp6 = tf.layers.dropout(pool2, rate = 0.25, training = is_train )
   
    conv5_1 = tf.layers.conv2d(inputs = dp6, filters = 64, kernel_size = [3,3], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    conv5_2 = tf.layers.conv2d(inputs = dp6, filters = 64, kernel_size = [5,5], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    conv5 = tf.concat([conv5_1,conv5_2],-1)
    bn5 = tf.layers.batch_normalization(inputs = conv5, training = is_train)
    act5 = tf.nn.relu(bn5)
    dp7 = tf.layers.dropout(act5, rate = 0.25, training = is_train )
    
    conv6_1 = tf.layers.conv2d(inputs = dp7, filters = 64, kernel_size = [3,3], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    conv6_2 = tf.layers.conv2d(inputs = dp7, filters = 64, kernel_size = [5,5], padding='same',activation = None, strides = [1,1],kernel_initializer = initializer)
    conv6 = tf.concat([conv6_1,conv6_2],-1)
    bn6 = tf.layers.batch_normalization(inputs = conv6, training = is_train)
    act6 = tf.nn.relu(bn6)
    
    pool3 = tf.layers.max_pooling2d(inputs = act6, pool_size = [2,2], strides = 2)
    flatten = tf.reshape(pool3, [-1, 8*8*128])
    dp1 = tf.layers.dropout(flatten, rate = 0.35, training = is_train )
    
    dense1 = tf.layers.dense(inputs = dp1, units = 512, activation = None)
    bn7 = tf.layers.batch_normalization(inputs = dense1, training = is_train)
    act7 = tf.nn.relu(bn7)
    dp2 = tf.layers.dropout(bn7, rate = 0.3, training = is_train )
    
    dense2 = tf.layers.dense(inputs = dp2, units = 20, activation = None)
    bn8 = tf.layers.batch_normalization(inputs = dense2, training = is_train)
    logits = tf.nn.softmax(logits = bn8,axis = 1)
    return logits


X = tf.placeholder(tf.float32, shape = (None,64,64,3))
y = tf.placeholder(tf.float32, shape = (None,20))
is_train = tf.placeholder(tf.bool)

logits = create_model(X,is_train,initializer)
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
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss,global_step=global_step)


val = pd.read_csv('../data/valid.csv',header = 0,index_col = 0).values
X_val = val[:,:-1]/255
val_size = X_val.shape[0]
X_val = X_val.reshape((-1,64,64,3))
y_val = val[:,-1].reshape((-1))
y_val_ohot = np.zeros((val_size,20))
y_val_ohot[range(val_size),y_val.astype(int)] = 1

img = 800
import copy
num_pix = 1000
print (X_val[0,[2,3],[2,4],:])
with tf.Session() as sess:
    saver.restore(sess,'./temp/model.ckpt')
    
    [preds,prob] = sess.run([predictions,logits], feed_dict = {X : X_val,is_train : False})
    tensor = tf.get_default_graph().get_tensor_by_name("Softmax:0")
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(X_val[img])
    ax[0].set_title('True class: {}, Pred class: {}\n( prob:'.format(y_val[img],preds[img]) + "%.2f)" %prob[img,preds[img]])
    grad = tf.gradients(tensor[:,0],X)
    new_X_val = copy.deepcopy(X_val[img]).reshape((-1,64,64,3))
    for i in range(5):
      new_X_val = new_X_val + 0.5 * sess.run(grad, feed_dict={X:new_X_val,is_train:False})[0]
    
    [preds,prob] = sess.run([predictions,logits], feed_dict = {X : new_X_val,is_train : False})  
    print(preds,prob)
    mini = np.min(new_X_val)
    maxi = np.max(new_X_val)
    img1 = ((new_X_val-mini)/(maxi-mini)).squeeze()
    
    ax[1].imshow(img1)
    ax[1].set_title('True class: {}, Pred class: {}\n( prob:'.format(y_val[img],preds) + "%.2f)" %prob[0,preds])
    
    new_X_val = X_val[img] - new_X_val.squeeze()
    mini = np.min(new_X_val)
    maxi = np.max(new_X_val)
    img2 = (new_X_val-mini)/(maxi-mini)
    
    ax[2].imshow(img2)
    ax[2].set_title('Difference between images')
    
    plt.savefig('img_{}.png'.format(img))
    
    fig.tight_layout()
      