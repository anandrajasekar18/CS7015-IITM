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
import argparse

parser = argparse.ArgumentParser(description='Training a CNN')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for gradient descent')
parser.add_argument('--batch_size', type=int, help='batch size (1 or multiples of 5)')
parser.add_argument('--init', type=int, help='batch size (1 for Xavier, 2 for He)')
parser.add_argument('--save_dir', default='./models',help='save dir for model')
parser.add_argument('--epochs', type=int, help='Number of epochs to be run')
parser.add_argument('--dataAugment', type=int, help='data augmentation is used or not. 1 for yes and 0 for no')
parser.add_argument('--train', default='../data',help='path to training set')
parser.add_argument('--val', default='../data',help='path to validation set')
parser.add_argument('--test', default='../data',help='path to test set')

args = parser.parse_args()





def flip_img_lr(img):
    index=img.index
    img_rs=np.reshape(img.values,(64,64,3))
    img_flip=np.fliplr(img_rs)
    img_flip=np.reshape(img_flip,-1)
    img_flip=pd.Series(img_flip,index=index)
    img.update(img_flip)
    return(img)


# def flip_img_ud(img):
#     index=img.index
#     img_rs=np.reshape(img.values,(64,64,3))
#     img_flip=np.flipud(img_rs)
#     img_flip=np.reshape(img_flip,-1)
#     img_flip=pd.Series(img_flip,index=index)
#     img.update(img_flip)
#     return(img)


def add_salt_pepper_noise(img):
    index=img.index
    img_rs=np.reshape(img.values,(64,64,3))
    
    X_imgs_copy = img_rs.copy()
    row, col, _ = X_imgs_copy.shape
    salt_vs_pepper = 0.3
    amount = 0.006
    num_salt = np.ceil(amount * X_imgs_copy.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy.size * (1.0 - salt_vs_pepper))
    
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_imgs_copy.shape]
    X_imgs_copy[coords[0], coords[1], :] = 1

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_imgs_copy.shape]
    X_imgs_copy[coords[0], coords[1], :] = 0
    
    X_imgs_copy=np.reshape(X_imgs_copy,-1)
    X_imgs_copy=pd.Series(X_imgs_copy,index=index)
    img.update(X_imgs_copy) 
    return img

def rotate_image(img):
    index=img.index
    img_rs=np.reshape(img.values,(64,64,3))
    
    angle=np.random.choice([-15,15,-30,30])
    img_rot=scipy.ndimage.rotate(img_rs, angle,reshape=False)
    
    img_rot=np.reshape(img_rot,-1)
    img_rot=pd.Series(img_rot,index=index)
    img.update(img_rot) 
    return img

if args.dataAugment == 1:
    data=pd.read_csv(args.train)

    features=data.drop(['id','label'],axis=1)/255
    features_copy1=features.copy()
    features_copy2=features.copy()
    features_copy3=features.copy()
    label=data['label']
    labels=label.copy()

    features_flip=features_copy1.apply(lambda row: flip_img_lr(row),axis=1)
    # features_ud=features_copy2.apply(lambda row: flip_img_ud(row),axis=1)
    # features_sp=features_copy2.apply(lambda row: add_salt_pepper_noise(row),axis=1)
    features_rotate=features_copy3.apply(lambda row: rotate_image(row),axis=1)
    # features_rotate=features_copy3.apply(lambda row: add_gaussian_noise(row),axis=1)
    features=features.append(features_flip, ignore_index=True)
    # features=features.append(features_sp, ignore_index=True)
    features=features.append(features_rotate, ignore_index=True)

    # label=label.append(labels,ignore_index=True)
    label=label.append(labels,ignore_index=True)
    label=label.append(labels,ignore_index=True)
    del features_copy1, features_copy2, features_copy3, labels, features_rotate, features_flip
    # del features_sp

else:
    data=pd.read_csv(args.train)
    features=data.drop(['id','label'],axis=1)/255
    label=data['label']



val = pd.read_csv(args.val,header = 0,index_col = 0).values
X_val = val[:,:-1]/255
val_size = X_val.shape[0]
X_val = X_val.reshape((-1,64,64,3))
y_val = val[:,-1].reshape((-1))
y_val_ohot = np.zeros((val_size,20))
y_val_ohot[range(val_size),y_val.astype(int)] = 1

test = pd.read_csv(args.test,header = 0,index_col = 0)
X_test = test.values/255
X_test = X_test.reshape((-1,64,64,3))


X_train = features.values
X_train = X_train.reshape((-1,64,64,3))
train_size = X_train.shape[0]

y_train = label.values
y_train_ohot = np.zeros((train_size,20))
y_train_ohot[range(train_size),y_train.astype(int)] = 1

print ("Data done")


print (X_train.shape,y_train.shape)



def create_mini_batches(X,y,batch_size):

    size = X.shape[0]
    indices = np.random.permutation(size)
    X = X[indices]
    y = y[indices]
    num_batches = int(size/batch_size)


    batches = []
    for i in range(num_batches):
        X_mini = X[i * batch_size:(i + 1)*batch_size]
        y_mini = y[i * batch_size:(i + 1)*batch_size]
        batches.append((X_mini, y_mini)) 
    if (num_batches != size/batch_size):
        X_mini = X[num_batches*batch_size:]
        y_mini = y[num_batches*batch_size:]
        batches.append((X_mini,y_mini))
    return batches

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

# def create_model(X,is_train,initializer = tf.contrib.layers.xavier_initializer()):
#     conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = [5,5], padding='same',activation = tf.nn.relu, strides = [1,1], kernel_initializer=initializer)
#     conv2 = tf.layers.conv2d(inputs = conv1, filters = 32, kernel_size = [5,5], padding='same',activation = tf.nn.relu, strides = [1,1],kernel_initializer=initializer)
#     pool1 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)

#     conv3 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = [3,3], padding='same',activation = tf.nn.relu, strides = [1,1],kernel_initializer=initializer)
#     conv4 = tf.layers.conv2d(inputs = conv3, filters = 64, kernel_size = [3,3], padding='same',activation = tf.nn.relu, strides = [1,1],kernel_initializer=initializer)
#     pool2 = tf.layers.max_pooling2d(inputs = conv4, pool_size = [2,2], strides = 2)

#     conv5 = tf.layers.conv2d(inputs = pool2, filters = 64, kernel_size = [3,3], padding='same',activation = tf.nn.relu, strides = [1,1],kernel_initializer=initializer)
#     conv6 = tf.layers.conv2d(inputs = conv5, filters = 128, kernel_size = [3,3], padding='valid',activation = tf.nn.relu, strides = [1,1], kernel_initializer=initializer)
#     pool3 = tf.layers.max_pooling2d(inputs = conv6, pool_size = [2,2], strides = 2)
#     flatten = tf.reshape(pool3, [-1, 6272])

#     dense1 = tf.layers.dense(inputs = flatten, units = 256, activation = tf.nn.relu)
#     dense2 = tf.layers.dense(inputs = dense1, units = 20, activation = None)
#     bn1 = tf.layers.batch_normalization(inputs = dense2, training = is_train)
#     logits = tf.nn.softmax(logits = bn1,axis = 1)
#     return logits

if args.init == 1:
    initializer = tf.contrib.layers.xavier_initializer()
else :
    initializer = tf.initializers.he_uniform()



X = tf.placeholder(tf.float32, shape = (None,64,64,3))
y = tf.placeholder(tf.float32, shape = (None,20))
is_train = tf.placeholder(tf.bool)

logits = create_model(X,is_train,initializer)
predictions = tf.argmax(logits, axis=1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits))
global_step = tf.Variable(0, trainable=False)
# learning_rate = args.lr
# decay_steps = 1000
# decay_rate = 0.5
# learning_rate = tf.train.inverse_time_decay(learning_rate, global_step,
# decay_steps, decay_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                            
with tf.control_dependencies(update_ops):                                                            
    optimizer = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(loss,global_step=global_step)

epochs = args.epochs
batch_size = args.batch_size
patience = 10

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run([init,init_l])
    prev_val_loss = sess.run(loss, feed_dict = {X : X_val, y: y_val_ohot, is_train : False})
    wait_time,i = 0,0
    while wait_time < patience and i < epochs:
        tot_train_loss = 0
        tot_train_acc = 0
        batches = create_mini_batches(X_train,y_train_ohot,batch_size)
        for k,mini_batch in enumerate(batches):
            X_mini, y_mini = mini_batch
            _,train_loss,train_preds = sess.run([optimizer,loss,predictions], feed_dict = {X : X_mini, y: y_mini, is_train : True})
            tot_train_loss+= 1/(k+1)*(train_loss - tot_train_loss)
            tot_train_acc+= 1/(k+1)*(accuracy_score(np.argmax(y_mini,axis=1),train_preds) - tot_train_acc)
        new_val_loss, val_preds = sess.run([loss,predictions], feed_dict = {X : X_val, y: y_val_ohot, is_train : False})
        if (new_val_loss < prev_val_loss):
            wait_time = 0
            prev_val_loss = new_val_loss
            saver.save(sess,args.save_dir+'model.ckpt')
        else:
            wait_time+=1
        i+=1
    #     print ("Epoch: {}".format(i))
    #     print ("Training loss:",tot_train_loss,"Validation loss:",new_val_loss)
    #     print ("Training accuracy:",tot_train_acc,"Validation accuracy:",accuracy_score(np.argmax(y_val_ohot,axis=1),val_preds),'\n')
    # saver.restore(sess,args.save_dir+'model.ckpt')
    # val_preds = sess.run(predictions, feed_dict = {X : X_val,is_train : False})
    # test_predictions = sess.run(predictions, feed_dict = {X : X_test, is_train : False})
    # print ("Best validation accuracy:",accuracy_score(np.argmax(y_val_ohot,axis=1),val_preds))
    # pd.DataFrame([test.index.astype(int),test_predictions.astype(int)], index = ['id','label']).T.to_csv('predictions_10.csv',index=False)





