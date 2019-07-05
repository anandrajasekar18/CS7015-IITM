import numpy as np 
np.random.seed(0) 
import pandas as pd
import tensorflow as tf
tf.set_random_seed(1)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import scipy
import argparse


parser = argparse.ArgumentParser(description='Training a RNN')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for gradient descent')
parser.add_argument('--batch_size', type=int, help='batch size (1 or multiples of 5)')
parser.add_argument('--init', type=int, help='batch size (1 for Xavier, 2 for He)')
parser.add_argument('--save_dir', default='./models',help='save dir for model')
parser.add_argument('--epochs', type=int, help='Number of epochs to be run')
parser.add_argument('--train', default='../data',help='path to training set')
parser.add_argument('--val', default='../data',help='path to validation set')
parser.add_argument('--prob', default='../data',help='prob')



args = parser.parse_args()

if args.init == 1:
    initializer = tf.contrib.layers.xavier_initializer()
else :
    initializer = tf.initializers.he_uniform()

train = pd.read_csv(args.train,header=0, index_col = 0)
val = pd.read_csv(args.val,header=0, index_col = 0)


X_train_raw=train['ENG'].values
y_train_ohot_raw=train['HIN'].values

eng_list=['<UNK>','<PAD>']
hin_list=['<GO>','<UNK>','<PAD>']

eng_corpus=train['ENG'].values
hin_corpus=train['HIN'].values

for i in eng_corpus:
    i=i.replace(" ", "")
    for j in i:
        if j not in eng_list:
            eng_list.append(j)
            
for i in hin_corpus:
    i=i.replace(" ", "")
    for j in i:
        if j not in hin_list:
            hin_list.append(j)


eng_vocab_size = len(eng_list)
max_time_eng = 61
max_time_hin = 62
hin_vocab_size = len(hin_list)



def eng_preprocess(eng, eng_letters):
    
    eng_words=[]
    for i in eng:
        i=i.replace(" ", "")
        eng_words.append(i)

    max_length=max_time_eng
    eng_words=np.asarray(eng_words)
    eng_letters=np.asarray(eng_letters)

    dictOfEng = { i : eng_letters[i] for i in range(0, len(eng_letters) ) }
    inv_dictOfEng={v: k for k, v in dictOfEng.items()}

    data_matrix=np.zeros((max_length,eng_letters.shape[0]))
    index=0
    for word in eng_words:
        word_copy=word
        word_copy=np.asarray(list(word_copy))
        word_int=np.zeros(word_copy.shape[0],dtype=np.int8)
    
        j=0
        for i in word_copy:
            if i not in eng_letters:
                i=inv_dictOfEng['<UNK>']
            else:
                i=inv_dictOfEng[i]
            word_int[j]=i
            j=j+1
        
        pad_length=max_length-word_int.shape[0]
        word_padded=np.pad(word_int,(0,pad_length), 'constant', constant_values=(inv_dictOfEng['<PAD>']))
        word_one_hot = np.zeros((max_length,eng_letters.shape[0]))
        word_one_hot[np.arange(max_length), word_padded] = 1
    
        if index==0:
            arrays=[data_matrix,word_one_hot]
            data_matrix=np.stack(arrays,axis=0)
            index=index+1
        else:
            data_matrix=np.concatenate((data_matrix,[word_one_hot]),axis=0)
        
    data_matrix=np.delete(data_matrix,0,0)

    return data_matrix

def hin_preprocess(hin,hin_letters):

    hin_words=[]
    for i in hin:
        i=i.replace(" ", "")
        hin_words.append(i)

    max_length=max_time_hin+2
    hin_words=np.asarray(hin_words)
    hin_letters=np.asarray(hin_letters)

    dictOfHin = { i : hin_letters[i] for i in range(0, len(hin_letters) ) }
    inv_dictOfHin={v: k for k, v in dictOfHin.items()}

    data_matrix=np.zeros((max_length,hin_letters.shape[0]))
    index=0
    for word in hin_words:
        word_copy=word
        word_copy=np.asarray(list(word_copy))
        word_int=np.zeros((word_copy.shape[0]+2),dtype=np.int8)
    
        j=1
        word_int[0]=inv_dictOfHin['<GO>']
        for i in word_copy:
            if i not in hin_letters:
                i=inv_dictOfHin['<UNK>']
            else:
                i=inv_dictOfHin[i]
            word_int[j]=i
            j=j+1
        word_int[j]=inv_dictOfHin['<PAD>']
        
        pad_length=max_length-word_int.shape[0]
        word_padded=np.pad(word_int,(0,pad_length), 'constant', constant_values=(inv_dictOfHin['<PAD>']))
        word_one_hot = np.zeros((max_length,hin_letters.shape[0]))
        word_one_hot[np.arange(max_length), word_padded] = 1
    
        if index==0:
            arrays=[data_matrix,word_one_hot]
            data_matrix=np.stack(arrays,axis=0)
            index=index+1
        else:
            data_matrix=np.concatenate((data_matrix,[word_one_hot]),axis=0)
        
    data_matrix=np.delete(data_matrix,0,0)

    return data_matrix

X_train=eng_preprocess(X_train_raw,eng_list)
y_train_ohot=hin_preprocess(y_train_ohot_raw,hin_list)

eng = set(' '.join(list(train.ENG)))
hin = set(' '.join(list(train.HIN)))
eng.add(('un','pa'))
hin.add(('pa'))

# def encoder(X,prob, batch_size, englen):
#   inp_embedding = tf.layers.Dense(256, activation = tf.nn.tanh)(X)
#   lstm_cell_encoder = tf.nn.rnn_cell.LSTMCell(num_units=512)
#   state = lstm_cell_encoder.zero_state(batch_size, dtype = tf.float32)
#   lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_encoder, output_keep_prob= prob)
  
#   outputs, state = tf.nn.dynamic_rnn(lstm_dropout, inp_embedding, initial_state =state, sequence_length = englen)

#   return outputs,state

def biencoder(X,prob, batch_size, englen):
  embedding_encoder = tf.get_variable("embedding_encoder", [len(eng_list), 256], initializer=initializer)
  inp_embedding = tf.nn.embedding_lookup(embedding_encoder, X)
  
  
  lstm_cell_encoder1 = tf.nn.rnn_cell.LSTMCell(num_units=512, activation = tf.nn.tanh)
  state1 = lstm_cell_encoder1.zero_state(batch_size, dtype = tf.float32)
  lstm_dropout1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_encoder1, output_keep_prob= prob)
  
  lstm_cell_encoder2 = tf.nn.rnn_cell.LSTMCell(num_units=512, activation = tf.nn.tanh)
  state2 = lstm_cell_encoder2.zero_state(batch_size, dtype = tf.float32)
  lstm_dropout2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_encoder2, output_keep_prob= prob)
  
  (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(lstm_dropout1, lstm_dropout2, inp_embedding,
                                                                             initial_state_fw=state1,
                                                                             initial_state_bw=state2, sequence_length = englen)
  outputs = tf.concat([output_fw, output_bw], axis=2)
  

  
  state = tf.nn.rnn_cell.LSTMStateTuple(c= last_state[0][0] , h= last_state[1][0]), tf.nn.rnn_cell.LSTMStateTuple(c=last_state[0][1], h= last_state[1][1])
  

  return outputs,state
#   return outputs,tf.nn.rnn_cell.LSTMStateTuple(c=c, h= h)

def decoder(enc_output, enc_state, target, prob, batch_size, output_size, is_train):
  
#   target = tf.argmax(target,axis=2)
  embedding_decoder = tf.get_variable("embedding_decoder", [len(hin_list), 256], initializer=initializer)

  
  lstm_cell_decoder1 = tf.nn.rnn_cell.LSTMCell(num_units=512, activation = tf.nn.tanh)
  lstm_cell_decoder2 = tf.nn.rnn_cell.LSTMCell(num_units=512, activation = tf.nn.tanh)
  
  lstm_dropout1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_decoder1, output_keep_prob= prob)
  lstm_dropout2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_decoder2, output_keep_prob= prob)
  
  W = tf.get_variable("W", shape=[1,1024, 256], initializer=initializer)
  U = tf.get_variable("U", shape=[512, 256], initializer=initializer)
  V = tf.get_variable("V", shape=[1,256,1], initializer=initializer)

  W_out = tf.get_variable("W_out", shape=[512, 86], initializer=initializer)
  b_out = tf.get_variable("b_out", shape=[1,86], initializer=initializer)
  
  
  dec_state1 = enc_state[0]
  dec_state2 = enc_state[0]
  embed_out = tf.nn.embedding_lookup(embedding_decoder, target[:,0])
  
  w_att = tf.matmul(outputs,tf.tile(W,[batch_size,1,1]))
  predictions = []
#   with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE) as scope :
  for i in range(max_time_hin+1):
      
      att_weights = tf.nn.softmax(tf.matmul(tf.nn.tanh(w_att + tf.reshape(tf.matmul(dec_state1[0],U),(-1,1,256))),tf.tile(V,[batch_size,1,1])),axis=1)
      weighted_inp = tf.matmul(tf.transpose(att_weights, perm = [0,2,1]),outputs)[:,0,:]
      output1, dec_state1 = lstm_dropout1(tf.concat([weighted_inp,embed_out],axis=-1),dec_state1)
      output2, dec_state2 = lstm_dropout2(output1,dec_state2)
#       preds = output_layer(output2)   
      preds = tf.matmul(output2,W_out)+b_out
      next_inp = tf.argmax(preds, axis = 1)
      predictions.append(tf.reshape(preds,(-1,1,hin_vocab_size)))

#       embed_out = tf.cond(is_train,lambda: tf.nn.embedding_lookup(embedding_decoder,target[:,i+1]), lambda: tf.nn.embedding_lookup(embedding_decoder,target[:,i+1]))
      embed_out = tf.cond(is_train,lambda: tf.nn.embedding_lookup(embedding_decoder,target[:,i+1]), lambda: tf.nn.embedding_lookup(embedding_decoder,next_inp))
  #     print(embed_out.shape)
  return tf.concat(predictions,axis=1)

# def w_decoder(enc_output, enc_state, target, prob, batch_size, output_size, is_train):
#   out_embedding = tf.layers.Dense(256, activation = tf.nn.tanh)
#   output_layer = tf.layers.Dense(output_size, activation = tf.nn.tanh)
  
#   lstm_cell_decoder1 = tf.nn.rnn_cell.LSTMCell(num_units=512, activation = tf.nn.tanh)
#   lstm_cell_decoder2 = tf.nn.rnn_cell.LSTMCell(num_units=512, activation = tf.nn.tanh)
  
#   lstm_dropout1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_decoder1, output_keep_prob= prob)
#   lstm_dropout2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_decoder2, output_keep_prob= prob)
  


  
#   dec_state1 = enc_state
#   dec_state2 = enc_state
#   embed_out = out_embedding(target[:,0,:])
#   print (embed_out.shape)
#   predictions = []
#   for i in range(max_time_hin+1):
#     output1, dec_state1 = lstm_dropout1(tf.concat([outputs[:,-1,:],embed_out],axis=-1),dec_state1)
#     output2, dec_state2 = lstm_dropout2(output1,dec_state2)
#     preds = output_layer(output2)    
#     next_inp = tf.argmax(preds, axis = 1)
#     predictions.append(tf.reshape(preds,(-1,1,hin_vocab_size)))
#     embed_out = tf.cond(is_train,lambda: out_embedding(target[:,i+1,:]), lambda: out_embedding(tf.one_hot(next_inp,depth = hin_vocab_size)))
# #     print(embed_out.shape)
#   return tf.concat(predictions,axis=1)

def create_mini_batches(X,y,englen, hinlen, batch_size):

    size = X.shape[0]
    indices = np.random.permutation(size)
    X = X[indices]
    y = y[indices]
    englen = englen[indices]
    hinlen = hinlen[indices]
    num_batches = int(size/batch_size)


    batches = []
    for i in range(num_batches):
        X_mini = X[i * batch_size:(i + 1)*batch_size]
        y_mini = y[i * batch_size:(i + 1)*batch_size]
        englen_mini = englen[i * batch_size:(i + 1)*batch_size]
        hinlen_mini = hinlen[i * batch_size:(i + 1)*batch_size]
        batches.append((X_mini, y_mini,englen_mini, hinlen_mini)) 
    if (num_batches != size/batch_size):
        X_mini = X[num_batches*batch_size:]
        y_mini = y[num_batches*batch_size:]
        englen_mini = englen[num_batches*batch_size:]
        hinlen_mini = hinlen[num_batches*batch_size:]
        
        batches.append((X_mini,y_mini,englen_mini, hinlen_mini))
    return batches


X_val_eng=eng_preprocess(val['ENG'].values,eng_list)
X_val_hin=hin_preprocess(val['HIN'].values,hin_list)

#X_test_eng = eng_preprocess(test['ENG'].values,eng_list)
#X_test_hin = hin_preprocess(val['HIN'].values,hin_list)[:,0,:].reshape((-1,1,86))


max_time_eng = 61
max_time_hin = 62

tf.reset_default_graph()

eng_placeholder = tf.placeholder(tf.int32, shape = (None,max_time_eng))
hin_placeholder = tf.placeholder(tf.int32, shape = (None,max_time_hin+2))
# go_placeholder = tf.placeholder(tf.float32, shape = (None,1,hin_vocab_size))
eng_len_placeholder = tf.placeholder(tf.int32, shape = (None))
hin_len_placeholder = tf.placeholder(tf.int32, shape = (None))
output_prob = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32,[])
is_train = tf.placeholder(tf.bool)

outputs,state = biencoder(eng_placeholder,output_prob,batch_size, eng_len_placeholder)
# print (outputs.shape)


predictions = decoder(outputs,state, hin_placeholder, output_prob, batch_size, len(hin_list) , is_train)


ce = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(hin_placeholder[:,1:],depth = len(hin_list),dtype=tf.int32), logits = predictions)
seq_mask = tf.sequence_mask(hin_len_placeholder, maxlen = max_time_hin + 1, dtype = tf.float32)
loss = tf.reduce_mean(ce*seq_mask)



learning_rate = args.lr


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                            
with tf.control_dependencies(update_ops):                                                            
  optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

epochs = args.epochs
batchsize = args.batch_size
patience = 5

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
saver = tf.train.Saver()

# print(X_val_eng.shape, X_val_hin.shape)



train_eng_len = train['ENG'].map(lambda x : len(x.split()) ).values
val_eng_len = val['ENG'].map(lambda x : len(x.split()) ).values
train_hin_len = train['HIN'].map(lambda x : len(x.split()) ).values + 1
val_hin_len = val['HIN'].map(lambda x : len(x.split()) ).values + 1


# X_val_eng = X_val_eng[:,:15,:]
# X_val_hin = X_val_hin[:,:18,:]
# X_train_trunc = X_train[train_eng_len<=15,:15,:]
# y_train_ohot_trunc = y_train_ohot[train_eng_len<=16,:18,:]
# train_eng_len_trunc = train_eng_len[train_eng_len<=15]





# print (X_train_trunc.shape,X_val_hin.shape)

with tf.Session() as sess:
    sess.run([init,init_l])
    prev_val_loss= sess.run(loss, feed_dict = {eng_placeholder : np.argmax(X_val_eng,axis=2), hin_placeholder : np.argmax(X_val_hin,axis=2), output_prob : 1, batch_size : X_val_eng.shape[0], is_train : False, eng_len_placeholder: val_eng_len, hin_len_placeholder: val_hin_len})
    print (prev_val_loss)
    wait_time,i = 0,0
    while wait_time < patience and i < epochs:
        tot_train_loss = 0
        tot_train_acc = 0
        batches = create_mini_batches(X_train,y_train_ohot,train_eng_len, train_hin_len, batchsize)
        for k,mini_batch in enumerate(batches):
            X_train_eng, X_train_hin, X_eng_len, X_hin_len = mini_batch
            
#             print (np.argmax(X_train_eng,axis=2)[:5,:15], '\n', np.argmax(X_train_hin,axis=2)[:5,:15], '\n',X_eng_len[:5], '\n',X_hin_len[:5])
#             X_train_eng=eng_preprocess(X_mini,eng_list)
#             X_train_hin=hin_preprocess(y_mini,hin_list)
#             print (X_train_hin.shape,X_train_eng.shape)
#             break
            _,train_loss,train_preds = sess.run([optimizer,loss,predictions], feed_dict = {eng_placeholder : np.argmax(X_train_eng,axis=2), hin_placeholder : np.argmax(X_train_hin,axis=2), output_prob : args.prob, 
                                                                                           batch_size : X_train_eng.shape[0], is_train : True, eng_len_placeholder: X_eng_len, hin_len_placeholder: X_hin_len})
            tot_train_loss+= 1/(k+1)*(train_loss - tot_train_loss)
            tot_train_acc+= 1/(k+1)*(np.mean([np.array_equal(np.argmax(X_train_hin[i,1:X_hin_len[i]+1,:],axis=1),np.argmax(train_preds[i,:X_hin_len[i],:],axis=1)) for i in range(train_preds.shape[0])])- tot_train_acc)
#             print(train_preds.shape)
        new_val_loss, val_preds = sess.run([loss,predictions], feed_dict = {eng_placeholder : np.argmax(X_val_eng,axis=2), hin_placeholder : np.argmax(X_val_hin,axis=2), output_prob : 1, batch_size : X_val_eng.shape[0], is_train : False,eng_len_placeholder: val_eng_len, hin_len_placeholder: val_hin_len})
        if (new_val_loss < prev_val_loss):
            wait_time = 0
            prev_val_loss = new_val_loss
            saver.save(sess,'./temp/model2.ckpt')
        else:
            wait_time+=1
        i+=1
        
        print ("Epoch: {}".format(i))
        print ("Training loss:",tot_train_loss,"Validation loss:",new_val_loss)
        print ("Training accuracy:",tot_train_acc,"Validation accuracy:",np.mean([np.array_equal(np.argmax(X_val_hin[i,1:val_hin_len[i]+1,:],axis=1),np.argmax(val_preds[i,:val_hin_len[i],:],axis=1)) for i in range(X_val_hin.shape[0])]),'\n')
        
#         break
    saver.restore(sess,'./temp/model2.ckpt')





# ls = [1,3,5]
# b = np.array([2,1,0])
# ls[b]
# # X_hin_len =

# # np.count(train_eng)
# # print (np.argmax(X_val_hin,axis=2)[:,20])
# train

# train['ENG'].map(lambda x : len(x.split()) ).value_counts()

# val['ENG'].map(lambda x : len(x.split()) ).value_counts()

# test['ENG'].map(lambda x : len(x.split()) ).value_counts()

# val['HIN'].map(lambda x : len(x.split()) ).value_counts()

# test['ENG'].map(lambda x : len(x.split()) ).value_counts()

# val_preds.shape

# print(np.argmax(X_val_hin[0,1:val_hin_len[0]+1,:],axis=1))
# print(np.argmax(val_preds[i,:val_hin_len[0],:],axis=1))

# np.mean([np.array_equal(np.argmax(X_val_hin[i,1:val_hin_len[i]+1,:],axis=1),np.argmax(val_preds[i,:val_hin_len[i],:],axis=1)) for i in range(X_val_hin.shape[0])])

# np.mean([np.array_equal(np.argmax(X_train_hin[i,1:X_hin_len[i]+1,:],axis=1),np.argmax(train_preds[i,:X_hin_len[i],:],axis=1)) for i in range(batchsize)])

# X_test_eng = eng_preprocess(test['ENG'].values,eng_list)
# X_test_hin = np.zeros((1000,64))
# X_test_hin[:,0] = 0
# test_eng_len = test['ENG'].map(lambda x : len(x.split()) ).values

# with tf.Session() as sess:
#     saver.restore(sess,'./temp/model.ckpt')
#     val_preds = sess.run(predictions, feed_dict = {eng_placeholder : X_val_eng, hin_placeholder : X_val_hin, output_prob : 1, batch_size : X_val_eng.shape[0], is_train : False,eng_len_placeholder: val_eng_len})
#     test_preds = sess.run(predictions, feed_dict = {eng_placeholder : X_test_eng, hin_placeholder : X_test_hin, output_prob : 1, batch_size : X_test_eng.shape[0], is_train : False,eng_len_placeholder: test_eng_len})
#     print ("Best validation accuracy:",np.mean([np.array_equal(np.argmax(X_val_hin[i,1:val_hin_len[i]+1,:],axis=1),np.argmax(val_preds[i,:val_hin_len[i],:],axis=1)) for i in range(X_val_hin.shape[0])]))
    
#     preds = np.argmax(test_preds,axis=2)

#     predns = []
#     for j in range(preds.shape[0]):
#       pred = list(map(b.get,list(preds[j])))
#       ind = pred.index('<PAD>')
#       predns.append(' '.join(pred[:ind]))
    
#     pd.DataFrame([test.index.astype(int),predns], index = ['id','HIN']).T.to_csv('predictions_10.csv',index=False)

# print (predictions)
#   np.save('eng.npy',X_train)
#   np.save('hin.npy',y_train_ohot)

# def sing_decoder(enc_output, enc_state, target, prob, batch_size, output_size, is_train):
#   out_embedding = tf.layers.Dense(256, activation = tf.nn.tanh)
#   output_layer = tf.layers.Dense(output_size, activation = tf.nn.tanh)
  
#   lstm_cell_decoder1 = tf.nn.rnn_cell.LSTMCell(num_units=512, activation = tf.nn.tanh)
  
#   lstm_dropout1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_decoder1, output_keep_prob= prob)
  
#   W = tf.get_variable("W", shape=[1,512, 256], initializer=tf.contrib.layers.xavier_initializer())
#   U = tf.get_variable("U", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
#   V = tf.get_variable("V", shape=[1,256,1], initializer=tf.contrib.layers.xavier_initializer())

  
#   dec_state1 = enc_state
  
#   embed_out = out_embedding(target[:,0,:])
  
#   w_att = tf.matmul(outputs,tf.tile(W,[batch_size,1,1]))
#   predictions = []


#   for i in range(max_time_hin+1):
      
#     att_weights = tf.nn.softmax(tf.matmul(tf.nn.tanh(w_att + tf.reshape(tf.matmul(dec_state1[0],U),(-1,1,256))),tf.tile(V,[batch_size,1,1])),axis=1)
#     weighted_inp = tf.matmul(tf.transpose(att_weights, perm = [0,2,1]),outputs)[:,0,:]
#     output1, dec_state1 = lstm_dropout1(tf.concat([weighted_inp,embed_out],axis=-1),dec_state1)

#     preds = output_layer(output1)    
#     next_inp = tf.argmax(preds, axis = 1)
#     predictions.append(tf.reshape(preds,(-1,1,hin_vocab_size)))
#     embed_out = tf.cond(is_train,lambda: out_embedding(target[:,i+1,:]), lambda: out_embedding(tf.one_hot(next_inp,depth = hin_vocab_size)))

#   return tf.concat(predictions,axis=1)

with tf.Session() as sess:
    sess.run([init,init_l])
    prev_val_acc= -0.1
    wait_time,i = 0,0
    while wait_time < patience and i < epochs:
        tot_train_loss = 0
        tot_train_acc = 0
        batches = create_mini_batches(X_train,y_train_ohot,train_eng_len, train_hin_len, batchsize)
        for k,mini_batch in enumerate(batches):
            X_train_eng, X_train_hin, X_eng_len, X_hin_len = mini_batch
            
#             print (np.argmax(X_train_eng,axis=2)[:5,:15], '\n', np.argmax(X_train_hin,axis=2)[:5,:15], '\n',X_eng_len[:5], '\n',X_hin_len[:5])
#             X_train_eng=eng_preprocess(X_mini,eng_list)
#             X_train_hin=hin_preprocess(y_mini,hin_list)
#             print (X_train_hin.shape,X_train_eng.shape)
#             break
            _,train_loss,train_preds = sess.run([optimizer,loss,predictions], feed_dict = {eng_placeholder : np.argmax(X_train_eng,axis=2), hin_placeholder : np.argmax(X_train_hin,axis=2), output_prob : 0.5, 
                                                                                           batch_size : X_train_eng.shape[0], is_train : True, eng_len_placeholder: X_eng_len, hin_len_placeholder: X_hin_len})
            tot_train_loss+= 1/(k+1)*(train_loss - tot_train_loss)
            tot_train_acc+= 1/(k+1)*(np.mean([np.array_equal(np.argmax(X_train_hin[i,1:X_hin_len[i]+1,:],axis=1),np.argmax(train_preds[i,:X_hin_len[i],:],axis=1)) for i in range(train_preds.shape[0])])- tot_train_acc)
#             print(train_preds.shape)
        new_val_loss, val_preds = sess.run([loss,predictions], feed_dict = {eng_placeholder : np.argmax(X_val_eng,axis=2), hin_placeholder : np.argmax(X_val_hin,axis=2), output_prob : 1, batch_size : X_val_eng.shape[0], is_train : False,eng_len_placeholder: val_eng_len, hin_len_placeholder: val_hin_len})
        new_val_acc = np.mean([np.array_equal(np.argmax(X_val_hin[i,1:val_hin_len[i]+1,:],axis=1),np.argmax(val_preds[i,:val_hin_len[i],:],axis=1)) for i in range(X_val_hin.shape[0])])
        if (new_val_acc > prev_val_acc):
            wait_time = 0
            prev_val_acc = new_val_acc
            saver.save(sess,args.save_dir+'model3.ckpt')
        else:
            wait_time+=1
        i+=1
        
        print ("Epoch: {}".format(i))
        print ("Training loss:",tot_train_loss,"Validation loss:",new_val_loss)
        print ("Training accuracy:",tot_train_acc,"Validation accuracy:",np.mean([np.array_equal(np.argmax(X_val_hin[i,1:val_hin_len[i]+1,:],axis=1),np.argmax(val_preds[i,:val_hin_len[i],:],axis=1)) for i in range(X_val_hin.shape[0])]),'\n')
        
#         break
    # saver.restore(sess,'./temp/model3.ckpt')



