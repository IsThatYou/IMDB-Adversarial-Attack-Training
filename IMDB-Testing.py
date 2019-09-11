from __future__ import absolute_import, division, print_function, unicode_literals
# from keras import backend as K
import tensorflow as tf
import keras
import numpy as np
import tensorflow.contrib.eager as tfe
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import time
from keras.initializers import Constant
from helpers import loadGloveModel,evaluate_acc,hotflip_attack,get_pred,attack,attack_dataset
print(tf.__version__)
def batch_attack(m,inp,targ):
#     print(inp.shape)
    adv_output = np.zeros((inp.shape[0],inp.shape[1]))
    for idx,(data,label) in enumerate(zip(inp,targ)):
        o,sentence = attack(m,data.numpy(),label.numpy(),verbose=False)
        sentence = np.array(sentence[0])
        adv_output[idx] = sentence
    return adv_output
def evaluate_accuracy(labels,preds):
    total = len(labels)
    counter = 0
    for l,p in zip(labels,preds):
        dummy_l = l.numpy()
        dummy_p = tf.sigmoid(p).numpy()[0]
        if dummy_p >=0.5:
            dummy_p = 1
        else:
            dummy_p = 0
#         print(dummy_l,dummy_p)
        if (dummy_l==dummy_p):
            counter+= 1
    return counter/total
def combined_loss_function(real,pred,pred_adv,a):
    pred = tf.reshape(pred, [-1])
    pred_adv = tf.reshape(pred_adv, [-1])
    loss_ = tf.losses.sigmoid_cross_entropy(real, pred) 
    loss_adv = tf.losses.sigmoid_cross_entropy(real, pred_adv) 
    print(loss,loss_adv)
    final = tf.add(loss,loss_adv)
    return final
if __name__ == '__main__':


    # 1.0 get the data
    tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
    imdb = keras.datasets.imdb
    num_features = 20000
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_features)
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    maxlen = 80
    x_train = sequence.pad_sequences(train_data, maxlen=maxlen)
    x_test = sequence.pad_sequences(test_data, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    # 1.1 get the word indices
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    # 1.2 get the glove embedding
    GLOVE_PATH = 'embeds/glove.840B.300d.txt'
    glove = loadGloveModel(GLOVE_PATH)
    glove_embedding = np.zeros(shape=(20000,300))
    for value in reverse_word_index:
        key = reverse_word_index[value]
        if key in glove:
            glove_embedding[0,:] = glove[key]
        else:
            glove_embedding[0,:] = np.random.uniform(size=(300,))

    # 2.0 setup dataset and model parameters
    BUFFER_SIZE = len(x_train)
    BATCH_SIZE = 32
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dims = 300
    lstm_units = 128

    dataset = tf.data.Dataset.from_tensor_slices((x_train, train_labels)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # 2.1 define the model
    class Classifier(tf.keras.Model):
        def __init__(self,vocab_size,embedding_dim,lstm_units):
            super(Classifier, self).__init__()
            
            self.lstm_units = lstm_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(glove_embedding))
            self.lstm = tf.keras.layers.LSTM(lstm_units,dropout=0.2, recurrent_dropout=0.2)
            self.dense = tf.keras.layers.Dense(32,activation=tf.nn.relu)
            self.pred = tf.keras.layers.Dense(1,activation=None)
            

        def call(self, x,is_training):
            x = self.embedding(x)
    #         num_samples = tf.shape(x)[0]
    #         hidden = tf.zeros((BATCH_SIZE, self.lstm_units))
    #         print(self.lstm.get_initial_state(x))
    #         print(x)
            o = self.lstm(x)     
    #         print(output.shape)
    #         o = tf.layers.dropout(o, rate=0.2, training=is_training)
    #         o = self.dense(o)
            o = self.pred(o)
    #         print(o)
    #         o = tf.nn.softmax(o)
    #         print(result)
            return o
    model = Classifier(num_features, embedding_dims, lstm_units)

    # 2.2 define loss
    optimizer = tf.train.AdamOptimizer()
    # [batch_size,class_size]
    # [barch_size,]
    def loss_function(real, pred):
    #     print("real",real,"pred", pred)
    #     weights = np.ones(real.shape[0])
    #     print(weights)
        pred = tf.reshape(pred, [-1])
        loss_ = tf.losses.sigmoid_cross_entropy(real, pred) 
        print(loss_)
        return loss_
    '''
    # 2.4 if there is already a model, load it                                                    
    root = tf.train.Checkpoint(optimizer=optimizer,
                            model=model,
                            optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint("saved/orgM/"))

    # 2.5 evaluate the model acc
    evaluate_acc(model,x_test,test_labels)
    '''
    # 3.1 load previous model
    BUFFER_SIZE = len(x_train)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dims = 300
    lstm_units = 128

    #dataset_comb = tf.data.Dataset.from_tensor_slices((x_test, test_labels)).shuffle(BUFFER_SIZE)
    #dataset_comb = dataset_comb.batch(BATCH_SIZE, drop_remainder=True)

    comb_adv_model = Classifier(num_features, embedding_dims, lstm_units)
    optimizer = tf.train.AdamOptimizer()

    root = tf.train.Checkpoint(optimizer=optimizer,
                            model=comb_adv_model,
                            optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint("saved/advM/"))
    # 3.2 test the model on original test data
    evaluate_acc(comb_adv_model,x_test,test_labels)

