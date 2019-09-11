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
from tqdm import tqdm

print(tf.__version__)
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        #print(word)
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
def evaluate_acc(model,x_test,test_labels,BATCH_SIZE):
    BUFFER_SIZE = len(x_test)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, test_labels)).shuffle(BUFFER_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
    acc = []
    attack_success = 0
    total = 0
    avg_change = 0
    for (batch, (inp, targ)) in enumerate(test_dataset):
        out = model(inp,False)
        right = 0
        for idx,each in enumerate(out):
            real = targ[idx]
    #         print(inp[idx])
    #         print("actual label:",np.argmax(real),"pred label:",np.argmax(each))
            pred = 0
            real = real.numpy()
            if tf.sigmoid(each).numpy()[0] < 0.5:
                pred = 0
            else:
                pred = 1
            if pred == real:
                right += 1
    #         if np.argmax(each) == np.argmax(real):
    #             right += 1
        batch_acc = right/BATCH_SIZE
        acc.append(batch_acc)
        if batch == 100:
            break
    print("model accuracy: ",np.mean(acc))
    return np.mean(acc)
def hotflip_attack(x,gradients,mask,model,verbose=True):
    embedding = model.embedding
    x_embedding = embedding(x)
    x_grad = gradients.values
    embedding_matrix = embedding.weights[0]
    # grad[x_t]^T * (x_t - x_p) is maximized 
    # The "Hotflip" attack described clearly in https://arxiv.org/abs/1903.06620, 
    # their code I used is here https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py
    # embedding = [20000,128]  ~ to get x_t
    # x_embedding = [1,80,128] ~ x_t
    # x_grad = [80,128]        ~ grad[x_t]
    # 1 use einsum to compute the grad[x_t]^T * x_p. 
    x_grad = tf.reshape(x_grad,[1,x_grad.shape[0],x_grad.shape[1]])
    product1 = tf.einsum("bij,kj->bik", x_grad, embedding_matrix)
#     print(product1.shape)
    # 2 compute grad[x_t]^T * x_t
    product2 = tf.einsum("bij,bij->bi", x_grad, x_embedding)
#     print(product2.shape)
    product2 = tf.expand_dims(product2,-1)
#     print(product2.shape)
    # 3 we want the maximum of this difference to increase the loss
    neg_dir_dot_grad = -1*(product2 - product1)
#     print(neg_dir_dot_grad.shape)
#     print(product2[0])
#     print("first column",product1[0,:,0])
#     print(neg_dir_dot_grad[0,:,0])
#     print("second column",product1[0,:,1])
#     print(neg_dir_dot_grad[0,:,1])
    
    # 4 maybe normalize it
    # 5 pick the max
    best = tf.argmax(neg_dir_dot_grad,2)
    # Now I have a tensor of size [1,80]. This means for each word i
    # I have a best perturbation generated from the vocab.
    # 6 this step chooses which particular word to attack. The heuristic
    # used here is to simply choose the word with max gradient norm.
    num_of_words = gradients.values.shape[0]
    grads = gradients.values
    grads_norm = tf.einsum("ij,ij->i",grads,grads)
#     print(grads_norm)
    grads_norm = np.einsum("j,j->j",mask,grads_norm)
    which_token = np.argmax(grads_norm)
    if verbose:
        print("perturbed index:",which_token)
        print("perturbed token index:",best.numpy()[0][which_token])
    x_p = best.numpy()[0][which_token]
    new_x = np.array(x)[0]
    new_x[which_token] = x_p
    return best,new_x,which_token


def get_pred(m,x,y,verbose=True):
    output = m(x, False)
    pred = 0
    conf = tf.sigmoid(output).numpy()[0]
    if conf[0] < 0.5:
        pred = 0
    else:
        pred = 1
    if verbose:
        print("\nreal:",y,"pred:",pred, "conf", conf[0],"\n")
    return output,y==pred
    
def attack(model,x,y,decode_review,loss_function,max_perturbed=5,verbose=True):
    success = 0
    if verbose:
        print("\nInitial sentence:")
        print(decode_review(x))
    x =x.reshape((1,80))
    ay = tf.reshape(y, [-1])
    with tf.GradientTape() as tape:
        output,same = get_pred(model,x,y,verbose=True)
    count = 0
    org =x
    mask = np.ones_like(x[0])
    if same:
        while same:
            if verbose:
                print("attack # %d ---------------------------- "%(count))
            with tf.GradientTape() as tape:
                x =x.reshape((1,80))
                output,same = get_pred(model,x,y)
                loss = loss_function(ay, output)
                variables = model.trainable_variables
                gradients = tape.gradient(loss, [variables[0]])
                candidates,x2,idx = hotflip_attack(x, gradients[0],mask,model,verbose=verbose)
            mask[idx] = 0
            if verbose:
                print("perturbed sentences:")
                print(decode_review(x2))
            x = x2
            x =x.reshape((1,80))
            output,same = get_pred(model,x,y)
#             print(decode_review(candidates.numpy()[0]))
    #         print(candidates.numpy()[0])
            count += 1
            if count == max_perturbed:
                success = 0
                break
        else:
            print("attack succeed!")
            success = 1
            org = x
    else:
        print("wrong prediction!")
        success = 2
    return success,org
def attack_dataset(m,x_train,train_labels,decode_review,loss_function):
    total = 0
    success = 0
    wrong_pred = 0 
    right_pred = 0
    perturbed_idx = []
    perturbed_sen = []
    for idx, x in tqdm(enumerate(x_train)):
        print("data #%d"%(idx))
        total += 1
        newx = np.array(x)
        y = np.array(train_labels[idx])
        o,sentence = attack(m,newx,y,decode_review,loss_function)
        if o == 0:
            right_pred += 1
        if o == 1:
            success += 1
            right_pred += 1
            perturbed_idx.append(idx)
            perturbed_sen.append(sentence)
        if o == 2:
            wrong_pred += 1
    print("success attack rate = %f"%(success/(total-wrong_pred)))
    print("prediction score = %f"%(right_pred/total))  
    return perturbed_idx,perturbed_sen
def batch_attack(m,inp,targ,decode_review,loss_function):
#     print(inp.shape)
    adv_output = np.zeros((inp.shape[0],inp.shape[1]))
    for idx,(data,label) in enumerate(zip(inp,targ)):
        o,sentence = attack(m,data.numpy(),label.numpy(),decode_review,loss_function,verbose=False)
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
