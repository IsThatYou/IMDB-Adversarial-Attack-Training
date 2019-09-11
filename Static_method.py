# this code doesn't work individually.
# it is here to show how to do hotflip attack (targeted or untergeted) with static tensorflow graph


##### get gradient function #####
sess = K.get_session()
# print(tf.trainable_variables())
# print(tf.all_variables())
# self.model.compile(loss="categorical_crossentropy",optimizer = Adam())
y_true = tf.placeholder(tf.float32, shape=(None,59))
variables = self.model.trainable_weights
# print(variables[0])
# print(self.model.__dict__)
outputTensor = self.model.outputs[0]
cei = K.categorical_crossentropy(y_true, outputTensor)
ce = K.mean(cei)
get_gradients = K.gradients(ce, variables)

Lmax = tf.reduce_max(outputTensor,1)
Lt = outputTensor[0][6]
Loss = Lt-Lmax
get_gradients_targ = K.gradients(Loss,variables)

a = tf.placeholder(tf.float32, shape=(1,None))
        x_grad = tf.placeholder(tf.float32,shape=(None,300))
        msk = tf.placeholder(tf.float32,shape=(None,))
        def hotflip_attack():
            embed = self.model.layers[1].layers[0]
            x_embedding = embed(a)
            
            embedding_matrix = embed.weights[0]
            
            # print(embedding_matrix.shape)
            # grad[x_t]^T * (x_t - x_p) is maximized 
            # The "Hotflip" attack described clearly in https://arxiv.org/abs/1903.06620, 
            # their code I used is here https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py
            # embedding = [20000,128]  ~ to get x_t
            # x_embedding = [1,80,128] ~ x_t
            # x_grad = [80,128]        ~ grad[x_t]
            # 1 use einsum to compute the grad[x_t]^T * x_p. 
        #     x_grad2 = tf.reshape(x_grad,[1,x_grad.shape[0],x_grad.shape[1]])
            product1 = tf.einsum("ij,kj->ik", x_grad, embedding_matrix)

            # 2 compute grad[x_t]^T * x_t
            product2 = tf.einsum("ij,ij->i", x_grad, tf.reshape(x_embedding,[-1,300]))
            product2 = tf.expand_dims(product2,-1)
            # print(product2.shape)
        #     print(product2)
            # 3 we want the maximum of this difference to increase the loss
            neg_dir_dot_grad = -1*(product2-product1)
            # print(neg_dir_dot_grad.shape)
        #     print(neg_dir_dot_grad.shape)
        #     print(product2[0])
        #     print("first column",product1[0,:,0])
        #     print(neg_dir_dot_grad[0,:,0])
        #     print("second column",product1[0,:,1])
        #     print(neg_dir_dot_grad[0,:,1])

            # 4 maybe normalize it
            # 5 pick the max
            best = tf.argmax(neg_dir_dot_grad,1)
            best2 = tf.argmin(neg_dir_dot_grad,1)
            # Now I have a tensor of size [1,80]. This means for each word i
            # I have a best perturbation generated from the vocab.
            # 6 this step chooses which particular word to attack. The heuristic
            # used here is to simply choose the word with max gradient norm.
        #     num_of_words = gradients.values.shape[0]
            grads_norm = tf.einsum("ij,ij->i",x_grad,x_grad)
            # print(grads_norm.shape)
            # print(msk.shape)
            grads_norm = tf.einsum("j,j->j",msk,grads_norm)
            which_token = tf.argmax(grads_norm,0)

        #     sess.run(tf.initialize_all_variables())
        #     idx = which_token.eval(session = sess)
        #     print("perturbed index:",idx)
        #     print("perturbed token index:",best.eval(session=sess)[0][idx])

        #     x_p = best.eval(session=sess)[0][idx]
        #     new_x = np.array(x)[0]
        #     new_x[idx] = x_p
            return which_token,best,best2,grads_norm
        which_token, best,best2,grads_norm= hotflip_attack()
