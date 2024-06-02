from tensorflow.keras.layers import Dense, Input
from utilis import *
import time


class NN:
    @classmethod
    def network(cls, dim):
        # input layer
        x_inp = Input((dim[0],))
        # hidden layers
        for i in range(1, len(dim)-1):
            if i == 1:
                x = Dense(dim[i], activation='tanh',
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros')(x_inp)
            else:
                x = Dense(dim[i], activation='tanh',
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros')(x)
        # output layer
        x = Dense(dim[-1], activation=None,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros')(x)

        model = tf.keras.Model(inputs=x_inp, outputs=x)

        return model


class PINN:
    @classmethod
    def grad(cls, model, data):
        x, t = extract_var(data)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            pred = model(tf.concat([x, t], axis=-1))
        ux, ut = jacobian(tape, pred, x, t)
        return ux, ut

    @classmethod
    def pde_loss(cls, model, input_train, beta):
        ux, ut = cls.grad(model, input_train)
        loss = ut + beta * ux
        return tf.reduce_mean(tf.math.square(loss))

    @classmethod
    def init_loss(cls, model, input_init):
        pred = model(input_init)
        x, _ = extract_var(input_init)
        loss = pred - tf.math.sin(x)
        return tf.reduce_mean(tf.math.square(loss))

    @classmethod
    def boundary_loss(cls, model, input_bc1, input_bc2):
        pred_bc1 = model(input_bc1)
        pred_bc2 = model(input_bc2)
        loss = pred_bc1 - pred_bc2
        return tf.reduce_mean(tf.math.square(loss))

    @classmethod
    def loss_fn(cls, model, input_train, input_init, input_bc1, input_bc2, beta):
        loss_pde = cls.pde_loss(model, input_train, beta)
        loss_init = cls.init_loss(model, input_init)
        loss_bc = cls.boundary_loss(model, input_bc1, input_bc2)
        return loss_pde + loss_init + loss_bc

    @classmethod
    def train(cls, model, input_train, input_init, input_bc1, input_bc2, beta, num_epoch, optimizer, input_test):
        """
        Train the PINN with training data, boundary and initial conditions.
        :param model: neural network
        :param input_train: training set
        :param input_init: initial condition data
        :param input_bc1: left boundary condition
        :param input_bc2: right boundary condition
        :param beta: constant parameter in pde
        :param num_epoch: number of epochs
        :param optimizer: optimizer used for training
        :param input_test: test set
        :return: the trained model
        """
        start_time = time.time()
        for i in range(len(beta)):
            for e in range(num_epoch+i*500+1):
                with tf.GradientTape(persistent=True) as tape:
                    loss = cls.loss_fn(model, input_train, input_init, input_bc1, input_bc2, beta[i])
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    if e % 1000 == 0:
                        print('Epoch {}, Loss: {}, beta: {}'.format(e, loss.numpy(), beta[i]))
                del e
                pred_test = prediction(model, input_test)
                # plot the prediction of test set
                plot(pred_test, beta[i], input_test.shape[0])
            del i
        print('Training time:', time.time() - start_time)
        return model
