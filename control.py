from data import *
from PINN_Curriculum import *


########################################################
" Parameters used in the data generation"
# domain of x and t
minval = [0., 0.]
maxval = [2 * np.pi, 1.]
# number of points in training set
ntrain = 1000
# number of points in the test set
ntest = 1000
# number of points at t = 0
nic = 100
# number of temporal points at the boundary
nbc = 50
# generate the data for training the PINN
input_train, input_init, input_bc1, input_bc2 = data_train(ntrain, nic, nbc, minval, maxval)
# generate the data for testing
input_test = data_test(ntest, minval, maxval)
########################################################
" Hyperparameters used in the PINN"
# number of units in each layer including the input\output layer
dim = [input_test.shape[1], 20, 20, 20, 20, 20, 20, 20, 20, 1]
# number of epochs in the training process
epoch = 1000
# optimization algorithm used in the PINN
opt = tf.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
# constant parameter in the PDE
beta = np.linspace(1, 30, 30)
print(beta)

# initialize the neural network
model = NN.network(dim)
model = PINN.train(model, input_train, input_init, input_bc1, input_bc2, beta, epoch, opt, input_test)

"""" Curriculum process"
for i in range(len(beta)):
    # generate the data for training the PINN
    input_train, input_init, input_bc1, input_bc2 = data_train(ntrain, nic, nbc, minval, maxval)
    # train the model
    model = PINN.train(model, input_train, input_init, input_bc1, input_bc2, beta[i], epoch, opt)
    # prediction of the test set
    pred_test = prediction(model, input_test)
    # plot the prediction of test set
    plot(pred_test, beta[i], input_test.shape[0])
    del i
model.save('PINN_Curriculum.keras')"""
