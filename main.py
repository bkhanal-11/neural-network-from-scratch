from utils import load_data
from simple_L_layer_NN import L_layer_model, predict


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
# CONSTANTS
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

pred_train, train_acc = predict(train_x, train_y, parameters)
pred_test, test_acc = predict(test_x, test_y, parameters)

print(f'\nTrain Accuracy: {train_acc}\nTest Accuracy: {test_acc}')
