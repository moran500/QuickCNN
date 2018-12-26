# import tensorflow as tf
import keras as k
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.callbacks import TensorBoard


if __name__ == '__main__':
    
    # load pre-processed data set from keras where source are hand-made numbers from 0-9 and also split to trained and test data
    mnist = k.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 
    
    # this will plot the picture which is encode in input data
#     plt.imshow(x_train[0])
#     plt.show()
    
    # normalizing training and test input data
    x_train = k.utils.normalize(x_train, axis=1)
    x_test = k.utils.normalize(x_test, axis=1)
    
    
    
    dense_layers_numbers = [1,2,3]
    dense_neurals_numbers = [64,128,256]
    optimazers = ['adam','SGD', 'RMSprop']
    
    for d_layer in dense_layers_numbers:
        for d_neurons in dense_neurals_numbers:
            for optimazer in optimazers:
                print("{}, {}, {}".format(d_layer,d_neurons,optimazer))
                
                # define the name of the model
                NAME = "quickCNN_MNIST_l_{}_N_{}_O_{}_{}".format(d_layer,d_neurons, optimazer,int(time.time()))
                
                # define tensorboard, this variable has to be forward in the fit function when the model is training
                tensorboard = TensorBoard(log_dir="log/{}".format(NAME))
                
                # create Sequential model
                model = k.models.Sequential()
                 
                # define the layers in CNN
                # if we would work with our own pictures database then here we would need to add as input layer the Conv2D to transform the pictures to list of flat number vectors
                # because we have data set from Keras, we dont need to do it and we can just say that the input will be flat vector
                model.add(k.layers.Flatten())
                # define how many of hidden layers will we have and will how money of neurons 
                for i in range(d_layer):
                    # add hidden layer with d_neurons and relu as a activation function
                    model.add(k.layers.Dense(d_neurons, activation='relu'))
                # add the output layer with activation function softmax
                model.add(k.layers.Dense(10,activation='softmax'))
        
                #compile the model with specify optimizer from list of testing optimazers and loss function and metrics
                model.compile(optimizer=optimazer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
                 
                #train the model with train data together with validation of the network
                model.fit(x_train, y_train, epochs=3, callbacks=[tensorboard], validation_data=[x_test, y_test])
                 
                # validate the model with test data
#                 validate = model.evaluate(x_test, y_test)
#                 print(validate)
                 
#                 #make prediction from test data
#                 prediction = model.predict(x_test)
#                 print(prediction[0])
#                  
#                 #translate the prediction
#                 print(np.argmax(prediction[0]))
                 
#                 plt.imshow(x_test[0],cmap=plt.cm.binary)
#                 plt.show()
    
    