# import tensorflow as tf
import keras as k
import matplotlib.pyplot as plt
import numpy as np

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
    
    # create Sequential model
    model = k.models.Sequential()
    
    # define the layers in CNN
    # if we would work with our own pictures database then here we would need to add as input layer the Conv2D to transform the pictures to list of flat number vectors
    # because we have data set from Keras, we dont need to do it and we can just say that the input will be flat vector
    model.add(k.layers.Flatten())
    # add hidden layer with 128 neurons and relu as a activation function
    model.add(k.layers.Dense(128, activation='relu'))
    # add the output layer with activation function softmax
    model.add(k.layers.Dense(10,activation='softmax'))
    
    #compile the model with specify optimizer and loss function and metrics
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    #train the model with train data
    model.fit(x_train, y_train, epochs=10)
    
    # validate the model with test data
    validate = model.evaluate(x_test, y_test)
    print(validate)
    
    #make prediction from test data
    prediction = model.predict(x_test)
    print(prediction[0])
    
    #translate the prediction
    print(np.argmax(prediction[0]))
    
    plt.imshow(x_test[0])
    plt.show()
    
    