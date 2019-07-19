# Import the necessary components from Keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dense

import numpy as np

(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()
n=1000

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print((n,x_train.shape[1],x_train.shape[2],1))

# The number of image categories
n_categories =max(y_train)+1
categories=np.array(np.unique(y_train))

print(categories,n_categories)
train_data=np.empty((n,x_train.shape[1],x_train.shape[2],1),dtype=float)
train_labels=np.zeros((n,n_categories),dtype=float)

for i in range(n):
    # Find the location of this label in the categories variable
    for jj in range(n_categories):
        if (categories[jj]==y_train[i]):
            #print(jj)
            j=jj

    train_data[i,:,:,0]=x_train[i,:,:]
    train_labels[i,j]=1
print(train_labels[0:3,:])


test_data=np.empty((n,x_test.shape[1],x_train.shape[2],1),dtype=float)
test_labels=np.zeros((n,n_categories),dtype=float)

for i in range(n):
    # Find the location of this label in the categories variable
    for jj in range(n_categories):
        if (categories[jj]==y_test[i]):
            #print(jj)
            j=jj

    test_data[i,:,:,0]=x_test[i,:,:]
    test_labels[i,j]=1
print(test_labels[0,:])


# Initializes a sequential model
model1 = Sequential()

# First layer
model1.add(Dense(10, activation='relu', input_shape=(784,)))

# Second layer
model1.add(Dense(10, activation='relu'))

# Output layer
model1.add(Dense(n_categories, activation='softmax'))

# Compile the model
model1.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

# Reshape the data to two-dimensional array
train_data_r = train_data.reshape(n, 784)

# Fit the model
model1.fit(train_data_r, train_labels, validation_split=0.2, epochs=3)

print('/////////////////////////////////////  MODEL 2: CONVOULTION ADDED /////////////')
img_rows=28
img_cols=28
# Initialize the model object
model2 = Sequential()

# Add a convolutional layer
model2.add(Conv2D(10, kernel_size=3, activation='relu',
               input_shape=(img_rows,img_cols,1)))

# Flatten the output of the convolutional layer
model2.add(Flatten())
# Add an output layer for the 3 categories
model2.add(Dense(n_categories, activation='softmax'))
# Compile the model
model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(test_data.shape,test_labels.shape)
# Fit the model on a training set
model2.fit(train_data,  train_labels,
          validation_split=0.2,
          epochs=3, batch_size=10)
# Evaluate the model on separate test data
model2.evaluate(test_data,test_labels,batch_size=10)

print('/////////////////////////////////////  MODEL 3: TWEAK CONVOULTION  /////////////')
# Initialize the model object
model3 = Sequential()

# Add a convolutional layer
model3.add(Conv2D(10, kernel_size=3, activation='relu',
               input_shape=(img_rows,img_cols,1),padding='same',strides=2))

# Flatten the output of the convolutional layer
model3.add(Flatten())
# Add an output layer for the 3 categories
model3.add(Dense(n_categories, activation='softmax'))
# Compile the model
model3.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model on a training set
model3.fit(train_data,  train_labels,
          validation_split=0.2,
          epochs=3)
# Evaluate the model on separate test data
model3.evaluate(test_data,test_labels)
print('')
print('/////////////////////////////////////  MODEL 4: DEEPER CONVOULTIONAL NETWORK  /////////////')
# Initialize the model object
model4 = Sequential(Seed=123)

# Add a convolutional layer
model4.add(Conv2D(15, kernel_size=3, activation='relu',
               input_shape=(img_rows,img_cols,1)))
# Add a convolutional layer
model4.add(Conv2D(5, kernel_size=3, activation='relu'))
# Flatten the output of the convolutional layer
model4.add(Flatten())
# Add an output layer for the 3 categories
model4.add(Dense(n_categories, activation='softmax'))
# Compile the model
model4.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model on a training set
model4.fit(train_data,  train_labels,
          validation_split=0.2,
          epochs=3)
# Evaluate the model on separate test data
model4.evaluate(test_data,test_labels)
