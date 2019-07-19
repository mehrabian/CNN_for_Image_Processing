# Import the necessary components from Keras
from keras.datasets import fashion_mnist

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()

n1=10000
n2=1000

def plot_loss(histroy,i):
    fig,ax=plt.subplots(2,2, sharex=True)
    #print(histroy)
    # Plot the training loss
    ax[0,0].plot(history['loss'])
    ax[0,0].set_title('loss')
    # Plot the validation loss
    ax[0,1].plot(history['val_loss'])
    ax[0,1].set_title('val_loss')
    # Plot the acc
    ax[1,0].plot(history['acc'])
    ax[1,0].set_title('acc')
    # Plot the val_acc
    ax[1,1].plot(history['val_acc'])
    ax[1,1].set_title('val_acc')
    fig.savefig('history_'+str(i)+'.png')
    plt.clf()

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print((n1,x_train.shape[1],x_train.shape[2],1))

# The number of image categories
n_categories =max(y_train)+1
categories=np.array(np.unique(y_train))

print(categories,n_categories)
train_data=np.empty((n1,x_train.shape[1],x_train.shape[2],1),dtype=float)
train_labels=np.zeros((n1,n_categories),dtype=float)

for i in range(n1):
    # Find the location of this label in the categories variable
    for jj in range(n_categories):
        if (categories[jj]==y_train[i]):
            #print(jj)
            j=jj

    train_data[i,:,:,0]=x_train[i,:,:]
    train_labels[i,j]=1
print(train_labels[0:3,:])


test_data=np.empty((n2,x_test.shape[1],x_train.shape[2],1),dtype=float)
test_labels=np.zeros((n2,n_categories),dtype=float)

for i in range(n2):
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
train_data_r = train_data.reshape(n1, 784)
# Fit the model
training1=model1.fit(train_data_r, train_labels, validation_split=0.2, epochs=5)

# Extract the history from the training object
history = training1.history

plot_loss(history,1)


print('')
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
training2=model2.fit(train_data,  train_labels,
          validation_split=0.2,
          epochs=3, batch_size=10)
history=training2.history
plot_loss(history,2)
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
training3=model3.fit(train_data,  train_labels,
          validation_split=0.2,
          epochs=3)
history=training3.history
plot_loss(history,3)
# Evaluate the model on separate test data
model3.evaluate(test_data,test_labels)
print('')
print('/////////////////////////////////////  MODEL 4: DEEPER CONVOULTIONAL NETWORK  /////////////')
# Initialize the model object
model4 = Sequential()

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
training4=model4.fit(train_data,  train_labels,
          validation_split=0.2,
          epochs=3)
history=training4.history
plot_loss(history,4)
# Evaluate the model on separate test data
model4.evaluate(test_data,test_labels)

# Summarize the model
model4.summary()

print('')
print('/////////////////////////////////////  MODEL 5: DEEPER CONVOULTIONAL NETWORK WITH POOLING /////////////')
model5=Sequential()
# Add a convolutional layer
model5.add(Conv2D(15, kernel_size=2, activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

# Add a pooling operation
model5.add(MaxPool2D(2))

# Add another convolutional layer
model5.add(Conv2D(5,kernel_size=2,activation='relu'))

# Flatten and feed to output layer
model5.add(Flatten())
model5.add(Dense(n_categories, activation='softmax'))
model5.summary()

# Compile the model
model5.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

# Fit to training data
training5=model5.fit(train_data,train_labels,epochs=5,validation_split=0.2)

history=training5.history
plot_loss(history,5)


print('')
print('/////////////////////////////////////  MODEL 6: DEEPER CONVOULTIONAL NETWORK WITH DROPOUT /////////////')
model6=Sequential()
# Add a convolutional layer
model6.add(Conv2D(15, kernel_size=2, activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

# Add a dropout layer
model6.add(Dropout(0.2))

# Add another convolutional layer
model6.add(Conv2D(5,kernel_size=2,activation='relu'))

# Flatten and feed to output layer
model6.add(Flatten())
model6.add(Dense(n_categories, activation='softmax'))
model6.summary()

# Compile the model
model6.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

# Fit to training data
training6=model6.fit(train_data,train_labels,epochs=5,validation_split=0.2)

history=training6.history
plot_loss(history,6)

print('')
print('/////////////////////////////////////  MODEL 7: DEEPER CONVOULTIONAL NETWORK BATCH NORMALIZATION /////////////')
model7=Sequential()
# Add a convolutional layer
model7.add(Conv2D(15, kernel_size=2, activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

# Add batch normalization layer
model7.add(BatchNormalization())

# Add another convolutional layer
model7.add(Conv2D(5,kernel_size=2,activation='relu'))

# Flatten and feed to output layer
model7.add(Flatten())
model7.add(Dense(n_categories, activation='softmax'))
model7.summary()

# Compile the model
model7.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

# Fit to training data
training7=model7.fit(train_data,train_labels,epochs=5,validation_split=0.2)

history=training7.history
plot_loss(history,7)
