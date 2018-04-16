# SOM * ann
# hybrid unsupervised and supervised

# predict who of the customers might be a fraud (supervised part)

# Part 1 - Identify the frauds with SOMS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1]  # all rows but all columns except for the last
y = dataset.iloc[:, -1]  # last column

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
X = sc.fit_transform(X)


som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)

som.random_weights_init(X)

# train
som.train_random(data=X, num_iteration=500)  #  reinforcement learning, update the weights after each observation

# Visualizing the result

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)  # som.dist will return a matrix of distances but we transpose the matrix
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):  # i indexes of customers
    w = som.winner(x)  # get the winning node for the customer x
    plot(w[0] + 0.5, # place colored marker on it. at the center of the winning node (square) , w 01 is coord of bottom left corner, so we add 0.5 to put the marker at the center
         w[1] + 0.5,
         markers[y[i]], # y contains the info about approval or not, so we paint if respective color
                        # interesting thing here, y has 0 and 1 and we use them as index in markers list
         markeredgecolor=colors[y[i]], # only color the edge
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)

show()

mappings = som.win_map(X)

print(mappings[(2, 6)])
print(mappings[(3, 4)])

frauds = np.concatenate((mappings[(2, 6)], mappings[(3, 4)]), axis=0)  # list of customers associated with these nodes, 0 is vertical concatentation
frauds = sc.inverse_transform(frauds)

# we got the result of frauds, we ll use to learn with ANN now

# Part 2 - Going from unsupervised to supervised

# Creating matrix of features
customers = dataset.iloc[:, 1:].values  # features. values - creates a numpy array
# we dont include customer id, so we take columns starting from the 2nd basically

# also need the dependent variable for supervised learning
# this variable would contain the outcome whether there was fraud or not, 0 and 1. Get it from part 1 SOM, i.e. frauds list
# initialize the vector of zeros of customers table size. so all not frauds, then replace some of these with 1 based on frauds list
is_fraud = np.zeros(len(dataset))

# put one for potential cheaters
# check if customer id is in the list of frauds
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:  # i customer, first column which is id
        is_fraud[i] = 1


# Now train the network with ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Standardize features by removing the mean and scaling to unit variance
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))  #input dim - number of features, 15 in this case

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)  # input, output
# we have little data, so 2 epochs is really enough

#classifier.save('models')
#classifier.save_weights('models')

# Part 3
# Now can make the final predictions
# Predicting the probability of frauds
y_predictions = classifier.predict(customers)

print(y_predictions)

y_predictions = np.concatenate((dataset.iloc[:, 0:1].values, y_predictions), axis=1) # 1 because we adding a column so horizontal concatenation
# 0:1 needed to make it a 2d array instead of a vector cauz y_predictions is a 2D array
# .values to make it a numpy array

# Sort by the probability now
# small trick to sort only second column but make the first column still correspond, numpy can do that

y_predictions = y_predictions[y_predictions[:, 1].argsort()] # sorts by the 1 column

print(y_predictions)







