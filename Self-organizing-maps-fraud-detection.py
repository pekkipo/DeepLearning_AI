import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom

#### Fraud detection with credit cards

dataset = pd.read_csv('Credit_Card_Applications.csv')

# rows are customers
# columns are attributes
# SOM will do customer segmentation, one of the customers segment will be a potential frauds group

# All customers are the input of our NN
# these inputs points will be mapped to a new output space
# between the input and the output space we have the neural network composed of neurons with each neuron being initialized as a vector of weights of the same size and the vector of customers
# 15 in this case as we have customer id + 14 attributes
# So for observation point, i.e. for each customer the output of the customer will be the neuron that is the closest to the customer
# for each customer the winning node (BMU also) we ll use gaussian neighbourhood function to update the weights of the neighbours of the winning node to move thenm closer to the point
# we do this for all the customers in the input space
# and we repeat it many times
# after each repeat the output space decreases and looses dimensions
# then we reach the point where the neighbourhood stops decreasing (similar to K-means clustering)
# that's the moment where we obtain the SOM

# After obtaining the SOM we basically get the general patterns
# WHile frauds are outliers, smth that doesn't comply with the general rules
# the frauds are the outlying neurons in the 2d SOM

# How to detect outliers?
# through the mean inter-neuron distance (MID)
# for each neuron the mean of the euclidean distance between this neuron and the neourons in its neighbourhood
# neighbourhood we define manually though
# outliers will be far from the neurons in its neighbourhood
# then we ll use inverse mapping function to identify which customers in the original input space that are associated with this winning node that is an outlier

#print(dataset.sample)

# Create two subsets
# want to especially distinguish cutomers whose credit was approved or not (0 or 1 in the last column)
X = dataset.iloc[:, :-1]  # all rows but all columns except for the last
y = dataset.iloc[:, -1]  # last column
# not trying to predict and make it supervised learning! Just want to distinguish between these groups later

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
# This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one.
# for SOM I need inputs in the range between 0 and 1
sc = MinMaxScaler(feature_range= (0, 1)) # normalization basically

X = sc.fit_transform(X)  # return normalized version of X
# so all the data in the dataset is now filled with values between 0 and 1
#print(X)


# Above was short preprocessing
# Now training the SOM
# Not gonna write own implementation, minisom implementation is really good and has free license by Giusuppe Vettigli
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)  # x and y are dimension of the grid. shouldn't be too small to detect outliers. input_len is the number of features, 14+1 in this case
# sigma is radius for MID
# we dont have that many customers, so 10 by 10 grid should be fine
# first we initalize the weight, see the algorithm steps
som.random_weights_init(X)

# train
som.train_random(data=X, num_iteration=1000)  #  reinforcement learning, update the weights after each observation

# Visualizing the result

# We ll see the 2D grid that will contain all final winning nodes and we will get MIDs for the specific nodes
# MID of a node - mean of the distances of all the neurons around the winning node based on sigma (radius of the neighbourhood).
# The higher is MID the further the winning node will be from it s neoigbours. So the more the MID - the more it is an outlier, cauz it s the furthest from general behaviour pattern

# We ll use colors. The larger the MID, the closer to white the color will be

from pylab import bone, pcolor, colorbar, plot, show

bone()  # init the window for the map

# put the different winning nodes on the map. put info about the MIDs
# different colors correspond to the different values of MIDs

pcolor(som.distance_map().T)  # som.dist will return a matrix of distances but we transpose the matrix

# add legend
colorbar()

# white are frauds

# now we ll distiguish between those who got approval
# more interested in the ones who got the approval and are frauds

# red squares - who didnt, green - did
markers = ['o', 's']
colors = ['r', 'g']

# loop over all customers
# for each customer get the winning node
# whether got approval or not - paint green or red

for i, x in enumerate(X):  # i indexes of customers
    w = som.winner(x)  # get the dinning node for the customer x
    plot(w[0] + 0.5, # place colored marker on it. at the center of the winning node (square) , w 01 is coord of bottom left corner, so we add 0.5 to put the marker at the center
         w[1] + 0.5,
         markers[y[i]], # y contains the info about approval or not, so we paint if respective color
                        # interesting thing here, y has 0 and 1 and we use them as index in markers list
         markeredgecolor=colors[y[i]], # only color the edge
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)

show()

# we care about the approval because that way we know the frauds who actually got away with this

# now use this SOM to get the list of customers who cheated
# we ll find the customers associated with the white winning fraud nodes

# from minisom we can get a dict which is a mapping of nodes and customers
mappings = som.win_map(X)
# key is the coordinates of the winning node
# size would give the number of customers
# each line would correspond to one customer in associated with the winning node

# first have to get the coordinates of the outlying winning nodes
# have to manually look at the SOM. coords of the bottom left corner of the outlying nodes [1, 8] [6, 3]
# I got two. then will have to concatenate these (add customers from both of them)
frauds = np.concatenate((mappings[(1, 8)], mappings[(3, 6)]), axis=0)  # list of customers associated with these nodes, 0 is vertical concatentation
# vertical means that we will add lines with customers below already existent ones

# inverse the mapping to get the original numbers, because we scaled the values during the preprocessing
frauds = sc.inverse_transform(frauds)

# after that the analyst have to check these customers

print(frauds)







