import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

print(dataset.sample)

# Create two subsets
# want to especially distinguish cutomers whose credit was approved or not (0 or 1 in the last column)
x = dataset.iloc[:, :-1]  # all rows but all columns except for the last
y = dataset.iloc[:, -1]  # last column
# not trying to predict and make it supervised learning! Just want to distinguish between these groups later

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
