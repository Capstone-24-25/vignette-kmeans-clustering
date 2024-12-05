import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans


##############################################
# Manual Implementation of K-means Clustering #
##############################################

class kMeans():
    def __init__(self, K, X): # Initialize k
        self.K = K                                                                      # store K value 
        self.centroids = X[np.random.choice(X.shape[0], size=K, replace=False)]         # initialize random centroids 
        self.assignments = np.zeros(X.shape[0], dtype=int)                              # create vector to store assignments
        self.X = X                                                                      # store data

    def update_centroids(self, closest_centroids):
        """This function will update the stored centroids according to the new clusters"""

        X_grouped = [self.X[closest_centroids == idx] for idx in range(len(self.centroids))]        # group the datapoints into clusters by the centroids they are assigned to
        self.centroids = [X.mean(axis=0) for X in X_grouped]                                        # calculate the mean of each group to find the centroids of the centroids clusters
        
    def find_cluster(self): 
        """This method will iteratively search for clusters"""

        while True:                                                                                                # begin iteratively searching for clusters
            distances = np.array([np.linalg.norm(self.X - centroid, axis=1) for centroid in self.centroids]).T     # calculate the distance matrix; rows are data points, columns are clusters
            closest_centroids = np.argmin(distances, axis=1)                                                       # calculate the closest centroid to each data point

            if np.array_equal(closest_centroids,self.assignments):      # check for convergence
                break                                                   # break if we have converged to a solution
            else:
                self.update_centroids(closest_centroids)                # otherwise, update the stored centroids
                self.assignments = closest_centroids                    # store the current assignments

        return self.assignments        # return our solution
    

#####################
# Practical Example #
#####################

red = pd.read_csv('data/winequality-red.csv', sep=';', header=0)        # load in red wine data
white = pd.read_csv('data/winequality-white.csv', sep=';', header=0)    # load in white wine data

red['color'] = 'red'                                                    # create color variable for red wine data
white['color'] = 'white'                                                # create color variable for white wine data

wine_data = pd.concat([red, white], axis=0)                             # combine both datasets into one
print(len(wine_data))                                                   # calculate the length of the datasets to confirm concatenation
print(pd.value_counts(wine_data['color']))                              # confirm the value counts of the red and white wine 


correlation_matrix = wine_data.drop(columns=['color']).corr()                           # create correlation matrix of the quantitative data
plt.figure(figsize=(10, 8))                                                             # set the figure size
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)      # create the heatmap to present the matrix 
plt.title('Correlation Matrix of Wine Quality Dataset')                                 # title the figure
plt.show()                                                                              # show the figure 