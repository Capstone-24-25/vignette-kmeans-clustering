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
plt.show()                                                                             # show the figure 

wine_data = wine_data.drop(columns=['free sulfur dioxide']) #drop free sulfur dioxide, as that is directly related


wine_data.hist(bins=20, figsize=(15, 10)) #generates histograms for distribution of each variable
plt.show() #shows graph


def remove_extreme_outliers_iqr(data, column, multiplier=3): #function that removes only extreme outliers 
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

columns_to_clean = ['fixed acidity', 'volatile acidity', 'chlorides', 'sulphates'] #cleans columns that still had large amounts of skewness even after transformation
for col in columns_to_clean:
    wine_data = remove_extreme_outliers_iqr(wine_data, col, multiplier=3)


numerical_columns = wine_data.select_dtypes(include=['number']) #finds all numerical columns in dataset
skewness = numerical_columns.skew() #calculates skewness of all numerical columns using Fisher-Pearson coefficient
print(skewness) #prints out skewness

# Log transformation for heavy skew
log_transform_vars = ['residual sugar', 'fixed acidity', 'volatile acidity', 'chlorides'] #identifies which columns are heavily skewed and should be log transformed
for col in log_transform_vars:
    wine_data[col] = np.log1p(wine_data[col]) #loops through the columns and log transforms them

# Square root transformation for moderate skew
sqrt_transform_vars = ['sulphates', 'alcohol'] #identifies which columns are moderately skewed and should be square root transformed
for col in sqrt_transform_vars:
    wine_data[col] = np.sqrt(wine_data[col]) #loops through the columns and square root transforms them

skewness = numerical_columns.skew() #recalculates skewness 
# Print skewness values
print(skewness) #prints out new skewness values after transformation