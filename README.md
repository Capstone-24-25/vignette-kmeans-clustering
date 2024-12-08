# vignette-kmeans-clustering
Conceptual explanation and practical example of the k-means clustering algorithm.

**Contributors:** Sophia Mirashidi, Leena Anqud, Nazhah Mir, Hannah Kim

## Abstract
This project utilizes the UC Irvine Wine Quality dataset to illustrate the impact of k-means clustering, a form of unsupervised learning that groups data points into clusters based on feature similarity and their proximity to the centroid. The dataset contained information about different wines, including their color, pH, residual sugar, density, and more. Using K-means clustering, we were able to predict the color of wine, red or white, based on a variety of factors. Ultimately, our model achieved an accuracy of 0.9763, with only 32 red wins and 60 white wines misclassified out of 6947 total observations.

## Repository Contents

-  **data**: contains the two sub-datasets for red and white wine quality
- **imgs**: contains high definition images of the histogram of variables from EDA and the scatterplot of clusters based on features
- **README.md**: contains basic information about the repository and project
- **scripts**: contains the first project draft and annotated code
- **vignette_files**: contains two folders
	1. ***figure-html***:  contains high definition images of histograms and correlation matrices from EDA and the cluster scatterplot
	2. ***libs***: contains downloaded project libraries
- **vignette.qmd**: the final draft of the project containing a complete overview of K-means clustering and a practical application

## References

Coburn, K. (2024). Unsupervised learning: K-means clustering [Lecture]. University of California, Santa Barbara.
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality \[Dataset\]. UCI Machine Learning Repository. <https://doi.org/10.24432/C56S3T.>\
IBM. (n.d.). K-means clustering. Retrieved December 3, 2024, from <https://www.ibm.com/topics/k-means-clustering> GeeksforGeeks. (2024). K means Clustering - Introduction. Retrieved December 3, 2024, from <https://www.geeksforgeeks.org/k-means-clustering-introduction/>

### To Learn More

Chapter 12.4.1 in the textbook [*An Introduction to Statistical Learning*](https://www.statlearning.com/) provides a much more in-depth explanation of the theoretical math foundations of K-means clustering. The textbook is available with examples in both Python and R.

This [handout](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) from Stanford University expands on the connection between K-means clustering and the Expectation Maximization algorithm that we briefly touch on in our explanation.