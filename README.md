# Numpy AI Models

This repository contains a collection of AI models implemented using NumPy. These models are part of my studies on various machine learning algorithms and their applications. The models currently included are:

1. **Principal Component Analysis (PCA)**
2. **Decision Trees**

Due to university policies, I cannot upload any projects until the submission date is reached. The models soon to be added are:
1. **K-Means Clustering**
2. **Gradient Boosting**


## Principal Component Analysis (PCA)

Principal Component Analysis is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional form while preserving as much variance as possible. This implementation in NumPy focuses on the following steps:
- Standardizing the data.
- Computing the covariance matrix.
- Calculating the eigenvalues and eigenvectors.
- Sorting the eigenvalues and eigenvectors.
- Transforming the data to the new basis.

## Decision Trees

Decision Trees are a type of supervised learning algorithm used for classification and regression tasks. This implementation includes testing different criteria for splitting the nodes:
- **Gini Impurity**
- **Entropy**
- **Misclassification Error**

### Findings on the Iris Dataset

Through experimentation with the Iris dataset, I found that using the Misclassification Error as the criterion provided the best performance compared to Gini Impurity and Entropy. The results indicated that Misclassification Error resulted in higher accuracy and better generalization for this specific dataset.

## K-Means Clustering

K-Means Clustering is a method for partitioning data into K distinct clusters. This model aims to minimize the variance within each cluster. 

## Gradient Boosting

Gradient Boosting is an ensemble learning technique that builds a model in a stage-wise fashion from weak learners, typically decision trees. For my implementation, I modified the decision tree algorithm to allow the leaf nodes to return residuals instead of predictions. This modification enables the model to correct errors from previous iterations, improving overall performance.



