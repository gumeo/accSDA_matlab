# accSDA_matlab

# Accelerated Sparse Discriminant Analysis

This provides Matlab functions accompanying the paper [Proximal Methods for Sparse Optimal Scoring and Discriminant Analysis](https://arxiv.org/pdf/1705.07194.pdf).

The package is under continuous development and most of the basic functionality is available! You can now apply sparse discriminant analysis with `accSDA`, see the tutorials below to get started.

# Why should you use this package?

Do you have a data set with a **lot of variables and few samples**? Do you have **labels** for the data?

Then you might be trying to solve an *p>>n* classification task.

This package includes functions that allow you to train such a classifier in a sparse manner. In this context *sparse* means that only the best variables are selected for the final classifier. In this sense you can also interpret the output, i.e., use it to identify important variables for your classification task. The current functions also handle cross-validation for tuning the sparsity, look at the documentation for further description/examples.

# Installation
You can use the package immediately after adding the folder containing all function files to your Matlab path.

# Usage
