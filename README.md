# accSDA_matlab

# Accelerated Sparse Discriminant Analysis

This provides Matlab functions accompanying the paper [Proximal Methods for Sparse Optimal Scoring and Discriminant Analysis](https://arxiv.org/pdf/1705.07194.pdf).

The package is under continuous development and most of the basic functionality is available! You can now apply sparse discriminant analysis with `accSDA`, see the tutorials below to get started.

# Why should you use this package?

Do you have a data set with a **lot of variables and few samples**? Do you have **labels** for the data?

Then you might be trying to solve an *p>>n* classification task.

This package includes functions that allow you to train such a classifier in a sparse manner. In this context *sparse* means that only the best variables are selected for the final classifier. In this sense you can also interpret the output, i.e., use it to identify important variables for your classification task. The current functions also handle cross-validation for tuning the sparsity, look at the documentation for further description/examples.

# Installation
You can use the package immediately after adding the folder containing all function files (found in the folder `accSDA_matlab`) to your Matlab path.

# Usage

You can train sparse discriminant vectors using the Matlab function `ASDA`, called using the command:
```Matlab
ASDAres = ASDA(X, Y, Om, gam, lam, cv, method, q, insteps, outsteps, intol, outtol, quiet, opts)
```
The function `ASDA` performs block coordinate descent to solve the sparse optimal scoring problem to train discriminant vectors as described in the paper [Proximal Methods for Sparse Optimal Scoring and Discriminant Analysis](https://arxiv.org/pdf/1705.07194.pdf). The function requires the following input arguments
* `X` a n x p training data matrix, with rows containing n observations of p-dimensional data vectors. This data must be standardized so that each predictor feature, i.e., column, has sample mean 0 and variance 1. We have provided the Matlab function `normalize` to standardize data if needed.
* `Y` a n x k training data class indicator matrix, where k is the number of unique class labels. This is a binary matrix with (i,j) entry equal to 1 if observation i belongs to class j and 0 otherwise.
* `Om` a p x p positive semidefinite matrix controlling the generalized Tikhonov regularization function in the sparse optimal scoring problem. We suggest using the identity matrix unless a domain-specific regularization is needed.
* `gam` a positive scalar giving the weight of the Tikhonov regularization penalty.
* `lam` defines the weight of the l_1 penalty term in the sparse optimal scoring problem. If `cv` is `true` then `lam` is a vector of positive scalars to be compared using cross validation. If `cv` is `false` then `lam` is a positive scalar giving the weight of the penalty term.
* `cv` a logical variable indicating whether to use cross validation to train the weight of the l1 norm penalty.
* `method` a string indicating the method to be used to solve the β-update subproblem:
  * `"PG"` proximal gradient method,
  * `"APG"`  accelerated proximal gradient method, or
  * `"ADMM"` alternating direction method of multipliers.
* `q` integer between 1 and k-1, indicating how many discriminant vectors to calculate.
* `insteps`, `outsteps` are positive integers indicating the number of iterations to be performed in the inner loop for subproblem solution and the outer loop, respectively, of the block coordinate descent algorithm.
* `intol`, `outtol` are positive scalars indicating stopping tolerance for the inner and outer loops, respectively, of the block coordinate descent algorithm.
* `quiet` a logical variable indicating whether to display intermediate output.
* `opt` a structured list variable providing any additional solver-dependent arguments.

The function `ASDA` returns a structured list `ASDAres` containing
* `ASDAres.B` a p x q matrix with columns containing the discriminant vectors calculated by the block coordinate descent method.
* `ASDAres.Q` a k x q matrix with columns containing the optimal scoring vectors.

If `cv = true` then the additional values are returned:
* `ASDAres.bestind`, which is the index of the value of `lam` chosen by cross validation,
* `ASDAres.bestlam`, which is the value of `lam` chosen by cross validation,
* `ASDAres.cvscores`, which is the matrix of cross validation scores for all folds and possible choices of parameter λ in `lam`.
You can test accuracy of the calculated discriminant vectors for nearest centroid classification using the provided Matlab function `predict`.

If using the **(accelerated) proximal gradient method**, i.e., `method = "PG"` or `method = "APG",` then you must provide the additional argument `opts.bt` to indicate whether to use backtracking line search (`opts.bt = true`) or constant step-size (`opts.bt = false`) in the (accelerated) proximal gradient method). If `opts.bt` is true then you need to provide the following additional arguments:
* `opts.L` estimate of the Lipschitz constant to initialize the backtracking line search.
* `opts.eta` a scalar > 1 giving the scaling factor for the backtracking line search.

If using the **alternating direction method of multipliers**, i.e., `method = "ADMM"`, you must provide the additional argument `opts.mu`, which is a positive scalar giving the weight of the augmented Lagrangian penalty in the ADMM scheme.

If using cross validation (`cv = true`) to train λ, you must provide the following arguments to the CV scheme:
* `opts.fold` a positive integer, less than n, indicating the number of folds to use in the CV scheme, and
* `opts.feat` a scalar in the interval (0,1) indicating the desired proportion of nonzero predictor variables in calculated discriminant vectors. We use misclassification as CV score for all calculated vectors that contain at most this proportion of nonzero features, and use the number of nonzero features otherwise.

# An example: Coffee spectrograms.

The following is a simple example using the Coffee data set from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).
The training set consists of *n=28* spectrographic observations of either *arabica* or *robusta* variants of instant coffee, each containing *p=286* feature variables. We include a standardized version of this data set in the folder `Data` in the repository as the file `Coffee.mat`.


We first define input arguments. Here, we use the accelerated proximal gradient method with backtracking and cross validation; we'll discuss the usage of other methods later.

```Matlab

% Load data set.
load('Data\Coffee.mat')

% Define required parameters.
p = 286; % Predictor variable sizes.
Om = eye(p) % Use identity matrix for Om.
gam = 1e-3;
q = 1; % One fewer than number of classes.
quiet = true; % Suppress output.

% Optimization parameters.
insteps = 1000;
intol = 1e-4;
outsteps = 25;
outtol = 1e-4;

% Choose method and additional required arguments.
method = "APG" % Use accelerated proximal gradient.
opts.bt = true; % Use backtracking.
opts.L = 0.25; % initial Lipschitz constant for BT.
opts.eta = 1.25; % scaling factor for BT.

% Use cross validation, and provide required arguments.
cv = true;
opts.folds = 7; % Use 7-fold cross-validation.
opts.feat = 0.15; % Want classifiers using at most 15% of features.
```

After defining all input arguments, we are ready to solve for the discriminant vectors.

```Matlab

% Call ASDA.
ASDAres = ASDA(X, Y, Om, gam, lam, cv, method, q, insteps, outsteps, intol, outtol, quiet, opts)
```
The `ASDAres` object contains the calculated discriminant vector, scoring vector, and additional data from the cross validation process. We can plot the discriminant vector using the command
```Matlab
plot(ASDAres.B)
```
which yields the following plot. (This may look slightly different when you run the code due to inherent variation in the cross validation scheme and due to random initialization of the scoring vectors.)

![Image of discriminant vector for Coffee data](./Examples/coffeedvs.png)


We can test performance of our classifiers using the `predict` function. We use the following code to do so:
```Matlab
% First calculate centroid matrix of training data.
C = diag(1./diag(Y'*Y))*Y'*X;


% Call predict to calculate accuracy on testing data.
% Note: predict requires matrix with class centroids as columns.
stats = predict(ASDAres.B, test, C')
```

This yields the following table containing proportion of misclassified testing observations and total cardinality of discriminant vectors.

Statistic | Value
-----|--------
Misclassification Rate | 0
Cardinality | 33 (out of 286 features)

(Again, these values may change depending on which value of λ is chosen during cross validation.)

# Another Example: Olive Oil spectrograms.

The following example illustrates the use of the alternating direction method of multipliers to classify $n=30$ samples of olive oil from olives grown in one of four different geographic regions using spectrographic measurements (*p=570*). This data set is also available from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) and the `Data` folder in our repository (stored as the file `OliveOil.mat`).

We set all input parameters and call ASDA using the following Matlab commands; note that we use ADMM and choose *λ = 0.01* without cross validation.

```Matlab
% Load data set.
clear; clc;
load('Data\OliveOil.mat')


%% 
% Define required parameters.
p = 570; % Predictor variable sizes.
Om = eye(p); % Use identity matrix for Om.
gam = 1e-3; 
q = 3; % One fewer than number of classes.
quiet = true; % Suppress output.

% Optimization parameters.
insteps = 1000;
intol = 1e-4;
outsteps = 25;
outtol = 1e-4;

% Choose method and additional required arguments.
method = "ADMM"; % Use alternating direction method of multiplier.
opts.mu = 5; % Set augmented Lagrangian parameter.

% Don't use cross validation. Set lambda.
cv = false;
lam = 1e-2; % Choose lambda.

%% Call ASDA.
ASDAres = ASDA(X, Y, Om, gam, lam, cv, method, q, insteps, outsteps, intol, outtol, quiet, opts);

```

As before, we plot the discriminant vectors using the command
```Matlab
plot(ASDAres.B)
```

![Image of discriminant vectors for Olive Oil data](./Examples/oliveoildvs.png)

We next perform nearest centroid classification in the span of the discriminant vectors.
```Matlab
% First calculate centroid matrix of training data.
C = diag(1./diag(Y'*Y))*Y'*X;


% Call predict to calculate accuracy on testing data.
% Note: predict requires matrix with class centroids as columns.
stats = predict(ASDAres.B, test, C')
```
This yields the following table of performance metrics:

Statistic | Value
-----|--------
Misclassification Rate | 0.0667 (or 2/30)
Cardinality | 67 (out of 1710 total features)
