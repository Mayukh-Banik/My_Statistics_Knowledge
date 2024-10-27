# AMS 380 Data Analysis Midterm 1 - Study Sheet

## Lecture 1 - Statistical Learning

Using a vector $\ X $ the equation to represent the value of $ Y $ in terms of $ X $ can be written as:
$$ Y = f(X) + \epsilon $$
where $ \epsilon $ captures all measurement errors and additional discrepencies.

In order to find an ideal $ f(X) $ what can be done is to find the ideal function so $ E[X] $ is as accurate as possible.
You want to find the ideal version of these equations which are in the class of regression functions:
$$ f(x) = E(Y | X = x) $$ 

### E[X]
Below are two equations for $ E[X] $ the first is the discrete case, second is continuos variables:
$$ E[X] = \sum_{0}^{n} x_i p_i = \int_{-\infty}^{\infty} xf(x)dx $$

Properties for $ E[X] $ (a linear operator):
$$
E[aX] = aE[X] \newline
E[\sum X_i ] = \sum E[X_i] \newline
E[X * Y] = E[X] * E[Y] \newline
$$


Regression functions are the optimal predictor of response variable with respect to a minimization function, eg:
$$
    f(x) = E(Y | X = x) \newline
    Minimize \space E[(Y- g(X))^2 | X = x] \space for \space g \space \epsilon \space X = x \newline
$$

Irreducible error can be constructed as $\epsilon = Y - f(x)$, where there is usually a distribution of values for $ Y $ even if the value of $ f(x) $ is known

Reducible Error: $ (f(x) - \hat{f}(x))^2 \newline$ 
Irreducible Error: $ Var(\epsilon) $
The usual case is that the number of points with $ X = x $ for a particular data point, so use $ \hat{f}(x) = Ave(Y | X \in N(x)) $ where $ N(x) $ gives a region around $ x $.

This is known as nearest neighbor averaging, which works with a small parameter $ (p) $ size and a large number of observations $ (n) $.
The effectiveness of this technique goes does dramatically as $ p $ increases due to the curse of dimensionality. 
This is due to when $ p $ starts to increase, a greater fraction of $ n $ has to be given by $ N(x) $ in order to lower variance, but in higher dimensions, a large range given by $ N(x) $ will not guarantee values are close in feature space to the point being looked at, losing meaning.

## Linear Model

$$
Y = BX + \epsilon \newline
$$

Overfitting and Underfitting: $\newline$
An overfitting model is one where the model contains more parameters then the data supports.
$\newline$
An underfitting model is where there are too few parameters to accurately represent the data. $\newline$
A model is overfit on when it makes no errors on the training data: $\sum (y_i - \hat{y_i})^2 = 0$.

Flexible models can easily learn from new data, but are more complex to understand and compute with less readability. 
Non-flexible models include decision trees while flexible models would be neural networks.
### Bias Variance Trade-off
$$
E[(y - \hat{f}(x))^2] = Var(\hat{f}(x)) + [Bias(\hat{f}(x))]^2 + Var(\epsilon)
\newline
Bias[\hat{f}(x)] = E[\hat{f}(x)] - f(x)
\newline
$$


Qualitative Labels are when they are distinct groups without numercal values, eg [white, black].
To choose a class based on input data you would choose: 
$$
K \space classes \space from \space k = 0 \to k \newline

C(x) = k \space where \space  k = \underset{k}{\operatorname{argmax}} \space [p_0(x), ..., p_K(x)]
$$
The missclassification rate in this situation can be the number of misclassifications over the number of classifications. 

## Lecture 2 - Multiple Linear Regression

$$
Y = B_0 + B_1X + \epsilon \newline
\hat{y} = \hat\beta_0 + \hat\beta_1x \newline
$$
Epsilon ($\epsilon$) is the error term with the hat over the terms indicating they are estimators of the true value.
$$
\hat{y}_i = \beta_0 + \beta_1x_i \newline
e_i = y_i - \hat{y}_i
$$
In this instance the $i$ th residual referes to $e_i$ where $\hat{y}_i = \beta_0 + \beta_1x_i$ is the $i$ th prediction.

The residial sum of squares is:
$$
\sum e_i^2 = \sum(y_i-\hat\beta_0-\hat\beta_1x_i)^2 = \sum (y_i - \hat{y}_i)^2
$$

The OLS estimators are:
$$
\sum(y_i-\hat\beta_0-\hat\beta_1x_i)^2 = 0 \newline
dRSS/d\beta_0 = -2\sum(y_i - \hat\beta_0 - \hat\beta_1x) \newline
0 = -2\sum(y_i - \hat\beta_0 - \hat\beta_1x) \newline
\hat\beta_o = \bar{y} - \hat\beta_1 \bar{x} \newline
dRSS/d\beta_1 = -2\sum x_i (y_i - \hat\beta_0 - \hat\beta_1x) \newline
\hat\beta_1 = \sum(x_i - \bar{x})(y_i - \bar{y})/\sum(x_i - \bar{x})^2 \newline
$$

Some notations for later:
$$
S_{xx} = \sum(x_i - \bar{x})^2 \space S_x = \sum(x_i - \bar{x}) \newline
SE(x) = \sigma_x/\sqrt{n}
$$

Variance/SE for simple linear regression:
$$
Var(\hat\beta_o) = \sigma^2(\frac{1}{n} + \frac{\bar{x}}{S_{xx}})
$$