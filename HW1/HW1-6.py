"""
Regression: for this part, you will use the average localization error (ALE) dataset from the previous
question. Begin by splitting the data (80/20) into a training and test set. As a baseline, use sklearn’s implementation to train a linear regression model on the training data and
evaluate its performance on the test set using the mean absolute error.
"""

import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Test-train split: 80/20
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Rescale to have mean 0 and std 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Baseline: sklearn linear regression
baseline = LinearRegression()
baseline.fit(X_train, y_train)
baseline_mean_abs_error = mean_absolute_error(y_test, baseline.predict(X_test))

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Linear model f(x) = w^T x + b
w = torch.zeros(X_train.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linear_model(X):
    return X @ w + b

# Mean squared error loss
def mse_loss(y, pred):
    return ((y - pred) ** 2).mean()

# SGD
optimizer = torch.optim.SGD([w, b], lr=0.01)
epochs, batch_size = 100, 64

for epoch in range(epochs):
    perm = torch.randperm(len(X_train))  # shuffle each epoch
    for start in range(0, len(X_train), batch_size):
        idx = perm[start:start + batch_size]
        loss = mse_loss(y_train[idx], linear_model(X_train[idx]))
        optimizer.zero_grad()  # clear the old gradients
        loss.backward()        # compute the gradients
        optimizer.step()       # update the weights

# Test the model: mean absolute error
test_pred = linear_model(X_test)
mean_abs_error = (y_test - test_pred).abs().mean().item()

print(f"Baseline test mean absolute error: {baseline_mean_abs_error:.4f}")
print(f"PyTorch model test mean absolute error: {mean_abs_error:.4f}")

# Comparison with baseline
diff = (mean_abs_error - baseline_mean_abs_error) * 100
if diff > 0:
    print(f"The linear regression model outperforms the baseline by {diff:.2f} percentage points.")
elif diff < 0:
    print(f"The linear regression model underperforms the baseline by {-diff:.2f} percentage points.")
else:
    print("The linear regression model matches the baseline accuracy.")
    
    
""" 
Earlier you saw that the squared error loss can be derived from the assumption that the residuals
of the fitted model should be (approximately) normal. However, the normal distribution has weak
tails, so it is sensitive to outliers. An alternative is the t-distribution, which has heavier tails.
2
The t-distribution is parameterized by the degrees of freedom ν. When ν → ∞, the t-distribution
converges to the standard normal distribution. The probability density function of the t-distribution
is:
p(yi − f(xi),ν) = Γ ν+1
√
2
νπΓ ν
2
1+ (yi−f(xi))2
where Γ is the gamma function.
The corresponding loss function is the negative log-likelihood:
L(yi,f(xi)) = −logp(yi − f(xi),ν)
−ν+1
2
ν
For this exercise, use the t-distribution with ν = 5. If you are using JAX, you can use
jax.scipy.stats.t.logpdf to implement this loss function. If you are using PyTorch, you can use
torch.distributions.StudentT.log prob.
Use your chosen framework to implement a linear regression model and train it on the training data
by minimizing this custom loss function with SGD. Evaluate the performance of your model on the
test set. How does it compare to the baseline?
"""

# t-distribution
student = torch.distributions.StudentT(df=5.0)
 
# Linear model f(x) = w^T x + b
w_t = torch.zeros(X_train.shape[1], requires_grad=True)
b_t = torch.zeros(1, requires_grad=True)
 
def linear_model_t(X):
    return X @ w_t + b_t
 
# t-distribution loss
def t_loss(y, pred):
    return -student.log_prob(y - pred).mean()
 
# SGD
optimizer_t = torch.optim.SGD([w_t, b_t], lr=0.01)
 
for epoch in range(epochs):
    perm = torch.randperm(len(X_train))  # shuffle each epoch
    for start in range(0, len(X_train), batch_size):
        idx = perm[start:start + batch_size]
        loss = t_loss(y_train[idx], linear_model_t(X_train[idx]))
        optimizer_t.zero_grad()  # clear the old gradients
        loss.backward()          # compute the gradients
        optimizer_t.step()       # update the weights
 
# Test the model
test_pred_t = linear_model_t(X_test)
mean_abs_error_t = (y_test - test_pred_t).abs().mean().item()
 
print(f"t-loss model test mean absolute error: {mean_abs_error_t:.4f}")
 
# Comparison with baseline
diff_t = baseline_mean_abs_error - mean_abs_error_t
if diff_t > 0:
    print(f"The t-loss linear regression model outperforms the baseline by {diff_t:.4f} in mean absolute error.")
elif diff_t < 0:
    print(f"The t-loss linear regression model underperforms the baseline by {-diff_t:.4f} in mean absolute error.")
else:
    print("The t-loss linear regression model matches the baseline mean absolute error.")