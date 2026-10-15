"""
Now you will implement a logistic regression model using a framework that supports automatic
differentiation.
Begin by defining a logistic regression model:
f(xi) = σ(wTxi +b)
where xi are the features for observation i, w are the weights, b is the bias, and σ is the sigmoid
function. Next, define a function for the binary cross-entropy loss:
L(yi,f(xi)) = −[yi log(f(xi)) + (1 − yi)log(1 − f(xi))]

Use your chosen framework to implement this model and train it on the training data by minimizing
the loss using SGD. Note that you may need to clip the output of the sigmoid function to avoid
numerical instability. Evaluate the performance of your model on the test set. How does it compare
to the baseline?
"""
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Test-train split: 80/20
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Rescale to have mean 0 and std 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Baseline: sklearn logistic regression
baseline = LogisticRegression()
baseline.fit(X_train, y_train)
baseline_accuracy = baseline.score(X_test, y_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Logistic model f(x) = sigmoid(w^T x + b)
w = torch.zeros(X_train.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def logistic_model(X):
    p = torch.sigmoid(X @ w + b)
    return torch.clamp(p, 1e-7, 1 - 1e-7)  # clip to avoid log(0)

# Binary cross-entropy loss
def bce_loss(y, p):
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()

# SGD
optimizer = torch.optim.SGD([w, b], lr=0.01)
epochs, batch_size = 100, 64

for epoch in range(epochs):
    perm = torch.randperm(len(X_train))  # shuffle each epoch
    for start in range(0, len(X_train), batch_size):
        idx = perm[start:start + batch_size]
        loss = bce_loss(y_train[idx], logistic_model(X_train[idx]))
        optimizer.zero_grad()  # clear the old gradients
        loss.backward()        # compute the gradients
        optimizer.step()       # update the weights

# Test the model
test_prob = logistic_model(X_test)
predictions = (test_prob >= 0.5).float()
accuracy = (predictions == y_test).float().mean().item()

print(f"Baseline accuracy on the test set: {baseline_accuracy:.4f}")
print(f"Logistic regression model accuracy on the test set:      {accuracy:.4f}")

# Comparison with baseline
diff = (accuracy - baseline_accuracy) * 100
if diff > 0:
    print(f"The logistic regression model outperforms the baseline by {diff:.2f} percentage points.")
elif diff < 0:
    print(f"The logistic regression model underperforms the baseline by {-diff:.2f} percentage points.")
else:
    print("The logistic regression model matches the baseline accuracy.")


"""
Logistic regression uses the function σ(zi) to map the real-valued zi := wTxi + b to a probability
between 0 and 1. However, this is not the only way to map a real-valued output to a probability.
Probit regression uses Φ(zi), where Φ is the cumulative distribution function of the standard normal
distribution. The model is then:
f(xi) = Φ(wTxi +b)
If you are using JAX, you can use jax.scipy.stats.norm.cdf to implement Φ. If you are using PyTorch,
you can use torch.distributions.Normal.
Implement a probit regression model. Using the same binary cross-entropy loss as before, train
the model on the training data by minimizing the loss with SGD. As before, you may need to
clip the output of the cumulative distribution function to avoid numerical instability. Evaluate the
performance of your model on the test set. How does it compare to the baseline?
"""

# Probit model f(x) = phi(w^T x + b)
normal = torch.distributions.Normal(0.0, 1.0)  # standard normal distribution
 
w_probit = torch.zeros(X_train.shape[1], requires_grad=True)
b_probit = torch.zeros(1, requires_grad=True)
 
def probit_model(X):
    p = normal.cdf(X @ w_probit + b_probit)
    return torch.clamp(p, 1e-7, 1 - 1e-7)    # clip
 
# SGD
optimizer_probit = torch.optim.SGD([w_probit, b_probit], lr=0.01)
 
for epoch in range(epochs):
    perm = torch.randperm(len(X_train))  # shuffle each epoch
    for start in range(0, len(X_train), batch_size):
        idx = perm[start:start + batch_size]
        loss = bce_loss(y_train[idx], probit_model(X_train[idx]))
        optimizer_probit.zero_grad()  # clear the old gradients
        loss.backward()               # compute the gradients
        optimizer_probit.step()       # update the weights
 
# Test the probit model
test_prob_probit = probit_model(X_test)
predictions_probit = (test_prob_probit >= 0.5).float()
accuracy_probit = (predictions_probit == y_test).float().mean().item()
 
print(f"Probit regression model accuracy on the test set:       {accuracy_probit:.4f}")
diff_probit = (accuracy_probit - baseline_accuracy) * 100
if diff_probit > 0:
    print(f"The probit regression model outperforms the baseline by {diff_probit:.2f} percentage points.")
elif diff_probit < 0:
    print(f"The probit regression model underperforms the baseline by {-diff_probit:.2f} percentage points.")
else:
    print("The probit regression model matches the baseline accuracy.")