"""
One common justification for using square error loss in machine learning is that the residuals of the
fitted model should be (approximately) normal. This justification can be approximately tested by fitting
a model to the data and then testing the residuals for normality! To do that, you are asked to fit several
ML models to the Average Localization Error (ALE) dataset and run a statistical test to see if the
residuals are approximately normally distributed. You should predict ale using the first 4 variables (you
can ignore sd ale). You can find different regression models in the sklearn package. Try whatever you
like (for example, linear model, svm, decision tree, or neural network). To test the normality, you are
allowed to use any test method. One recommendation is to draw a qqplot and run normaltest in the
scipy package
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
matplotlib.use("TkAgg")
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


from pathlib import Path
df = pd.read_csv(Path(__file__).parent / "mcs_ds_edited_iter_shuffled.csv")
X = df.iloc[:, :4].values  
y = df["ale"].values
# Check data shape and feature names
print(f"n = {len(df)} rows, features = {list(df.columns[:4])}\n")

# Fit four models: linear regression, SVM, decision tree, and neural network (MLP).
models = {
    "Linear Regression": LinearRegression(),
    "SVM (SVR, RBF kernel)": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.03)),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=0),
    "Neural Network (MLP)": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=5000,
                     alpha=1e-3, random_state=0)),
}
 
# 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=0)
 
# Fit each model, compute residuals, and run D'Agostino-Pearson normaltest
residuals, pvalues = {}, {}
for name, model in models.items():
    y_hat = cross_val_predict(model, X, y, cv=cv)
    r = y - y_hat
    residuals[name] = r
 

    stat, p = stats.normaltest(r)
    pvalues[name] = p
    verdict = "normal" if p > 0.05 else "NOT normal"
    print(f"{name:22s} K2 = {stat:7.3f}   p = {p:.4f}  ->  {verdict}") # K2 measures how far the residuals are from normal.
 
# Q-Q plots of the residuals, one per model
# If data points are closed to the red line, it means the residuals are approximately normal.
fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
for ax, (name, r) in zip(axes, residuals.items()):
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title(f"{name}\nnormaltest p = {pvalues[name]:.3g}")
plt.tight_layout()
plt.show()