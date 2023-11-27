import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import seaborn as sns; sns.set()
from scipy import stats
from sklearn import datasets
import os


# Load file
os.chdir(r"C:\Users\jajko\Downloads")
auto = pd.read_csv('auto.csv')


# Remove missing values
auto = auto.drop(auto[auto.values == '?'].index)
auto = auto.reset_index()


# Convert datatype to quantive
datatypes = {'quant': ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year'],
             'qual': ['origin', 'name']}
quants = auto[datatypes['quant']].astype(np.float_)
auto = pd.concat([quants, auto[datatypes['qual']]], axis=1)




# Own implementation of linear regression
def linear_model(X, Y):
    XTX_inv = np.linalg.inv(X.T @ X)
    XTY = X.T @ Y
    beta = XTX_inv @ XTY
    return beta

def predict(beta, X):
    return X @ beta



intercept = pd.DataFrame({'intercept': np.ones(auto.shape[0])})

X = pd.concat([intercept, auto['horsepower']], axis=1)
Y = auto['mpg']

coefficient = linear_model(X, Y)
y_pred = predict(coefficient, X)

MSE = np.sum(np.square(Y - y_pred)) / Y.size
variance = MSE * (np.linalg.inv(X.T @ X).diagonal())
standard_error = np.sqrt(variance)

t_statistic = coefficient / standard_error
p_values = 2*(1 - stats.t.cdf(X.shape[0], np.abs(t_statistic)))


results = pd.DataFrame({
    'feature': X.columns,
    'coefficient': coefficient,
    'standard error': standard_error,
    't_statistic': t_statistic,
    'p_value': p_values
})
results.set_index('feature')

#print(results)


# By library implementation of linear regression
X = auto['horsepower']
X = sm.add_constant(X)
Y = auto['mpg']
results = sm.OLS(Y, X).fit()
#print(results.summary())


# Prediction and intervals
X_ex = np.array([1, 98])
Y_ex = predict(coefficient, X_ex)
#print(np.round(Y_ex, 3), 'mpg')

model_min = results.conf_int(alpha=0.05)[0]
model_max = results.conf_int(alpha=0.05)[1]
confidence_interval = [predict(model_min, X_ex), predict(model_max, X_ex)]
#print(confidence_interval)


# Display of linear regression line
df = pd.concat([auto['horsepower'], auto['mpg']], axis=1)
ax = sns.scatterplot(x='horsepower', y='mpg', data=df)
ax.plot(auto['horsepower'], y_pred)
plt.show()


# Diagnatiostic plots
def lm_stats(X, Y, y_pred):
    try:
        Y.shape[1] == 1
        Y = Y.iloc[:,0]
    except:
        pass
    Y = np.array(Y)

    residuals = np.array(Y - y_pred)

    # Hat Matrix
    H = np.array(X @ np.linalg.inv(X.T @ X)) @ X.T

    # Leverage
    h_ii = H.diagonal()

    # Estimate variance
    sigma_est = []
    for i in range(X.shape[0]):
        # Exclude ith observation from estimation of variance
        external_residuals = np.delete(residuals, i)

comp = pd.concat([Y, y_pred], axis=1)
print(comp)