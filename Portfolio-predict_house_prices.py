# %% [markdown]
# ## California Housing dataset <br>
# **Data Set Characteristics:** <br> 
# :Number of Instances: 20640 <br>
# :Number of Attributes: 8 numeric, predictive attributes and the target <br>
# :Attribute Information: <br>    
# - MedInc        median income in block group <br>
# - HouseAge      median house age in block group <br>    
# - AveRooms      average number of rooms per household   <br>
# - AveBedrms     average number of bedrooms per household    <br>
# - Population    block group population    <br>
# - AveOccup      average number of household members    <br>
# - Latitude      block group latitude    <br>
# - Longitude     block group longitude <br> <br>
# :Missing Attribute Values: None <br> 
# This dataset was obtained from the StatLib repository. <br>
# https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html <br>
# The **target** variable is the median house value for California districts,<br> expressed in **hundreds of thousands of dollars ($100,000).**  <br> <br>
# This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). <br> <br> A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts. <br> <br> It can be downloaded /loaded using the <br>
# :func: `sklearn.datasets.fetch_california_housing` function.<br> 
# ..rubric:: References<br><br>
# - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,  Statistics and Probability Letters, 33 (1997) 291-297'

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# %% [markdown]
# ## Load Data from sklearn and make a DataFrame

# %%
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

ca_house_df = pd.DataFrame(data.data, columns=data.feature_names)
ca_house_df['Target'] = data.target
ca_house_df.head(2)

# %% [markdown]
# ### Short exploration of the data via .info() and .describe()

# %%
ca_house_df.info()

# %%
ca_house_df.describe()

# %% [markdown]
# ## Visulize Data via Pairplot()

# %%
sns.pairplot(data=ca_house_df, 
             plot_kws={'color': 'gray'}, 
             diag_kws={'color': 'darkblue', 'fill': True})

# %% [markdown]
# From this we can eliminate the following columns for the 
# - AveOccup
# 
# as this gives only a line, with some outliers

# %% [markdown]
# ## Training and Testing Data
# *.columns helps with getting the X and y values*

# %%
ca_house_df.columns

# %% [markdown]
# **X: Numerical features of houses.** <br>
# **y: Target value.**

# %%
X = ca_house_df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Latitude', 'Longitude']]
y = ca_house_df["Target"]

# %% [markdown]
# **Splitting Data into training and testing sets**

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %% [markdown]
# **Instance of a LinearRegression() model named lm**

# %%
lm = LinearRegression()

# %% [markdown]
# **Train/fit lm on the training data.**

# %%
lm.fit(X=X_train, y=y_train)

# %% [markdown]
# **Print out the coefficients of the model**

# %%
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df

# %% [markdown]
# From this it can be seen, that <br>
# - MedInc        median income in block group <br>
# - AveBedrms     average number of bedrooms per household    
# 
# have the most influence on the target price. **Remember target is expressed in values of $100,000 instances** 

# %% [markdown]
# ## Predicting Test Data

# %%
predictions = lm.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, edgecolors="black")

# %% [markdown]
# Import plotly.express to create an interactive plot, for further analyzing of the data

# %%
import plotly.express as px

# %%
px.scatter(x=y_test, y=predictions)

# %% [markdown]
# ## Evaluating the Model

# %%
print('MAE:', metrics.mean_absolute_error(y_test, predictions).round(2))
print('MSE:', metrics.mean_squared_error(y_test, predictions).round(2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)).round(2))
print(f"R^2: {round(metrics.r2_score(y_test, predictions), 4)}")


# %% [markdown]
# ## Plot Residuals

# %%
plt.figure(figsize=(12, 4))
sns.histplot((y_test-predictions), bins=100, kde=True, color="gray");

# %%
px.histogram((y_test-predictions))

# %% [markdown]
# ## Conclusion
# 
# The Data seems somewhat compatiple with a linear regression. <br> <br>
# Highest influence on the price:
# - MedInc        median income in block group <br>
# - AveBedrms     average number of bedrooms per household <br>
# 
# The Metrics:
# - Root Mean Squared Error: 0.73
# - $R^2$ score: 0.6056
# 
# **Result: An acceptable prediction of house prices.**


