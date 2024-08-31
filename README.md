# California Housing Price Prediction

## Overview

This project is a demonstration of my ability to apply linear regression to real-world data. Using the California Housing dataset, I built and evaluated a predictive model that estimates housing prices based on a variety of demographic and geographic features. This project highlights my skills in data preprocessing, visualization, model training, and evaluation using Python and popular data science libraries.

## Key Features

- **Data Exploration:** Understanding the structure and characteristics of the dataset.
- **Data Visualization:** Using Seaborn to visualize relationships between different features.
- **Modeling:** Implementing a linear regression model using scikit-learn.
- **Model Evaluation:** Assessing the model's performance using metrics such as MAE, MSE, RMSE, and R².

## Skills Demonstrated

- **Python Programming:** Effective use of Python for data analysis and modeling.
- **Data Wrangling:** Cleaning and preparing the dataset for analysis.
- **Statistical Analysis:** Implementing and interpreting linear regression.
- **Data Visualization:** Creating insightful visualizations to understand data relationships.
- **Machine Learning:** Training and evaluating a linear regression model.


## Project Details

### Dataset

The dataset used is the California Housing dataset, which is derived from the 1990 U.S. Census. It contains information on housing prices and various features across different districts in California. The target variable is the median house value.

- **Number of Instances:** 20,640
- **Number of Attributes:** 8 numerical features + 1 target variable

### Tools and Libraries

- **Python**
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computations.
- **Matplotlib & Seaborn:** For data visualization.
- **Scikit-learn:** For machine learning, including model training and evaluation.
- **Plotly:** For interactive data visualization.

### Key Steps in the Project

1. **Data Loading and Exploration**
    - Loaded the California Housing dataset using scikit-learn.
    - Conducted initial exploration using .info() and .describe() to understand the dataset's structure and statistical summary.
      ````
      from sklearn.datasets import fetch_california_housing
      data = fetch_california_housing()

      ca_house_df = pd.DataFrame(data.data, columns=data.feature_names)
      ca_house_df['Target'] = data.target
      ````
      
2. **Data Visualization**

    - Used Seaborn's `pairplot` to visualize the relationships between different features and the target variable.

      ````
      sns.pairplot(data=ca_house_df, plot_kws={'color': 'gray'}, diag_kws={'color': 'darkblue', 'fill': True})
      ````
      
3. **Data Preparation**

    - Selected relevant features and split the data into training and testing sets.

      ````
      from sklearn.model_selection import train_test_split
      
      X = ca_house_df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Latitude', 'Longitude']]
      y = ca_house_df["Target"]
      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
      ````
      
4. Model Training

    - Trained a LinearRegression model using the training data.

      ````
      from sklearn.linear_model import LinearRegression
      lm = LinearRegression()
      lm.fit(X=X_train, y=y_train)
      ````
5. Model Evaluation

    - Evaluated the model using several metrics and visualized the residuals.
      ````
      from sklearn import metrics
      
      predictions = lm.predict(X_test)
      
      print('MAE:', metrics.mean_absolute_error(y_test, predictions).round(2))
      print('MSE:', metrics.mean_squared_error(y_test, predictions).round(2))
      print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)).round(2))
      print(f"R^2: {round(metrics.r2_score(y_test, predictions), 4)}")
      ````

### Results and Interpretation
The model was able to predict house prices with the following performance:

- **Root Mean Squared Error (RMSE):** 0.73
- **R² Score:** 0.6056
  
These metrics indicate a reasonable fit, with the most influential features being MedInc (Median income) and AveBedrms (Average number of bedrooms).

### Conclusion
This project demonstrates my ability to handle a complete machine learning pipeline, from data preprocessing to model evaluation. The linear regression model built in this project provides a solid foundation for predicting housing prices, with potential for further refinement and tuning.

### Future Enhancements
- **Feature Engineering:** Adding new features or transforming existing ones to improve model performance.
- **Advanced Models:** Exploring more complex models like Random Forest or Gradient Boosting for better accuracy.
- **Hyperparameter Tuning:** Using grid search or random search to optimize model parameters.
### License
This project is licensed under the MIT License. See the LICENSE file for details.
