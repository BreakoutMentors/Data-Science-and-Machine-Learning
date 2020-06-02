# Linear Regression

## Single-Variable Linear Regression
[Challenge 1: Single-Variable Linear Regression with Pokemon data](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/challenges/challenge_simple_linear_regression.ipynb)
1. Load the Pokemon dataset into a pandas dataframe.
2. List the first 5 rows of the dataframe.
3. Show the top 5 Pokemon based on the "Total" column.
4. Plot the correlation matrix heatmap between all numeric columns.
5. Add annotations to the correlation heatmap.
6. Make a single variable linear regression model between the "Total" column and the "Attack" column.
- Hint: If you get stuck see the [single-variable linear regression what makes us happy example](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/examples/simple_linear_regression_what_makes_us_happy.ipynb)
<br><br>

[Challenge 2: Single-Variable Linear Regression with your own dataset](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/challenges/Single_Variable_Linear_Regression_Challenge_2.ipynb)
1. Find a dataset you're interested in. Hint: the dataset should have several
numeric columns. Here are some places to look for data: [kaggle](https://www.kaggle.com/datasets), [Google Dataset Search](https://datasetsearch.research.google.com/), [awesome-datasets Github](https://github.com/awesomedata/awesome-public-datasets#socialsciences).
2. Explore the dataset. How many numeric columns are there? What is your y
variable? What other information in the data might be useful?
3. Plot the correlation matrix heatmap between all numeric columns.
4. Add annotations to the correlation heatmap.
5. Pick an x variable from the numeric columns.
6. Make a single variable linear regression model between the y variable and the
 x variable.
- Hint: If you get stuck see the [single-variable linear regression what makes us happy example](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/examples/simple_linear_regression_what_makes_us_happy.ipynb)
<br><br>


## Multiple Linear Regression
[Challenge 1: Multiple Linear Regression with Pokemon data](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/challenges/Challenge_Multi_Variable_Linear_Regression.ipynb)
1. Load the Pokemon dataset into a pandas dataframe.
2. Plot a bar chart of the counts of each type in the "Type 1" column.
3. Make a multi-variable linear regression model between the "Total" column and all the numeric columns.
4. Create a dataframe containing the model coefficients for each numeric column
5. Calculate the R^2 (or the correlation coefficient) between the "Total" column (y variable) and the numeric columns (x variables).
6. Create a dataframe containing two columns the actual "Total" values from the pokemon dataframe and the predicted "Total" values from the model.
7. Make a multi-bar bar chart to show the difference between the first 25 rows of the actual and the predicted Total values.
8. Evaluate the performance of the model by calculating the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) between the actual Total values and the predicted total values. Hint: use Sklearns metrics module to calculate these.  
- Hint: If you get stuck see the [multi-variable linear regression what makes us happy example](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/examples/Linear_Regression_What_Makes_Us_Happy.ipynb)
<br><br>

[Challenge 2: Multiple Linear Regression with your own dataset](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/challenges/Multiple_Linear_Regression_Challenge_2.ipynb)
1. Find a dataset you're interested in. Hint: the dataset should have several
numeric columns. Here are some places to look for data: [kaggle](https://www.kaggle.com/datasets), [Google Dataset Search](https://datasetsearch.research.google.com/), [awesome-datasets Github](https://github.com/awesomedata/awesome-public-datasets#socialsciences).
2. Explore the dataset. How many numeric columns are there? What is your y
variable? What other information in the data might be useful?
3. Plot the correlation matrix heatmap between all numeric columns.
4. Add annotations to the correlation heatmap.
5. Pick your x variables from the numeric columns.
6. Make a multiple linear regression model between the y variable and the
 x variable.
7. What does the model tell us? Write your conclusion.
- Hint: If you get stuck see the [multi-variable linear regression what makes us happy example](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/examples/Linear_Regression_What_Makes_Us_Happy.ipynb)
<br><br>
