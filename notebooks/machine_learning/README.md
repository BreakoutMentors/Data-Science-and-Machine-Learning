# Machine Learning Teaching Guide (for mentors)

### Summary
The goal of this series is to enable your student to apply machine learning
and data science techniques to solve a problem. We take a top-down learning
approach by introducing each topic using an example of the concept applied to a
real-world problem. The student starts by learning a foundational machine learning algorithm:
[linear regression](./lesson%201%20-%20linear%20regression). Next we introduce [logistic regression for classification](), and gradually work up to more complex algorithms. [After the final
lesson](), your student will be able to build a multi-class classification
model and train it on a dataset of their choosing. Along the way, your student
will work with popular data science libraries like [Pandas](../basics/Basic_Pandas_Functions_for_Data_Science.ipynb) for transforming and tidying data, [Matplotlib's Pyplot](../basics/Basic_Matplotlib_Pyplot.ipynb) for visualizing data, and [Sklearn](../basics/Basics_Sklearn.ipynb) and
[Tensorflow]() for building and training machine learning models.

**Lesson format:** lessons can be accessed via Google Colab notebooks. Simply click
on the "open in colab" button in a jupyter notebook from this repo and you and your student
will be able to follow along and interact with the selected lesson. In addition, we provide some
pre-lesson material (i.e., video/short reading) to introduce the topic to your student.
We also include some post-lesson material (generally, short readings) to enable your
student to review and engage more deeply with the concepts. Note, you don't have
to use these resources, they are simply supplemental material meant to help you and your student work through the topics.


## [Lesson 1 - Linear Regression](./lesson%201%20-%20linear%20regression)
**Learning Objective:** fit a linear regression model (single-variable and multiple-variable) to a dataset and
understand what type of data can be used in such models (i.e., numerical data).
- [Simple Linear Regression](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/examples/simple_linear_regression_what_makes_us_happy.ipynb)
  - *Pre-lesson:* [Linear Regression Video from Crash Course](https://www.youtube.com/watch?v=WWqE7YHR4Jc&t=13s)
  - *Post-lesson:* [Data 8: Types of Data](https://docs.google.com/presentation/d/1DIllYGoPGrhpS-2rKyEZOLJQgEcQrE3EqJX0Q-Ys2qA/edit#slide=id.g3f12e5cfb6_0_4), [Data 8: Review of Simple Linear Regression](https://docs.google.com/presentation/d/1TXu2sV9026yzy09uZmTdZSxayKR3ff4yixjbLmbbh-M/edit#slide=id.g30c77890ad_0_0)
  - *Other resources:* [Pandas Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basic_Pandas_Functions_for_Data_Science.ipynb), [Sklearn Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basics_Sklearn.ipynb), [Matplotlib Pyplot Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basic_Matplotlib_Pyplot.ipynb)
  - *Challenges:*
    - [Simple Linear Regression with Pokemon data](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/challenges/challenge_simple_linear_regression.ipynb)
    - [Simple Linear Regression with student's dataset](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/challenges/Single_Variable_Linear_Regression_Challenge_2.ipynb)
- [Multiple Linear Regression](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/examples/Linear_Regression_What_Makes_Us_Happy.ipynb)
  - *Post-lesson:* [Review of Multiple Linear Regression](https://www.scribbr.com/statistics/multiple-linear-regression/)
  - *Challenges:*
    - [Multiple Linear Regression with Pokemon data](https://colab.research.google.com/github/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/challenges/Challenge_Multi_Variable_Linear_Regression.ipynb)
    - [Multiple Linear Regression with student's dataset](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%201%20-%20linear%20regression/challenges/Multiple_Linear_Regression_Challenge_2.ipynb)

[Mini Lesson 1 - Cleaning and Preparing Data](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/mini_lessons/Cleaning_Data.ipynb)
**Learning Objective:** understand how to clean and prepare data for machine learning models and statistical analysis.
- [Cleaning and Preparing Data](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/mini_lessons/Cleaning_Data.ipynb)
  - *Challenges*:
    - [Cleaning and Preparing your own dataset](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/mini_lessons/cleaning_and_preparing_your_own_dataset.ipynb)

## [Lesson 2 - Classification Logistic Regression](https://github.com/krmiddlebrook/intro_to_graphing_in_python/tree/master/notebooks/machine_learning/lesson%202%20-%20logistic%20regression)
**Learning Objective:** fit a logistic regression classification model (single-variable and multiple-variable) to a dataset and
understand what type of data can be used in such models (i.e., numerical and categorical data).
- [Classification: Logistic Regression Classifier](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/machine_learning/lesson%202%20-%20logistic%20regression/Classification_Logistic_Regression.ipynb)
  - *Pre-lesson:* [Video: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
  - *Other resources:* [Beyond Accuracy: Precision and Recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c)
  - *Challenges:*

## [Lesson 3 - Neural Networks](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%203%20-%20Neural%20Networks/Introduction_to_Neural_Networks.ipynb)
**Learning Objective:** use neural networks to solve regression and classification tasks and understand at a the high-level how these networks work.
  - *Pre-lesson:* [Video: 3blue1Brown - But What is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)
  

### Other resources
- [Video: Intro to Google Colab](https://www.youtube.com/watch?v=inN8seMm7UI), [Notebook: Intro to Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=5fCEDCU_qrC0)
- [Pandas Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basic_Pandas_Functions_for_Data_Science.ipynb)
- [Matplotlib Pyplot Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basic_Matplotlib_Pyplot.ipynb)
- [Sklearn Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basics_Sklearn.ipynb)
