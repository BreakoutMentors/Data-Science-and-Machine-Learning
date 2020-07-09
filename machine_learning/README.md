# Data Science & Machine Learning

### Summary
Our goal is to enable students to apply machine learning and data science techniques to solve real-world problems. We use a *top-down learning* method: each lesson includes some new material that is presented through a practical application on real-world data.

We start our journey by learning linear regression - a fundamental algorithm in machine learning. Next, we introduce logistic regression - an extension of linear regression - to solve classification problems. After mastering these foundations, we are ready to explore neural networks and deep learning. In this section, we walk you through the process of building deep neural networks from scratch, showing that these networks can achieve remarkable results on different regression and classification tasks. By the final lesson, you will be able to build a feed forward neural network that can recognize 10 types of handwritten digits.

Throughout these lessons, you will learn how to work with popular data science libraries like [Pandas](../basics/Basic_Pandas_Functions_for_Data_Science.ipynb) plus [NumPy](../basics/Basics_NumPy.ipynb) (for manipulating data), [Matplotlib's Pyplot](../basics/Basic_Matplotlib_Pyplot.ipynb) (for visualizing data), and [Sklearn](../basics/Basics_Sklearn.ipynb) (for building and training machine learning models).

**Always Open in Colab** - many of the lessons contain images that do not show on Github. So always click the button to 'Open in Colab'

## Lesson 0: Introduction to Machine Learning
**Learning Objective:** learn the fundamental terminology for data science and machine learning.   
- [Terminology Introduction](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%200%20-%20machine%20learning/Intro_to_Machine_Learning.ipynb)


## Lesson 1: Linear Regression
**Learning Objective:** fit a linear regression model to a dataset and understand what type of data can be used in such models (i.e., numerical data).
- [Single-variable Linear Regression](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/examples/simple-linear-regression.ipynb)
  - *Pre-lesson:* [Linear Regression Video from Crash Course (~13 min)](https://www.youtube.com/watch?v=WWqE7YHR4Jc&t=13s) (don't worry about any intimidating math, you don't need to understand it in order to use the machine learning algorithm)
  - *Post-lesson:* [Data 8: Types of Data](https://docs.google.com/presentation/d/1DIllYGoPGrhpS-2rKyEZOLJQgEcQrE3EqJX0Q-Ys2qA/edit#slide=id.g3f12e5cfb6_0_4), [Data 8: Review of Single-variable Linear Regression](https://docs.google.com/presentation/d/1TXu2sV9026yzy09uZmTdZSxayKR3ff4yixjbLmbbh-M/edit#slide=id.g30c77890ad_0_0)
  - *Other resources:* [Pandas Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basic_Pandas_Functions_for_Data_Science.ipynb), [Sklearn Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basics_Sklearn.ipynb), [Matplotlib Pyplot Basics](https://github.com/krmiddlebrook/intro_to_graphing_in_python/blob/master/notebooks/Basic_Matplotlib_Pyplot.ipynb), [NumPy Basics](../basics/Basics_NumPy.ipynb)
  - *Challenges:*
    - [Single-variable Linear Regression with Pokemon data](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/challenges/simple-linear-regression.ipynb)

- [Mini Lesson: Cleaning and Preparing Data](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/mini_lessons/Cleaning_Data.ipynb)

  **Learning Objective:** understand how to clean and prepare data for machine learning models and statistical analysis.
    - *Challenges:*
      - [Cleaning and Preparing your own dataset](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/mini_lessons/cleaning_and_preparing_your_own_dataset.ipynb)
      - [Single-variable Linear Regression with student's dataset](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/challenges/simple-linear-regression-2.ipynb)
- [Single-variable Linear Regression using a Neural Network - Deep Dive](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/examples/linear-regression-deep-dive.ipynb)
  - *Challenges:*
    - [Revisiting Single-variable Linear Regression with Pokemon](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/challenges/revisting-simple-linear-regression-pokemon.ipynb)
    - [Revisiting Single-variable Linear Regression with Student's dataset](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/challenges/simple-linear-regression-2-revisited.ipynb)

- [Multiple Linear Regression](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/examples/multiple-linear-regression.ipynb)
  - *Post-lesson:* [Review of Multiple Linear Regression](https://www.scribbr.com/statistics/multiple-linear-regression/)
  - *Challenges:*
    - [Multiple Linear Regression with Pokemon data](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/challenges/multiple-linear-regression-pokemon.ipynb)
    - [Multiple Linear Regression with student's dataset](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%201%20-%20linear%20regression/challenges/multiple-linear-regression-2.ipynb)

## Lesson 2: Classification with Logistic and Softmax Regression
**Learning Objective:** fit a logistic regression classification model (single-variable and multiple-variable) to a dataset and
understand what type of data can be used in such models (i.e., numerical and categorical data).
- [Logistic Regression Classifier](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%202%20-%20logistic%20regression/logistic-regression.ipynb)
  - *Pre-lesson:* [Video: Logistic Regression (~9 min)](https://www.youtube.com/watch?v=yIYKR4sgzI8)
  - *Other resources:* [Beyond Accuracy: Precision and Recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c)
  - *Challenges:*
    - [Logistic Regression with Pokemon](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%202%20-%20logistic%20regression/challenges/logistic-regression-pokemon.ipynb)
    - [Logistic Regression with student's dataset](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%202%20-%20logistic%20regression/challenges/logistic-regression-2.ipynb)
- [Softmax Regression Classifier](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%202%20-%20logistic%20regression/softmax-regression.ipynb)
  - *Challenges:*
    - [Softmax Regression with Pokemon](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%202%20-%20logistic%20regression/challenges/softmax-regression-pokemon.ipynb)

- [Mini Lesson: Finding and preparing image data](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/mini_lessons/image_data.ipynb)
  - [Softmax regression with student's dataset](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%202%20-%20logistic%20regression/challenges/softmax-regression-2.ipynb)

## Lesson 3: Neural Networks and Deep Learning
**Learning Objective:** build neural networks to solve regression and classification tasks and understand (at a high level) how these networks work.
- [Introduction to Neural Networks](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%203%20-%20Neural%20Networks/intro-to-neural-networks.ipynb)
  - *Pre-lesson:* [Video: 3blue1Brown - But What is a Neural Network? (~19 min)](https://www.youtube.com/watch?v=aircAruvnKk)
  - *Other resources:* [Blog post: Jay Alammar's visual guide to the basics of Neural Networks (level: intermediate)](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/), [Blog post: Jay Alammar's visual guide to the math behind the basics of Neural Networks (level: intermediate)](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/), [Blog post: Jordi Pons - Multi-Layer Perceptrons (level: intermediate/advanced)](http://www.jordipons.me/apps/teaching-materials/mlp.html)
  - *Challenges:*
    - [Feed Forward Neural Network with Fashion-MNIST](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%203%20-%20Neural%20Networks/challenges/neural_networks_1.ipynb)

- [Mini lesson: Using external image datasets in Colab](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/mini_lessons/external-datasets-in-colab.ipynb)
  - *Challenges:*
    - [Neural Network with your own image dataset](https://github.com/BreakoutMentors/Data-Science-and-Machine-Learning/blob/master/machine_learning/lesson%203%20-%20Neural%20Networks/challenges/neural_networks_own_data.ipynb)

## Advanced Deep Learning Resources
**Learning Objective:** continue learning about deep learning with these excellent resources!
- [Lecture: MIT Deep Learning Basics](https://www.youtube.com/watch?v=O5xeyoRL95U&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf)
- [Book: Dive into Deep Learning](https://d2l.ai/index.html)
- [Course: Fastai](https://course.fast.ai/)

### Other resources
- [Video: Intro to Google Colab (~5 min)](https://www.youtube.com/watch?v=inN8seMm7UI), [Notebook: Intro to Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=5fCEDCU_qrC0)
- [Pandas Basics](../basics/Basic_Pandas_Functions_for_Data_Science.ipynb)
- [Matplotlib Pyplot Basics](../basics/Basic_Matplotlib_Pyplot.ipynb)
- [Sklearn Basics](../basics/Basics_Sklearn.ipynb)
- [NumPy Basics](../basics/Basics_NumPy.ipynb)
