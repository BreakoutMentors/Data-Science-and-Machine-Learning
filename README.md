# Intro to the Foundations of Deep Learning

![Cat Dog Deep Learning](./images/cat-dog-ml-gif.gif)

### Summary

Hello and welcome to the repo for [Breakout Mentors Machine Learning (ML) Academy](https://breakoutmentors.com/machine-learning-and-artificial-intelligence-academy/)! My name is [Kai Middlebrook](https://www.linkedin.com/in/kaimiddlebrook/). I'm a full-stack data scientist with lots of experience building ML models that solve real-world problems. I'm also the co-creator of the Breakout Mentors ML Academy and the primary author of this repo.  

The motivation for creating this repo and the [Breakout Mentors ML Academy](https://breakoutmentors.com/machine-learning-and-artificial-intelligence-academy/) was to make ML accessible and inclusive to high school students from a wide range of backgrounds (don't worry if you're not in high school, as we've helped plenty of people both young and old learn ML). We subscribe to the idea that you don't need a Ph.D. to benefit from ML, nor do you need to be a mathematical wizard to grasp it. Anyone can learn ML if they have an open mind, a passion for learning, and a desire to solve the problems that matter to them. If that's you, then you've come to the right place!

To help you along in your ML journey, we (Breakout Mentors) adopt a philosophy of teaching the "whole game" and integrate "just-in-time" teaching/learning. Meaning, you as the student get to play with actual code and solve real-world problems using modern data science tools right away instead of being overwhelmed by an endless stream of equations and complex theories. You'll pick up the math and theory as you progress through the material and work on projects. 

As you'll soon find, each lesson contains some new concepts, a practical application of those concepts on real-world data, and is followed by a challenge problem or two for you to practice your understanding. We introduce concepts at a high-level and occasionally opt to omit a few fine-grained details which might have otherwise stunted rather than elevated your understanding, as our primary goal is to develop your intuition for ML concepts. After completing all the lessons, you'll be ready to learn more advanced ML concepts and build tools to solve problems you care about.  

_Course outline:_
We start our journey by learning linear regression - a fundamental algorithm in ML to solve problems where the goal is to predict a numerical output value. We begin by learning how we can reinterpret the familiar equation for a line $y=mx+b$ from an ML perspective. We next introduce logistic and softmax regression, which extends linear regression to classification problems by showing how we can use special functions called activation functions to convert numerical outputs to valid probability scores. We then introduce deep linear neural networks by leveraging our understanding of regression and learning how we can stack many linear layers with activation functions to make learning complex relationships possible. Along the way, you'll discover how to build neural networks from scratch and come to understand how to train and evaluate them. 

Some of the data science tool we will work with include: [Pandas](./basics/Basics_Pandas.ipynb) and [NumPy](./basics/Basics_NumPy.ipynb) (for manipulating data), [Matplotlib's Pyplot](./basics/Basic_Matplotlib_Pyplot.ipynb) (for visualizing data), and [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) (for building and training machine learning models).

**Always Open in Colab** - many of the lessons contain content that may not appear on Github. For the best learning experience, open lessons in Google Colab by click the button 'Open in Colab' when you view a notebook.

## Lesson 0: Introduction to Deep Learning Concepts

**Learning Objective:** learn at the highest level the fundamental terminology and concepts behind deep learning.

- [Video: Intro to Google Colab (~5 min)](https://www.youtube.com/watch?v=inN8seMm7UI)
- [Terminology and Concepts Introduction](./machine_learning/lesson%200%20-%20machine%20learning/Intro_to_Machine_Learning.ipynb)
- [Gradio MNIST](./machine_learning/lesson%200%20-%20machine%20learning/Gradio_MNIST_Tutorial.ipynb)

## Lesson 1: Linear Regression

**Learning Objective:** Understand how to build, train, and evaluate single-layer linear neural networks to solve regression problems.

- [Part 1: Single-variable Linear Regression using Sklearn (Beginner/Friendly)](./machine_learning/lesson%201%20-%20linear%20regression/examples/Linear_Regression_Using_Sklearn.ipynb)
- [Part 2: Single-variable Linear Regression using PyTorch (Intermediate/More Experienced)](./machine_learning/lesson%201%20-%20linear%20regression/examples/From_Linear_Regression_to_Deep_Learning.ipynb)

  - _Pre-lesson:_ [Video: Crash Course - Linear Regression (~13 min)](https://www.youtube.com/watch?v=WWqE7YHR4Jc&t=13s)
  - _Other resources:_ [Pandas Basics](./basics/Basics_Pandas.ipynb), [Sklearn Basics](./basics/Basics_Sklearn.ipynb), [Matplotlib Pyplot Basics](./basics/Basic_Matplotlib_Pyplot.ipynb), [NumPy Basics](./basics/Basics_NumPy.ipynb)
  - _Challenges:_
    - [Predict your school's comedy show attendance (First challenge from the Single-variable Linear Regression Colab lesson)](./machine_learning/lesson%201%20-%20linear%20regression/examples/From_Linear_Regression_to_Deep_Learning.ipynb)

- [Cleaning and Preparing Data](./machine_learning/mini_lessons/Cleaning_Data.ipynb)

  - _Challenges:_
    - [Cleaning and Preparing your own dataset](./machine_learning/mini_lessons/cleaning_and_preparing_your_own_dataset.ipynb)
    - [Single-variable Linear Regression with student's dataset](./machine_learning/lesson%201%20-%20linear%20regression/challenges/simple-linear-regression-2.ipynb)

- [Single-variable Linear Regression as Neural Networks, a Deep Learning Perspective](./machine_learning/lesson%201%20-%20linear%20regression/examples/linear_regression_neural_network.ipynb)

  - _Challenges:_
    - [Revisiting Single-variable Linear Regression with Airbnb data](./machine_learning/lesson%201%20-%20linear%20regression/challenges/challenge_simple_linear_regression_neural_network.ipynb)
    - [Revisiting Single-variable Linear Regression with Student's dataset](./machine_learning/lesson%201%20-%20linear%20regression/challenges/simple-linear-regression-2-revisited.ipynb)

- [Multiple Linear Regression](./machine_learning/lesson%201%20-%20linear%20regression/examples/Multiple_Linear_Regression.ipynb)
  - _Challenges:_
    - [Multiple Linear Regression with Airbnb data](./machine_learning/lesson%201%20-%20linear%20regression/challenges/challenge_2_single_variable_linear_regression_neural_network_.ipynb)
    - [Multiple Linear Regression with student's dataset](./machine_learning/lesson%201%20-%20linear%20regression/challenges/multiple-linear-regression-2.ipynb)

## Lesson 2: Classification with Logistic and Softmax Regression

**Learning Objective:** understand how to build, train, and evaluate single-layer linear neural networks to solve binary and multi-class classification problems.

- [Logistic Regression Classifier](./machine_learning/lesson%202%20-%20logistic%20regression/Classification_Logistic_Regression.ipynb)

  - _Pre-lesson:_ [Video: Logistic Regression (~9 min)](https://www.youtube.com/watch?v=yIYKR4sgzI8)
  - _Other resources:_ [Beyond Accuracy: Precision and Recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c)
  - _Challenges:_
    - [Logistic Regression with Airbnb data](./machine_learning/lesson%202%20-%20logistic%20regression/challenges/challenge_logistic_regression.ipynb)
    - [Logistic Regression with student's dataset](./machine_learning/lesson%202%20-%20logistic%20regression/challenges/logistic-regression-2.ipynb)

- [Softmax Regression Classifier](./machine_learning/lesson%202%20-%20logistic%20regression/Classification_Softmax_Regression.ipynb)

  - _Challenges:_
    - [Softmax Regression with Airbnb data](./machine_learning/lesson%202%20-%20logistic%20regression/challenges/challenge_softmax_regression.ipynb)

- [How to prepare image data for neural networks](./machine_learning/mini_lessons/image_data.ipynb)
  - _Challenges:_
    - [Softmax regression with student's dataset](./machine_learning/mini_lessons/preparing_image_data.ipynb)

## Lesson 3: Image Classification with Deep Neural Networks

**Learning Objective:** understand how to build, train, and evaluate deep linear neural networks to solve regression and classification problems.

- [Introduction to Deep Linear Neural Networks](./machine_learning/lesson%203%20-%20Neural%20Networks/Intro_to_Neural_Networks.ipynb)

  - _Pre-lesson:_ [Video: 3blue1Brown - But What is a Neural Network? (~19 min)](https://www.youtube.com/watch?v=aircAruvnKk)
  - _Other resources:_ [Blog post: Jay Alammar - A visual guide to the basics of Neural Networks (level: intermediate)](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/), [Blog post: Jay Alammar - A visual guide to the math behind the basics of Neural Networks (level: intermediate)](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/)
  - _Challenges:_
    - [Feed Forward Neural Network with Fashion-MNIST](./machine_learning/lesson%203%20-%20Neural%20Networks/challenges/neural_networks_1.ipynb)

- [Using external image datasets in Colab](./machine_learning/mini_lessons/external-datasets-in-colab.ipynb)
  - _Challenges:_
    - [Neural Network with your own image dataset](./machine_learning/lesson%203%20-%20Neural%20Networks/challenges/neural_networks_own_data.ipynb)

- [Introduction to CNNs for Medicine](./machine_learning/lesson%203%20-%20Neural%20Networks/Intro_to_CNNs.ipynb)

## Lesson 4: Building ML Apps

**Learning Objective:** understand how to deploy web apps that use our models.

- [Gradio (High Level)](https://www.gradio.app/)
  - [Digit and Letter Classifier](./machine_learning/lesson%204%20-%20ML%20Apps/Gradio/EMNIST_Gradio_Tutorial.ipynb)
  - Transfer Learning Tutorial
    - [Finetuning a Pretrained Network](./machine_learning/lesson%204%20-%20ML%20Apps/Gradio/Intro_to_Transfer_Learning.ipynb)
    - [Deploy Image Classification App](./machine_learning/lesson%204%20-%20ML%20Apps/Gradio/Pretrained_Model_Gradio_App.ipynb)
  - Transfer Learning Challenge
    - [Finetuning a Pretrained Network](./machine_learning/lesson%204%20-%20ML%20Apps/Gradio/Finetuning_Challenge.ipynb)
    - [Deploy Image Classification App](./machine_learning/lesson%204%20-%20ML%20Apps/Gradio/Transfer_Learning_App_Challenge.ipynb)

## Advanced Deep Learning Resources

**Learning Objective:** continue learning about deep learning!

- [Lecture: MIT Deep Learning Basics](https://www.youtube.com/watch?v=O5xeyoRL95U&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf)
- [Course: Fastai](https://course.fast.ai/)
- [Book: Dive into Deep Learning](https://d2l.ai/index.html)

### Other resources

- [Video: Intro to Google Colab (~5 min)](https://www.youtube.com/watch?v=inN8seMm7UI), [Notebook: Intro to Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=5fCEDCU_qrC0)
- [Pandas Basics](./basics/Basics_Pandas.ipynb)
- [Matplotlib Pyplot Basics](./basics/Basic_Matplotlib_Pyplot.ipynb)
- [Graphing Basics](./basics/Basics_Graphing.ipynb)
- [Sklearn Basics](./basics/Basics_Sklearn.ipynb)
- [NumPy Basics](./basics/Basics_NumPy.ipynb)
- [Beyond Accuracy: Precision and Recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c)
- [Video Series: 3Blue1Brown - Neural Networks (4 videos)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Blog post: Jay Alammar's visual guide to the basics of Neural Networks (level: intermediate)](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)
- [Blog post: Jay Alammar's visual guide to the math behind the basics of Neural Networks (level: intermediate)](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
