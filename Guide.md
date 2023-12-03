<div align="center">
  <img src="https://miro.medium.com/max/1352/0*bdcBLuhT3N22-KEi">
</div>


# Table of contents
1. [Introduction](#introduction)
    1. [Disclaimer](#disclaimer)
    2. [Pre-Requisite AI Concepts](#ai_concepts)
2. [AI Ecosystem](#aieco)
    1. [Anaconda](#aiecoana)
    1. [Numpy](#aieconumpy)
    2. [Tensorflow](#aiecotf)
        1. [Keras](#aiecokeras)
        2. [scikit-learn](#aiecoscik)
    3. [Pandas](#aiecopandas)
    4. [Jupyter notebook](#jn)
3. [Installations](#ins)
4. [Tensorflow Basics: Neural Network Model](#tf1)
    1. [Example](tf1example)
    2. [Disecting the Example](#tf1disect)
        1. [Model, Layers and Nodes](#tf1mln)
    3. [Saving and loading trained Model](#tf1sltm)
5. [Loss & Final Activation Function](#la)
6. [Optimize Function](#of)
7. [Avoiding Underfitting and Overfitting](#auo)
8. [Neural Network Architecture Types](#nnat)
9. [Converting Real World Problem To Mathematical Problem](#mp)
10. [Scenarios](#scenarios)
11. [Deploying Model](#dm)
    1. [Integrating via ML .NET](#dmmlnet)
    2. [Exposing Model via Python REST API](#dmemrest)
    3. [Google Cloud Compute: Serverless function](#dmgccsf)
12. [Cheat Sheets](#cheatsheets)

# Introduction <a name="introduction"></a>

# Disclaimer <a name="disclaimer"></a>
I have made this guide/knowledgebase not to get specific usecase information, but instead as a guide that will point into specific direction for searching and learning a topic or scenario. Furthermore, it is not based on general preference of technologies within accepted AI ecosystem, rather guide is based on my personal preference of AI ecosystem. It is particularly Python,Tensorflow and Deep Learning centric approach. This guide does **NOT** specifically deals with Machine Learning except where it overlap with Deep Learning. 

this approach is because in my limited knowledge, input to output ratio of learning/professional advantage is maximum in this way.

Some information provided here may not be accurate, as I am still learning. If you encounter any such information please fix the doc.

I have written the doc in a style, that focuses use of Math centric language with tech centric language. Particularly because DL becomes far easier to do and research if we are able to map Mathimatical concepts on to Technology we use and vice-versa.

# Pre-Requisite AI Concepts <a name="ai_concepts"></a>
I assume anyone reading this doc have basic, nothing too deep, bare-minimum understanding of following AI/Math concepts, 
1. [Neural Network](https://en.wikipedia.org/wiki/Neural_network).
2. Difference between Machine Learning and Deep Learning. [(Feature engineering vs numerous layers)](https://hackernoon.com/deep-learning-vs-machine-learning-a-simple-explanation-47405b3eef08).
3. [What](https://www.mathsisfun.com/sets/function.html) is a [function](https://www.youtube.com/watch?v=GY6Q2f2kvY0). Any correlation in data can be represented mathematically as function, and yes even presence of your face in an image can be mathematically represented as a function. All [prediction/fitting model are nothing but a function](https://math.stackexchange.com/questions/2005065/create-a-function-based-on-data-set/2005200) that represent entire dataset. 
4. Traning of model, is actually model trying to [find function](https://www.datasciencecentral.com/profiles/blogs/6448529:BlogPost:570512) of a data or more specifically fitting function over data.
5. High-[Dimensional](https://en.wikipedia.org/wiki/Dimension) Space/[Tensors (square,cube,tesseract)](https://www.heatonresearch.com/2017/07/30/tensors.html)
6. Simple functions can [compose more complicated functions](https://en.wikipedia.org/wiki/Function_composition). If you [distort](https://en.wikipedia.org/wiki/Transformation_(function)) fabric of [space enough](https://en.wikipedia.org/wiki/Curvilinear_coordinates), there is [high chance](https://en.wikipedia.org/wiki/Universal_approximation_theorem) you might get to the desired complicated function just by using simple functions (for details see section 2.3.5 of Deep Learning with Python - Manning).
7. [Matrix](https://math.stackexchange.com/a/2782730) is n-Dimensional Array representing a data point in high dimensional space, which may or may not be in transformed space. You can [distort the space/data point](https://en.wikipedia.org/wiki/Transformation_matrix) using [Properties](https://en.wikipedia.org/wiki/Matrix_(mathematics)#Basic_operations) of Matrix [Operations.](https://ltcconline.net/greenl/courses/203/MatricesApps/MatrixOpsProps.htm)
8. Machine Learning/Deep Learning is nothing except finding the function that represents the given data, if inputs and outputs are in numerical values. these inputs may be matrix or scalar values.
9. Function can be determined of any non-numeric data, if it is converted to numeric form without destrying it's existing [correlations](https://en.wikipedia.org/wiki/Correlation_and_dependence).
10. Function of Model, problems of [Under fit & Overfit](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765).

# AI Ecosystem <a name="aieco"></a>
Python is the language used with pip as it's package manager, following is the rough hierarchy of tools,libraries and frameworks used. 

**NOTE:** see full-screen raw text of this doc on notepad if following table is not visible in rendered mode.

+-------------------------+---------------------------------------------------------------------------------------------------------------+
|                         |                                Keras (collection of common utilities for ML/DL)                               |
+-------------------------+------------------------------------------------------------+--------------------------------------------------+
| Pandas (data wrangling) | Tensorflow (handle mathematical workflows including ML/DL) | scikit (handle Machine Learning pieces/workflow) |
+-------------------------+------------------------------------------------------------+--------------------------------------------------+
|                                                     NumPY (data structure handling)                                                     |
+-----------------------------------------------------------------------------------------------------------------------------------------+
|                                                      Anaconda (virtual environment)                                                     |
+-----------------------------------------------------------------------------------------------------------------------------------------+
|                                                            Python (language)                                                            |
+-----------------------------------------------------------------------------------------------------------------------------------------+



## Anaconda <a name="aiecoana"></a>
Python have it's [own virtual environment manager](https://docs.python.org/3/tutorial/venv.html), but instead [Anaconda](https://www.anaconda.com/) is used for simplicity sake. In anaconda we can create and activate 'virtual environment' which provides isolation of installed packages. Which means on a single computer, different version of a package can be installed in different environment, therefore if an application requires certain version of package it remains isolated from requirements of other apps.

## Numpy <a name="aieconumpy"></a>
Numpy is a python library, that is extreamely fast when computing on large data, all operations it carries out happen in single datastructure/type/class called ndarray. Since all major framework(e.g tensorflow,pandas,keras) are built on this library, their APIs are compatible with this type and therefore are inter-compatible. Due to this in 95% cases there is no type casting/transforming.

## Tensorflow <a name="aiecotf"></a>
Tensorflow is production-friendly DL frameowrk built over Numpy. Although its default data type/class is 'tensor' it can use ndarray almost everywhere. whereever it can not use 'tf.convert_to_tensor' function. It provides powerful apis through which you can create and train complex Neural Networks in less than 10 lines.

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

### Keras <a name="aiecokeras"></a>
Keras is a python library originally created in a open source ML project, for creation of layers,nodes etc in a model i.e common utilities. Google liked this library so much that now it is also available as a module inside Tensorflow framework.
### Scikit-learn <a name="aiecoscik"></a>
[Scikit-learn](https://scikit-learn.org/stable/) is a python library focused on ML (not DL), It contain large library of various implementations of learning algorithms. These algorithms can be used with Tensorflow as well.

## Pandas <a name="aiecopandas"></a>
[Pandas](https://pandas.pydata.org/about/index.html) is a python library built over numpy focused on Data wrangling and data cleaning. It's default datatype is series/dataframe is also compatible with ndarray.

## Jupyter notebook <a name="jn"></a>
[Jupyter Notebook](https://jupyter.org/) provides you with an easy-to-use, interactive data science environment across many programming languages that doesn't only work as an IDE, but also as a presentation or education tool.

# Installations <a name="ins"></a>
Install [Python 3.7](https://www.python.org/downloads/) followed by [Anaconda](https://www.anaconda.com/distribution/#windows).

Open Anaconda Navigator, and create a new python environment called **ML**. 

There are 2 ways to setup environment,
## Method 1: Automatic Setup
Go to folder **DependencyManagement** in this repository, and run **setupEnv.bat** file.

## Method 2: Manual Setup

Once created, open Anaconda Command Prompt and change directory to your folder.
```
cd C:\DEV\
```
Once inside folder, activate the environment.
```
activate ML
```
when the environment is activated, install Tensorflow.
```
pip install tensorflow==2.0
```
Similarly install jupyter and pandas and scikit-learn
```
pip install jupyter
pip install pandas
pip install scikit-learn  
```

# Tensorflow Basics: Neural Network Model <a name="tf1"></a>
There are 5 steps to create neural network and teach it,



1) Declare structure of neural network. Sequential take array of keras.layers as parameter.

```
import tensorflow as tf

model = tf.keras.Sequential()
```

2) Compile the model, provide an optimizer function and a loss function.
 
 ```
 model.compile(optimizer='sdg',loss='mean_squared_error')
 ```

3) Deep Learning, here learning actually happens! both input and outputs are ndarrays and epochs is a number

```
model.fit(input,output,epochs)
```

4) Now use model to predict output value.

```
model.predict([10.0])
```

5) Note, for development purpose you might want to use evaluate function instead of predict, as it tells you loss,accuracy etc as well.
```
model.evaluate()
```
6) Bonus: you can view details of model as well
```
model.summary()
```
## Example <a name="tf1example"></a>
Here is a simplist example of machine learning, suppose you have percentage data of Gdp/economic growth and industrial output growth of the country.

If  Gdp/economic growth is **x**, industrial output growth is **y**, and the secret unknown function between them is **y=2x-1**

Therefore suppose data is defined as ***xs*** and ***ys*** in this code, and we want to predict what will be growth in industrial output if GDP growth is at 10.
```
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
```
As promised Learning and Prediction took less than 10 lines.

## Disecting the Example  <a name="tf1disect"></a>
The scenario we have dealt with above, where input is a number and output is calculated to be a number as well. In Maths, this is called [Linear Regression.](https://en.wikipedia.org/wiki/Linear_regression). Since our input data (**xs**) is 1-Dimensional array it is called [Simple Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression), if our input had higher dimensions than 1 it would have been called [Multi-Variate Linear Regression](https://en.wikipedia.org/wiki/General_linear_model)

<div align="center">
<h1>Simple Linear Regression (1-Variable/Dimension)</h1>
  <img  src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg">
</div>
<div align="center">
<h1>Multi-Variate Linear Regression (2-Variables/Dimensions)</h1>
  <img  src="https://www.dataquest.io/wp-content/uploads/2018/05/linear-reg-r.jpg">
</div>


### Model, Layers and Nodes <a name="tf1mln"></a>

In following line, a neural network is created that have linear stack of layers. this type of neural network is also called [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) because data always move in a single direction.
<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/5/54/Feed_forward_neural_net.gif">
</div>

```
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
```
In above line, we have created following neural network. *keras.layers.Dense* represent center hidden layer, with units defining number of nodes in that layer. It should be noted neural network diagrams can be deceptive to an untrained eye, each node always output a single value and therefore output shape of a layer is always same as number of nodes in the layer, it is a single output of the node that then goes out to single or multiple nodes. When defining a layer, input_shape parameter should never be given except when you are defining first layer of the model as input_shape represent the input layer. 

(Above statement aboout input_shape is [not entirely true](https://keras.io/getting-started/sequential-model-guide/), but for our purpose we should understand it in this way).
<div align="center">
  <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAkYAAADYCAYAAAAOLix2AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAA21SURBVHhe7d3/q57zHwdw/5FfRH5SylEokjJ+4xfJZhgrDBHLl6JYicjGbIiFfJmFzWxTfhE/oEmWYbLFjLQae3963V7vebvc9/mcbefbde3xqCvn+n6dcz/f1/287/ucOa0AADCiGAEAJMUIACApRgAASTECAEiKEQBAUowAAJJiBACQFCMAgKQYAQAkxQgAIClGAABJMQIASIoRAEBSjAAAkmIEAJAUIwCApBgBACTFCAAgKUYAAEkxAgBIihEAQFKMAACSYgQAkBQjAICkGAEAJMUIACApRgAASTECAEiKEQBAUowAAJJiBACQFCMAgKQYAQAkxQgAIClGAABJMQIASIoRAEBSjAAAkmIEAJAUIwCApBgBACTFCAAgKUYAAEkxAgBIihEAQFKMAACSYgQAkBQjAICkGAEAJMUIACApRgAASTECAEiKEQBAUowAAJJiBACQFCMAgKQYAQAkxQgAIClGAABJMQIASIoRAEBSjAAAkmIEAJAUIwCApBgBACTFCAAgKUYAAEkxAgBIihEAQFKMAACSYgQAkBQjAICkGAEAJMUIACApRgAASTECAEiKEQBAUowAAJJiBACQFCMAgKQYAQAkxQgAIClGAABJMQIASIoRAEBSjAAAkmIEAJAUIwCApBgBACTFCAAgKUYAAEkxAgBIihEAQFKMAACSYgQAkBQjAICkGAEAJMUIACApRgAASTECAEiKEQBAUowAAJJiBACQFCMAgKQYAQAkxQgAIClGAABJMRqIzZs3l0cffbRcd9115bzzzitTU1Nl6dKl5bHHHitbtmzJraAf5Jkhked+UYx6bs+ePeWyyy4r999/f1m7dm3ZtWtXOXDgQNm/f3/ZuXNneeaZZ0brLr/88rJ3797cCxYneWZI5LmfFKMei0F16aWXlq+//jqXTPbVV1+VSy65pKxbty6XwOIizwyJPPeXYtRTmzZtKnfddVfOzdwdd9xRXnvttZyDxUGeGRJ57jfFqIe2bdtWbrzxxpw7fsuXLy8ffvhhzsHCkmeGRJ77TzHqoTPOOKMcPXo0547fkSNHyllnnZVzsLDkmSGR5/5TjHomflEvfoHvZG3fvr08+OCDOQcLQ54ZEnkeBsWoZ84555zy22+/5dyJO3jwYDn33HNzDhaGPDMk8jwMilGP/PDDD+Wiiy7KuZN34YUXln379uUczC95ZkjkeTgUox7ZunVrWbFiRc6dvJtuumn0i4KwEOSZIZHn4VCMemTjxo3l+eefz7mTt2HDhnLzzTeXF154wWSa92nlypXybBrMNBd5ns3jMXOKUY/Eq4d4FTFb4k9KP/jgg5yD+SXPDIk8D4di1CPxefNsfoZ9wQUXlB9//DHnYH7JM0Miz8OhGPVM/NXDoUOHcu7E/fLLL/7qgQUnzwyJPA+DYtQz8e9k7NixI+dOXPw7GQ899FDOwcKQZ4ZEnodBMeqhM888c/Svo56ow4cPl7PPPjvnYGHJM0Miz/2nGPVQvJq44YYbcu74LV26tOzcuTPnYGHJM0Miz/2nGPVU/B+YV61alXMzd/vtt5c333wz52BxkGeGRJ77TTHqsWeffbZcfPHFZffu3blksi+++GL0FxPr16/PJbC4yDNDIs/9pRj13N69e8uSJUvKfffdV55++unRL/7t379/NMXXsWz16tXlyiuvHP2T9bCYyTNDIs/9pBgNxLvvvlvWrFlTli1bVqampkZTfB3LYh30iTwzJPLcL4oRAEBSjAAAkmIEAJAUIwCApBgBACTFCAAgKUYAAEkxAgBIihEAQFKMAACSYgQAkBQjAICkGAEAJMUIACApRgAASTECAEiKEQBAUowAAJJiBACQFCMAgKQYAQAkxQgAIClGAABJMWLRuOeee8r69etzDk4NX375ZVmyZEk5cOBALoH+GsJ9XDFiUYiB9Oqrr5YtW7bkEhi+KEPxRBLZV4zou6HcxxWjWRQ3tieffDLn5k+csw831T179pStW7cem7799tty9OjRXPv3z08xWjwm5Tkeo1gX73Q8/PDDufTfduzYMZrGiRtn7Huy4hhxrMXu8OHD5aOPPjqW+48//rj88ccfufZv9WfK3ItcnnbaacemKKYLbdJYW4xOhfu4YjSL5jLc0z2Z9KUYbd68uTz33HPl7bffHj05rFy5sqxbt+7YoBrCgBqSSXmeyZO4YvSP+Flt2LChPPLII+W7774rDzzwQLnzzjtHTzCVYjQ/4h2NbhEat2yc6TI9U5OyP2msLUanwn1cMZpFcxnuxVyMDh48OBog9RXE559/nmv+LQZLfB9xc4nvZdOmTaNr//nnn0frhzCghmRSnhWjv/3111/l+++/L9u3bz+W/f379+faf8TPKq7ziSeeGL1TFD/T2Of111/PLRSj+TApz2EmmRx6MXIf/4diNIvacEdgIhznn3/+6O3aa6655tjb5/Um2X0rt3ujrwGLVzTTve0b54xtW3GuOGfdr4b4k08+yS3+3qaeL9Z3zxHHjPVxnPb6q3iF8OKLL5bly5ePrvN4B9TLL79c1q5dWw4dOpRbsJi0eW7F4xjraj5DN2+RofokEtvVcXD66aeP1sXjH/5f7mJ57BvLurrjpWqPWXMbua/nDO21t+Mrvg6xbayPc48bc7///nu59dZbRx8l1tzHNNNi9N5775Vt27blFsyHyEXNZFfNUjw+9XGqYvn777//r0zFO39tRmOqx+6eJ44d+ZvuPh4ZGTfWYt8YM7FPHQf1eFVcax0Hcd7uOWr+4lrreGi5j/+XYjSL2nBHYNq/NIlBUQdLBLbegENdF/vUgIfYt968Y3msH2dcMeqK/aPRt4M+zlnP2/6uSF0ex4zvYdJ5Y3l8L0eOHMkl06sDqr4Fe9VVV5Vdu3blWhabePxroWmneoNu8xk5iMxUdT6ydssttxzLUDs/09zFsna8VLG+HS/jxDbxJBLHbJ946njqHrsuj2m6vxSLfd56662cm14cI467Zs2a0RNq/GxWr149KlfMn245btUsRz7HFaOYr/kM3YzG+nhcY3m7XYhtapGp+eqK/cYVo9aka6zni+O6j88OxWgWteGOsEV4qna+W2Rq4Lv71OVh0oAKk4pR3Ly7T2btceLrOuDbJ76YYt/YfrrBGvtNTU0deyVSp/i8eZw6oOoTw1NPPeXJYRGb9PjH4xjrYoqvI0NtbkN7U64ZrmoGZ5q7SdfRHS9VbN8Wulp86jhpr7cdI3WK65p07Oree+8tV1xxxX+y/9lnn+UW/4hzxrHid4tWrVo1enUdH6Uxv2omx4nHO8pLt3SEeOzqfbLuPy6TtXh1z1OPHeJYMd81KeNx3shVzWbcO0N7nPb62hzH5D5+YhSjWdQGsB0MIeYjTKEGuarr2m1Cd5+YH6fe8FsR9vqEEOo2MdUntvqEVQd0V7vNOLHPibzSiGt76aWXyjvvvFPeeOONXMtiE4//uBtqzU+bpe528RjH1M10qFmeae4mXce4Y8e4ikzGPqHdpj1nHZv1+rvabcaJsXW87xjFE+7u3btHxWjjxo3lp59+yi2YD5NyFGomIz9tMWrna6ZDHKvNaKjHaLcL7XzdpmvStcU7QHX7dpt6/vY63Mdnj2I0i9rgRtjaG2vMR5hChLAtLTX8sX+sq4Oy3W7SgApxzti31W4f/20/Foh1MdX5WB8fb9TzVrF+ugH1559/lscff7xcf/31o+2O57PpeFKIv2yIX9z75ptvcgsWk3j8x92s43GMdW0+2ht45CheccbjHF+3RSX+W9/Wn2nuYn7cdcT+kadWd9t2DMV5avbrOeMaY5uuOPZ0xSjOc+21146+75r7mGbyO0bxS9exbTyptH/mzNyLLHQf7+6yNsuRj8hyPG7xdUwhHtP246mYj2PEdrEsjhFivo6FEDmo+7S6uQ2xb1vSutdZsxz7hjiu+/jsUIxmURvuCOmkYhT/jcDXtzvroAkR/ro8tquBjv3rL652xbK6Tz1eXEv9OCEGZgzUOoBiffc4sWzcMaYbUCFu7Pv27fvXX+ZMGlCvvPLK6PPo+v3GL6AuW7ZsdC0zfbXC/Gnz3IpMxLo2H/F1zVvkNPJdH+ea3VgX28S6WBZmkrtJ19EeN6b6BNaOoThXHXch1sXUareP48VxY5quGIXIbGxXcx/TpGK0YsWKctttt42u79dffy133313ufrqq8unn36aWzFf2sc7pm4e2lzFvSnyE49bzXjkrPvL17E81lf1nhzHie1qluqxu/ffeux6TfV47fiIe3g7DmKd+/jcUIwWQB1oCyUCXUMNp5IYe/HkBCdjJmVjrrmPzx3FaAEsZDGK87Zvz8KpIp7Mxr3zBMdroYuR+/jcUowWwEIVo3h1Ud9ehVNJfFxSPyaDk7WQxch9fO4pRgAASTECAEiKEQBAUowAAJJiBACQFCMAgKQYAQAkxQgAIClGAABJMQIASIoRAEBSjAAAkmIEAJAUIwCApBgBACTFCAAgKUYAAEkxAgBIihEAQFKMAACSYgQAkBQjAICkGAEAJMUIACApRgAASTECAEiKEQBAUowAAJJiBACQFCMAgKQYAQAkxQgAIClGAABJMQIASIoRAEBSjAAAkmIEAJAUIwCApBgBACTFCAAgKUYAAEkxAgBIihEAQFKMAACSYgQAkBQjAICkGAEAJMUIACApRgAASTECABgp5X84ANzlOxxiiQAAAABJRU5ErkJggg==">
</div>

Since x is a scalar value, there should be only 1 node in input layer. If it was high dimensional matrix, with m rows and n columns. The input layer would have mxn number of nodes, 1 node for each data matrix cell value.

Sequential method takes array of layers in parameter, to create larger neural network as well. Can you guess what will be the architecture of this neural network?
```
 model = keras.Sequential([
    layers.Dense(7, activation='relu', input_shape=[10]),
    layers.Dense(5, activation='relu'),
    layers.Dense(1)
  ])
```
Here is the answer, I hope you got it right.
<div align="center">
  <img src="data:image/png;base64,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">
</div>

Each “connection” between a single node in one layer to each of the nodes in the next layer has a variable “weight” (positive or negative), which represents how strongly that source node influences target node. When the network is initialized, all these weights are chosen randomly.

Each neuron contains a variable 'bias', which is chosen randomly at start.

These variables are the defining part of a neuron in neural network. Which we will use to apply [most basic linear fuction (line equation)](https://en.wikipedia.org/wiki/Linear_equation) i.e 

```
y = mx+b
```
In our context, this [can be re-written](https://becominghuman.ai/understanding-neural-networks-better-19e175ed30b9) as follows.

```
y = (w1)(x1) + (w2)(x2) + (w3)(x3) + ... + (wn)(xn) + b
```
where y is sum of bias and sum of all the products of weight and it's source values for each input connection of neuron.
<div align="center">
  <img src="https://miro.medium.com/max/1400/1*zd2AuD53JxsJCBeyelb-2A.png">
</div>
Once y is determined, it is passed through an activation function which either returns 0 or returns y, It determines if this neuron is activated.  

These simple transformation over multiple layers can [form very complicated and powerful functions](https://en.wikipedia.org/wiki/Universal_approximation_theorem), adding intellegence to our model. 

Every model requires [2 functions](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c),
1. Loss function which calculates degree of error in mapping input to output.
2. Optimize function, which attempts to reduce degree of error calculate by Loss function, by changing weights and biases.

When data is fed through the model to train, its weights and biases gets [updated/optimized](https://en.wikipedia.org/wiki/Mathematical_optimization) through an algorithm called **optimized function**, to [find the values of the variables](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c) which creates the desired function. 

```
model.compile(optimizer='sdg',loss='mean_squared_error')
```
In above line we have defined loss function to use 'mean squared error' algorithm to define how accurate the model/function is.

<div align="center">
  <img src="https://cdn-media-1.freecodecamp.org/images/MNskFmGPKuQfMLdmpkT-X7-8w2cJXulP3683">
</div>

While optimizer is being used is Stochastic Gradient Descent (SGD), which find derivation of loss function to get closer to wieghts. see section 2.4 of book Deep Learning with Python - Manning for details of how it actually works.

<div align="center">
  <img src="https://miro.medium.com/max/1400/1*pCmHfUWN7vkyZRSGnhipzg.png">
</div>

In below code, model is learning. epochs is the number of times xs and ys will pass through data to learn the wieghts and biases.
```
model.fit(xs, ys, epochs=500)
```

## Saving and loading trained model <a name="tf1sltm"></a>

In following line, neural network created above will be saved and loaded again from folder 'Model'
```
keras_model_path = "Model/temp_model_save"
model.save(keras_model_path)

restored_keras_model = tf.keras.models.load_model(keras_model_path)
restored_keras_model.predict([10.0])
```
For cloning, following code can be used,
```
model_copy= keras.models.clone_model(model1)
model_copy.build((None, 10))
model_copy.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model_copy.set_weights(model.get_weights())
```
replace 10 with number of variables in input layer
# Loss & Final Activation Function <a name="la"></a>
Although [Activation functions](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) exist on [each layer](https://www.youtube.com/watch?v=m0pIlLfpXWE), it is on last layer we must chose whose output should is of same category that we are attempting to solve, it is critical in producing desired output.

**Example,**
1. [Linear function](https://en.wikipedia.org/wiki/Linear_function_(calculus)) can have any output between -infinity to +infinity. It is approprite if you are predicting a numerical value. 
2. ReLU function can have any output between 0 to +infinity. It is appropriate if you are predicting a numerical value that must be only positive. Relu is just a [rectifier]((https://en.wikipedia.org/wiki/Rectifier_(neural_networks))) of linear function and there exists many rectifier functions.
3. [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) can have any output between 0 and 1, It is appropriate to find out [probability](https://en.wikipedia.org/wiki/Probability) of something.
4. [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) takes a matrix as input and outputs matrix of same dimensions such that all numbers in output matrix are between 0 and 1 and sum of all numbers in output matrix is equals to 1. Or simply put it takes matrix as input and output its [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution). 


Which Loss and Activation Functions [should I use?](https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8)
<div align="center">
  <img src="https://miro.medium.com/max/1400/1*IhP8BdoUpAbssltq0VBafg.png">
</div>

As Loss Function is used to determine how much error there is we can use following functions.

1. Entropy is uncertainty is in a given probability graph.[Cross-entropy Function](https://machinelearningmastery.com/cross-entropy-for-machine-learning/), Which gives a measure of the difference between two probability distributions for a given variable or set of events. Therefore, if we are predicting probability of [something](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) Entropy is a good function for [loss detection](https://en.wikipedia.org/wiki/Loss_function). 

2. [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error), gives the average squared difference between the estimated values and the actual value. Good for [regression loss calculation](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0).

# Optimize Function <a name="of"></a>
There are various optimization functions that can be used to find minima of the loss function (state with least error), these optimizer functions have their advantages and pitfalls. Adam, SGD, RMSProp, AdaGrad are just few of the optimizer functions. Generally speaking Adam is the fastest optimizer but sometimes it is hard to converge particularly in later stages of training, in such cases SDG is better as it is slower optimizer but generally gives better accuracy and do not get stuck while converging.
It is important to understand in high-dimensional data, even local minima can give us excellent accuracy, another reason to avoid finding global maxima is that it might indicate that model is now overfitted
<div align="center">
  <img src="https://miro.medium.com/max/700/1*JZbxrdzabrT33Yl-LrmShw.png">
</div>

In such case, the accuracy on training dataset may reach near 100% but when deployed in production the model accuracy is not as good. Therefore better approach is to create model that can generalize well on production data than to have high-accuracy model on test dataset.

# Avoiding Underfitting and Overfitting <a name="auo"></a>
General techniques to avoid underfitting,
* Increase model complexity i.e increase number of neurons per layer and number of layers in the model.
* Increase number of features, performing feature engineering. e.g in dataset many columns with a column name a new column 'marital status' can be introduced based on ***title*** present in name i.e Miss, Mrs., Mr. etc
* Remove noise from the data.
* Increase the number of epochs or increase the duration of training to get better results.
* Synthetic data generation.
* converting fix ranged data to range of 0 to 1
* Ensembling

In case of overfitting, general techniques are.
* Adding regularization
* K-Fold Cross Validation
* Ensembling
* Early stopping
* Adding Dropout function/layer
* Get a larger dataset

## Ensemble Models
One way to have better performing model is to have multiple models trained with various configurations and the merge their prediction via averaging or voting depending on type of expected output. For example following class can be used to handle multple models and applying ensembling on them
```
class Models():
    models=[]
        
    def add(self,model):
        self.models.append(model)
    
    def predict(self,modelInput):
        results=[]
        for model in self.models:
            results.append(model.predict(modelInput))
        return results
    
    def predictEnsemble(self,modelInput,ensembleType):
        results = self.predict(modelInput)
        if ensembleType=='average':
            return (sum(results)/len(results))
        elif ensembleType=='averageRound':
            return round(sum(results)/len(results), 2)
        elif ensembleType=='voting':
            return max(results,key=results.count)
        else:
            return max(results)
  ```
Example in following case, prediction by majority will be voted as final prediction.
```
models = Models()
models.add(model1)
models.add(model2)
models.add(model3)
models.predictEnsemble([13],'voting')
```
Although averaging and voting are [most well known](https://cs231n.github.io/neural-networks-3/#ensemble) deep learning ensembling techniques, these can also be used on non-deep learning, machine learning implementations like [BaggingClassifier, RandomForestClassifier, AdaBoostClassifier](https://medium.com/@rahulkaliyath/day-5-ensemble-methods-e12366fafdb).

# Neural Network Architecture Types <a name="nnat"></a>
Artificial Neural Network (ANN) is capable of learning any nonlinear function. Hence, these networks are popularly known as Universal Function Approximators.  It has many [types](https://www.asimovinstitute.org/neural-network-zoo/)

### Multilayer Perceptron
The simplest kind of neural network, It is Feed-Forward i.e a neural network that have linear stack of layers and processed data only moves forward to next layer. It is good at solving a large number of problems particularly on independent structured data.

### Recurrent Neural Network
 Recurrent Neural Network has a recurrent connection on the hidden state i.e output of a layer is connected back to it's own input.
<div align="center">
  <img src=" https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/assets_-LvBP1svpACTB1R1x_U4_-LwEQnQw8wHRB6_2zYtG_-LwEZT8zd07mLDuaQZwy_image-1.png">
</div>
This type of neural network architechture works best when input is in sequence of data unit, where sequence of data unit is important e.g google search predicting next words in a search sentence. RNN captures the sequential information present in the input data i.e. dependency between the words in the text while making predictions:
<div align="center">
  <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/1d_POV7c8fzHbKuTgJzCxtA.gif">
</div>
Some RNN also contain memmory storage of last state on neuron called Long short-term memory (LSTM) Neural Netwrok. 

### Convolution Neural Network
Convolution Neural Network have filters/kernels (mathematical operations) applied on adjecent group of data units, these models are particularly good when adjecency of a data unit matters e.g in an image finding cat features by applying filters on adjecent pixels. This is also good network for sequences of data for same reason.
<div align="center">
  <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/1BMngs93_rm2_BpJFH2mS0Q.gif">
</div>
<div align="center">
  <img src="https://www.swissquantumhub.com/wp-content/uploads/2019/11/Capture-d%E2%80%99%C3%A9cran-2019-11-06-%C3%A0-09.12.29.png">
</div>




<div align="center">
  <img src="https://miro.medium.com/max/4000/1*cuTSPlTq0a_327iTPJyD-Q.png">
</div>

# Converting Real World Problem To Mathematical Problem <a name="mp"></a>
When you have a real world problem to solve note what data type do you have as input data, and in what is the [category of your problem](#la) to find out what output should be desired. If input data is not numeric, our first task is to convert it into numeric data, without losing [correlation](https://en.wikipedia.org/wiki/Correlation_and_dependence) between data.

## Textual Data
There are many ways to convert [Textual data](https://freecontent.manning.com/deep-learning-for-text/) into numerical representation. I have discussed them under:

### High-Frequency Words Matrix
If the dataset contains collection of textual data e.g sentences,reviews,news articles etc, this technique can be applied. First create a [tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer#texts_to_matrix), with number of top high frequency words. Lets assume you put in num_words=1000 and **train_text** has length 5, it will create a tokenizer that will take top 1000 most occurring words. Then on next line pass entire collection of text that will break sentences into tokens/words and tokenizer will store the top 1000 high-frequency words. Then on last line transform collection of text into 2-d matrix, of 5x1000 length with each cell representing one of the top 1000 most occurring words in each text. Each cell will have numeric data of if that word has appear in that sentence i.e either 1 or 0.

```
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_text)
x_train = tokenize.texts_to_matrix(train_text)
```

### Hot Encoding
Sometimes you will have textual or numeric data that are more like labels than sentences or numbers. E.g In a dataset of images of fruits apple,mango and banana are the labels of the images. It is not proper to define these labels as numeric values as 1,2 and 3 respectively, because numeric value associates some wieghts to labels. It is better to associate vectors [0 0 1],[0 1 0] and [1 0 0] with the above labels respectively, it will not only transform textual data into number but also if matrix value is summed they hold same weights. This is called Hot Encoding. In this example **cat_collection** is array of labels, while **num_classes** is the number of unique labels in the array.

```
hotEncodedCategories = keras.utils.to_categorical(cat_collection, num_classes)
```

## Images
Images can be loaded as an multi-dimensional array with each pixel representing 3d vector representing rbg values, with width and height as 4th and 5th dimension and number of samples as 6th dimension. In case of Image dataset is greyscaled, the dataset is not 6d but 4d i.e rgb color channels removed and grayscale added.

These images can be passed through multiple layers of Convolution2D to create patterns and detect features in image.
<div align="center">
  <img src="https://miro.medium.com/max/640/1*h01T_cugn22R2zbKw5a8hA.gif">
  <img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-10-at-12-58-30-pm.png?w=242&h=256">
</div>
<div style='position:relative; padding-bottom:calc(56.25% + 44px)'>
<iframe src='https://gfycat.com/ifr/HandsomeMerryHyena' frameborder='0' scrolling='no' width='100%' height='100%' style='position:absolute;top:0;left:0;' allowfullscreen></iframe></div>


 Additionally to reduce computation cost after selecting important features, we can down-sample input by using MaxPooling2D class.
<div align="center">
  <img src="https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png">
  <img src="https://computersciencewiki.org/images/9/9e/MaxpoolSample.png">
</div>



[Example](https://freecontent.manning.com/deep-learning-for-image-like-data/) of MNIST dataset (handwritten digit) [recognition](https://github.com/tensorchiefs/dl_book/blob/master/chapter_02/nb_ch02_02.ipynb), (see ReferenceCode\notebooks\CNN_Example)
```


import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
layers = keras.layers
models = keras.models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten , Activation
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers

from tensorflow.keras.datasets import mnist


#saved in location
# %USERPROFILE%\.keras\datasets
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

# separate x_train in X_train and X_val, same for y_train
X_train=x_train[0:50000] / 255 #divide by 255 so that they are in range 0 to 1
Y_train=to_categorical(y_train[0:50000],10) # one-hot encoding

X_val=x_train[50000:60000] / 255
Y_val=to_categorical(y_train[50000:60000],10)

X_test=x_test / 255
Y_test=to_categorical(y_test,10)

del x_train, y_train, x_test, y_test

X_train=np.reshape(X_train, (X_train.shape[0],28,28,1))
X_val=np.reshape(X_val, (X_val.shape[0],28,28,1))
X_test=np.reshape(X_test, (X_test.shape[0],28,28,1))

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)



batch_size = 128
nb_classes = 10
img_rows, img_cols = 28, 28
kernel_size = (3, 3)
input_shape = (img_rows, img_cols, 1)
pool_size = (2, 2)
epochCount = 10

model = Sequential()
  
model.add(Convolution2D(8,kernel_size,padding='same',input_shape=input_shape)) #A
model.add(Activation('relu')) #B
model.add(Convolution2D(8, kernel_size,padding='same')) #A
model.add(Activation('relu')) #B
model.add(MaxPooling2D(pool_size=pool_size)) #C
  
model.add(Convolution2D(16, kernel_size,padding='same')) #D
model.add(Activation('relu')) #B
model.add(Convolution2D(16,kernel_size,padding='same')) #D
model.add(Activation('relu')) #B
model.add(MaxPooling2D(pool_size=pool_size)) #E
  
model.add(Flatten())#F
model.add(Dense(40))#G
model.add(Activation('relu')) #D
model.add(Dense(nb_classes))#G
model.add(Activation('softmax'))#H
  
 # compile model and initialize weights
model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
  
 # train the model
history=model.fit(X_train, Y_train,
                   batch_size=128,
                   epochs=epochCount,
                   verbose=2,
                   validation_data=(X_val, Y_val)
                  )



pred=model.predict(X_test)
print(confusion_matrix(np.argmax(Y_test,axis=1),np.argmax(pred,axis=1)))
acc_fc = np.sum(np.argmax(Y_test,axis=1)==np.argmax(pred,axis=1))/len(pred)
print("Acc_fc = " , acc_fc)

```

* #A Uses a convolutional layer with eight kernels of the size 3 x 3

* #B Applies the relu activation function to the feature maps

* #C This maxpooling layer has a pooling size of 2 x 2 and a stride of two

* #D Uses a convolutional layer with sixteen kernels of the size 3 x 3

* #E This maxpooling layer transforms the 14 x 14 x 16 input tensor to a 7 x 7 x 16 output tensor

* #F Flattens the output of the previous layer resulting in a vector of length 784 (7*7*16)

* #G Results into nb_classes (here ten) outputs

* #H Uses the softmax to transform the ten outputs to ten prediction probabilitie




## Videos
TBD
## Audio
TBD

## Sequences, Time Series
TBD

# Scenarios <a name="scenarios"></a>
TBD

BBC news article category prediction: 

xpai\Learning\ReferenceCode\bbc_news\runbbc.bat

# Deploying Model <a name="dm"></a>

## Integrating via ML .NET <a name="dmmlnet"></a>

**Note:** So far have spent 60+ hours on directly integrating exported tensorflow model into C# console application (both on .net framework and .net core) using ML .NET library and other libraries in eco system. The code is extremely consfusing and libraries dont seems to work well espeacially their pre-requisite ecosystem is very ambiguous. Have reached the conclusion to skip integrating in .net directly and use another path way instead.

TBD ... may be!

## Exposing Model via Python REST API <a name="dmemrest"></a>
Import python package flask, copy tensorflow saved model from top example in folder as filename 'model'. Now from same folder run following code (see folder \ReferenceCode\notebooks\Exposing_Model_Python_REST_API) as python script

```
import tensorflow as tf
import flask
from flask import request,jsonify
import json
import numpy as np


app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/predict', methods=['GET'])
def testAPI2():
    if 'input' in request.args:
        input = float(request.args['input'])
    else:
        return "Error: No input field provided. Please specify an input."
    inputArray = [input]
    restored_keras_model = tf.keras.models.load_model('model')
    return {
        "prediction": str(restored_keras_model.predict(inputArray).flat[0]),
        "Input":input
    }

    
app.run()
```
Once server starts, go to following url:
http://127.0.0.1:5000/predict?input=5.3

REST API will take 5.3 as input and output prediction as json.

## Google Cloud Compute: Serverless function <a name="dmgccsf"></a>
TBD

https://towardsdatascience.com/models-as-serverless-functions-7930a70193d4


# Cheat Sheets <a name="cheatsheets"></a>