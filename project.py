#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:42:57 2021

@author: Leonardo Savasta
"""

# Importing libraries
from matplotlib import pyplot as plt
import numpy as np
import os
import re

def trainLinearModel(X, Y):
    """
    
    Parameters
    ----------
    X : Numpy 2d array of floats
        Predictor data
    Y : Numpy array of floats
        Response values

    Returns
    -------
    weights : Numpy 2d array of floats
        Weights for linear regression.

    """
    
    A = np.linalg.pinv(np.dot(X.T, X))
    B = np.dot(X.T, Y)
    weights = np.dot(A,B)
    
    return weights


def calculateError(W, X, Y):
    """

    Parameters
    ----------
    W : Numpy 2d array of floats
        Weights for linear regression
    X : Numpy 2d array of floats
        Predictor data
    Y : Numpy array of floats
        Response values

    Returns
    -------
    error : Float
        Mean Squared Errors of model

    """
    
    A = np.dot(X, W) - Y
    error = (1/len(Y)) * np.dot(A.T, A)
    
    return error

# Obtain current directory
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

input_file = input("Enter the name of your data file: ")
training_file = open(os.path.join(__location__, input_file), 'r')

# Unpacking number of rows and number of features in textfile

rows, features = [int(x) for x in re.findall('\d+', training_file.readline())]
X = np.zeros([rows, features+1])
Y = []

# Unpacking data into array
for idx in range(rows):
    row = training_file.readline().split("\t")
    X[idx] = np.array([1] + row[:-1])
    Y.append(float(row[-1].strip()))

Y = np.array(Y)
W = trainLinearModel(X, Y)

print("\nWeights: ", W)
print("\n Training J value (Error): ", calculateError(W, X, Y))

input_file = input("Enter the name of your testing data file: ")
testing_file = open(os.path.join(__location__, input_file), 'r')

rows, features = [int(x) for x in re.findall('\d+', testing_file.readline())]
test_X = np.zeros([rows, features+1])
test_Y = []

# Unpacking data into array
for idx in range(rows):
    row = testing_file.readline().split("\t")
    test_X[idx] = np.array([1] + row[:-1])
    test_Y.append(float(row[-1].strip()))

test_Y = np.array(test_Y)

print("\n Testing J value (Error): ", calculateError(W, test_X, test_Y))