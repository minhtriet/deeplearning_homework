# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np 
 

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """
    X = X[200:205,:]
    Y = Y[400:420,:]
    return np.sqrt((np.square(X[:,np.newaxis] - Y).sum(axis=2)))
 

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    argmins = np.sqrt((np.square(dists[:,np.newaxis] - labels).sum(axis=2))).argmin(axis=1)
    return labels[argmins]
     