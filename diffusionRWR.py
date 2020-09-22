# -*- coding: utf-8 -*-
"""
diffusionRWR

"""
import numpy as np

def diffusionRWR(A,maxiter,restartProb):
    #A: Jaccard distance matrix

    n=len(A)

    #Add self-edge to isolated nodes
    index=np.sum(A,axis=0)==0
    diag=np.diag(index)
    A[diag]=1

    #normalize the adjacency matrix
    P=A/A.sum(axis=0)
    
    #Personalized PageRank
    restart=np.eye(n)
    Q=np.eye(n)
    for i in range(1,maxiter):
        Q_new=(1-restartProb)*np.dot(P,Q)+restartProb*restart
        delta=np.linalg.norm((Q-Q_new))
        Q=Q_new
        if delta<1e-6:
            break
    return Q
        
