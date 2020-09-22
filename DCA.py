# -*- coding: utf-8 -*-
"""
DCA
"""
from __future__ import division
import os
import numpy as np
import pandas as pd
import math
import scipy.linalg as la
import utils
import diffusionRWR as RWR


def DCA(networks,dim,rsp,maxiter):
    i=1
    Q=np.array([])
    print(networks)
    for network in networks:
        fileID=network+'.csv'
        net=pd.read_csv(fileID,header=None).values
        #net=net.as_matrix(columns=None)
        tQ=RWR.diffusionRWR(net,maxiter,rsp)
        if i==1:
            Q=tQ
        else:
            #concatenate network
            Q=np.hstack((Q,tQ))
        i+=1
    print(Q.shape)
    nnode=len(Q)
    alpha=1/nnode
    Q=np.log(Q+alpha)-math.log(alpha)
    Q=np.dot(Q,Q.T)
    
    #use SVD to decompose matrix
    U,Sigma,VT=la.svd(Q,lapack_driver='gesvd',full_matrices=True)
    #Sigma=np.dot(np.eye(dim),np.diag(Sigma[:dim]))
    Sigma=np.dot(np.eye(Sigma.shape[0]),np.diag(Sigma))
    #U=U[:,:dim]b h
    #get context-feature matrix, since we use Q*PT to get square matrix, we need to sqrt twice
    X=np.dot(U,np.sqrt(np.sqrt(Sigma)))
    return X


maxiter = 20
restartProb = 0.50
dim_drug = 100
dim_prot = 400
dir='./data'

subsets=['Enzyme','Ion_channel','GPCR','Nuclear_receptor']
drugNets = ['simmat_dg']
proteinNets = ['simmat_dc']
for subset in subsets:
    networks=[os.path.join(dir,subset,drugNet) for drugNet in drugNets]
    X = DCA(networks, dim_drug, restartProb, maxiter)
    X = np.array(X)
    utils.outputCSV(os.path.join(dir,subset,'drug_vector_d.csv'),X)
for subset in subsets:
    networks=[os.path.join(dir,subset,proteinNet) for proteinNet in proteinNets]
    Y = DCA(networks, dim_drug, restartProb, maxiter)
    Y=np.array(Y)
    utils.outputCSV(os.path.join(dir, subset, 'protein_vector_d.csv'), Y)


