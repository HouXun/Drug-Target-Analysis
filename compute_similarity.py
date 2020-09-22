# -*- coding: utf-8 -*-
"""
convert data into similarity between drug or protein network
using jaccard distance 

"""
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import utils

'''
#no side information
dir='./data'
Nets = []
subsets=['Enzyme','Ion_channel','GPCR','Nuclear_receptor']
for subnet in subsets:
    for net in Nets:
        inputID=os.path.join(dir,subset,net)+'.txt'
        M=pd.read_table(inputID, sep='	', header=None).values[1:, 1:]
        # jaccard distance
        Sim=1-pdist(M,'jaccard')
        Sim=squareform(Sim)
        Sim=Sim+np.eye(len(Sim))
        Sim=np.nan_to_num(Sim)
    
        #output csv file
        outputID=os.path.join(dir,subset,net)+'.csv'
        utils.outputCSV(outputID,Sim)
'''

subsets=['Enzyme','Ion_channel','GPCR','Nuclear_receptor']
Nets = ['simmat_dc', 'simmat_dg','admat_dgc']
dir='./data'
for subset in subsets:
    for net in Nets:
        inputID=os.path.join(dir,subset,net)
        M = pd.read_table(inputID+'.txt', sep='	', header=None).values[1:, 1:]
        utils.outputCSV(inputID+'.csv', M)
