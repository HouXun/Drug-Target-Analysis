# -*- coding: utf-8 -*-
"""
score
"""

import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras import layers
from keras import Input

def preprocessing(A,B,admat):
    A_data=[]
    B_data=[]
    labels=[]
    for i,a in enumerate(A):
        for j,b in enumerate(B):
            A_data.append(a.tolist())
            B_data.append(b.tolist())
            labels.append(admat[i,j])
    A_data=np.array(A_data)
    B_data=np.array(B_data)
    labels=np.array(labels)
    return A_data,B_data, labels

subsets=['Enzyme','Ion_channel','GPCR','Nuclear_receptor']
dir='./data'
for subset in subsets[0:1]:
    drug=pd.read_csv(os.path.join(dir,subset,'drug_vector_d.csv'),header=None).values
    protein = pd.read_csv(os.path.join(dir, subset, 'protein_vector_d.csv'), header=None).values
    interaction=pd.read_csv(os.path.join(dir,subset,'admat_dgc.csv'),header=None).values
    print(subset,drug.shape,protein.shape,interaction.shape)

    drug_data, protein_data, labels=preprocessing(drug,protein,interaction)
    print(drug_data.shape,protein_data.shape,labels.shape)

    drug_dim = drug_data.shape[1]
    protein_dim = protein_data.shape[1]

    size=len(labels)
    index=range(size)
    train_index,test_index=train_test_split(index,test_size=0.33,random_state=0)

    drug_train, drug_test = drug_data[train_index], drug_data[test_index]
    protein_train, protein_test = protein_data[train_index], protein_data[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

    class_weight = compute_class_weight('balanced', np.array([0, 1]), labels_train)

    drug_input = Input(shape=(drug_dim,), dtype='float32', name='drug')
    drug_feature = layers.LeakyReLU()(drug_input)
    protein_input = Input(shape=(protein_dim,), dtype='float32', name='protein')
    protein_feature = layers.LeakyReLU()(protein_input)
    concatenated = layers.concatenate([drug_feature, protein_feature], axis=-1)
    score = layers.Dense(1, activation='sigmoid')(concatenated)
    model = Model([drug_input, protein_input], score)
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

    history=model.fit([drug_train,protein_train],labels_train,validation_split=0.33,shuffle=True,epochs=50,batch_size=128,verbose=0)

    y_true=labels_test
    y_score=model.predict([drug_test,protein_test])
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)

    np.savetxt('Results/fpr_2.txt',fpr)
    np.savetxt('Results/tpr_2.txt', tpr)

    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('False Positive Ratio')
    plt.ylabel('True Positive Ratio')
    plt.show()
    AUC = auc(fpr, tpr)
    print(AUC)
