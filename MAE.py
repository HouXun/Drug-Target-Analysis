# -*- coding: utf-8 -*-
"""
score
"""

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras import layers
from keras import Input
from keras import regularizers


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

    mae_list=[]
    kf = KFold(n_splits=3)
    epoch=20

    count=0
    for train_index, val_index in kf.split(labels):
        count+=1
        print('count: ',count)
        drug_train, drug_val = drug_data[train_index], drug_data[val_index]
        protein_train, protein_val = protein_data[train_index], protein_data[val_index]
        labels_train, labels_val = labels[train_index], labels[val_index]

        class_weight = compute_class_weight('balanced', np.array([0,1]), labels_train)

        drug_input = Input(shape=(drug_dim,), dtype='float32', name='drug')
        drug_feature = layers.LeakyReLU()(drug_input)
        protein_input = Input(shape=(protein_dim,), dtype='float32', name='protein')
        protein_feature = layers.LeakyReLU()(protein_input)
        concatenated = layers.concatenate([drug_feature, protein_feature], axis=-1)
        """ With regularization parameter 
        score = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(concatenated)
        """
        score = layers.Dense(1, activation='sigmoid')(concatenated)
        model = Model([drug_input, protein_input], score)
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

        """
        history=model.fit([drug_train,protein_train],labels_train,validation_data=([drug_val,protein_val],labels_val),epochs=epoch,batch_size=128,verbose=0)
        
        """
        history = model.fit([drug_train, protein_train], labels_train,
                            validation_data=([drug_val, protein_val], labels_val), epochs=epoch, batch_size=128,class_weight=class_weight,
                            verbose=0)
        val_mae=history.history['val_mean_absolute_error']
        mae_list.append(val_mae)

    mean_mae=np.mean(mae_list,axis=0)
    """
    np.savetxt('Results/mae_2.txt',mean_mae)
    np.savetxt('Results/mae_1.txt',mean_mae)
    """
    np.savetxt('Results/mae_a.txt', mean_mae)

    plt.plot(range(1,epoch+1),mean_mae,'b',label='Validation mae per epoch')
    plt.title('Validation mae per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Mae')
    plt.legend()
    plt.show()