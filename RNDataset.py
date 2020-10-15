'''
Instituto de Ciencias Matematicas e de Computacao - USP São Carlos
SCC5809: Redes Neurais

Exercício 03: MLP + RBF
Equipe:
ID. Matricula (01) - 12116252 Dheniffer Caroline Araújo Pessoa

ID. Matricula (02) - 12114819 Douglas Queiroz Galucio Batista 

ID. Matricula (03) - 12116738 Laleska Mesquita
'''

from random import shuffle
import numpy as np
import urllib.request

class RNData():
    
    @staticmethod
    def getWinesData(normalize = False, binarize = True):

        #carregando e baixando a base de dados da web arquivo 'wine.data'
        target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        
        data = urllib.request.urlopen(target_url).read().split(b'\n')
        data_matrix = np.zeros((len(data)-1,len(data[0].split(b',')))) 
        #print('data_matrix: ', data_matrix)
        
        for idx, line in enumerate(data):
            if len(line) > 0:
                #print('----> ',[x for x in line.strip().split(b',') if x])
                #print('line ---> ', line.split(b','))
                data_matrix[idx] = [float(x) if len(x) else 0.0 for x in line.split(b',') if x]

        #print('\ndata_matrix: ', data_matrix)
        X = data_matrix[:,1:]
        y = data_matrix[:,0]

        #print('\n\n----:> X ',X)
        #print('----:> Y ',y)

        if normalize:
            X = (X - X.min())/(X.max() - X.min())
        if binarize:
            y = RNData.binarizeLabels(y, 3)
        
        #print('\n\n----:> X ',X)
        #print('----:> Y ',y)
    
            
        return X,  y
    
    @staticmethod
    def binarizeLabels(labels, n):
        targets = np.zeros((labels.shape[0], n))
        #print('targets: ',targets)
        
        for i in range(labels.shape[0]):
            targets[i][int(labels[i]) - 1] = 1
        return targets
    
    @staticmethod
    def divCamadas(X, y, test_split):
        data_map = {}
        
        # divisão do conjunto de dados de acordo com cada classe
        for idx, ex in enumerate(X):
            if str(y[idx]) not in data_map.keys():
                data_map[str(y[idx])] = [(ex, y[idx])]
            else:
                data_map[str(y[idx])].append((ex, y[idx]))
        
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        
        # "Embaralhando" as listas e obtendo uma parte de cada classe
        for set_ in data_map.values():
            shuffle(set_)
            limit = round(test_split * len(set_))
            
            X_test.extend([ex[0] for ex in set_[:limit]])
            y_test.extend([ex[1] for ex in set_[:limit]])
            X_train.extend([ex[0] for ex in set_[limit:]])
            y_train.extend([ex[1] for ex in set_[limit:]])
            
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
            