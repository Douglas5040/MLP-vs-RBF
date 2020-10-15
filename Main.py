'''
Created on Sep 30, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Exercise 3: MLP + RBF
@author: Damares Resende
'''

import numpy as np
from RNDataset import RNData
from RNModels import RBFNet, MLPNet
from matplotlib import pyplot as plt

def main():
    qtd_rodadas = 10
    log = {'RBF TN': np.zeros(qtd_rodadas), 'RBF TS': np.zeros(qtd_rodadas), \
           'MLP TN': np.zeros(qtd_rodadas), 'MLP TS': np.zeros(qtd_rodadas)}
    
    for i in range(qtd_rodadas):
        print('############## Executando rodada ' + str(i + 1) + ' ##############\n')
        
        # Carregando os dados da web (https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)
        X, y = RNData.getWinesData(normalize = True, binarize = True)
        X_train, X_test, y_train, y_test = RNData.divCamadas(X, y, 0.2)
        # print('X_train: ',X_train)
        # print('\nX_test: ',X_test)
        # print('\ny_train: ',y_train)
        # print('\ny_test: ',y_test)

        # Rede Neural RBF
        rbf = RBFNet(X_train.shape[1], 20, y_train.shape[1])
        rbf.train(X_train, y_train)
        
        log['RBF TN'][i] = rbf.evaluate(X_train, y_train)
        log['RBF TS'][i] = rbf.evaluate(X_test, y_test)
        
        print('>> RBF Acurácia - training: ' + str(round(log['RBF TN'][i],3)) + ' %')
        print('>> RBF Acurácia - testing: ' + str(round(log['RBF TS'][i],3)) + ' %\n')
        
        rbf = RBFNet(X_train.shape[1], 1, y_train.shape[1])
        rbf.train(X_train, y_train)
        
        # Rede Neural MLP 
        mlp = MLPNet(nn_inputs=X_train.shape[1], nn_hidden1=15, nn_hidden2=5, nn_targets=y_train.shape[1])
        mlp.train(X_train, y_train, n_epochs=20000, l_rate=0.3, momentum=0.75)
        
        log['MLP TN'][i] = mlp.evaluate(X_train, y_train)
        log['MLP TS'][i] = mlp.evaluate(X_test, y_test)
        
        print('>> MLP Acurácia - treinando: ' + str(round(log['MLP TN'][i],3)) + ' %')
        print('>> MLP Acurácia - testando: ' + str(round(log['MLP TS'][i],3)) + ' %\n')
        
        
    plt.plot(range(qtd_rodadas), log['RBF TN'], '-o', label='RBF Treinamento')
    plt.plot(range(qtd_rodadas), log['RBF TS'], '-o', label='RBF Teste')
    plt.plot(range(qtd_rodadas), log['MLP TN'], '-o', label='MLP Treinamento')
    plt.plot(range(qtd_rodadas), log['MLP TS'], '-o', label='MLP Teste')
    plt.legend()
    
    plt.xlabel('Rodadas')
    plt.ylabel('Acurácia (%)')
    plt.title("Performace da MLP vs RBF")
    plt.tight_layout()
    plt.show()
    
    print('############## Fim =) ##############')

if __name__ == '__main__':
    main()