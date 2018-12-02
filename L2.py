from sklearn.linear_model import LogisticRegression
import zipfile
import pandas
import numpy as np
import os

TRAINING_EDGE=0.9
SAMPLES=['Base8.zip', 'Base16.zip', 'Base24.zip', 'Base32.zip', 'Base40.zip', 'Base96.zip']

#os.listdir('samples/Lab 2/') for all samples
for fileName in SAMPLES:
    name=os.path.splitext(fileName)[0]
    filepath=zipfile.ZipFile('./samples/Lab 2/{}'.format(fileName), 'r').extract('{}.txt'.format(name), 'Lab2_unzipped')

    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    dFrame=pandas.read_csv(filepath,sep=" ",header=None,dtype=str)

    X=[[int(c) for c in i] for i in dFrame.values[:,0]]
    Y=np.array(dFrame.values[:,1]).astype(int)

    trainingEdge=int(len(dFrame.values) * TRAINING_EDGE)

    logreg.fit(X[:trainingEdge], Y[:trainingEdge])
    print('Score for {} is {}'.format(name, logreg.score(X[trainingEdge:], Y[trainingEdge:])))








