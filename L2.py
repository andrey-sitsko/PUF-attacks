from sklearn.linear_model import LogisticRegression
import zipfile
import pandas
import numpy as np
import os

TRAINING_EDGE=0.7
NROWS=200
SAMPLES=['Base8.zip', 'Base16.zip', 'Base24.zip']

chartDataX = []
chartDataY = []
trainingEdge=0

#for fileName in SAMPLES:
for fileName in os.listdir('samples/Lab 2/'):
    name=os.path.splitext(fileName)[0]
    filepath=zipfile.ZipFile('./samples/Lab 2/{}'.format(fileName), 'r').extract('{}.txt'.format(name), 'Lab2_unzipped')

    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    dFrame=pandas.read_csv(filepath,sep=" ",header=None,dtype=str,nrows=NROWS)

    X=[[int(c) for c in i] for i in dFrame.values[:,0]]
    Y=np.array(dFrame.values[:,1]).astype(int)

    trainingEdge=int(len(dFrame.values) * TRAINING_EDGE)

    logreg.fit(X[:trainingEdge], Y[:trainingEdge])

    score=logreg.score(X[trainingEdge:], Y[trainingEdge:])
    chartDataX.append(int(fileName.split("Base")[1].split(".zip")[0]))
    chartDataY.append(int(score * 100))

    print('Score for {} is {} training set length {}'.format(name, score, trainingEdge))

import seaborn as sns
import matplotlib.pyplot as plt
chart=sns.regplot(x=chartDataX, y=chartDataY, fit_reg=False)
chart.set_title("LogisticRegression, training set {}, test set {}".format(trainingEdge, NROWS - trainingEdge))
plt.show()
