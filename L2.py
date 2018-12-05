import zipfile
import pandas
import numpy as np
import os

TRAINING_EDGE=0.5
NROWS=20000

def drawChart(X,Y,title):
  import seaborn as sns
  import matplotlib.pyplot as plt
  chart=sns.regplot(x=X, y=Y, fit_reg=False)
  chart.set_title(title)
  plt.show()

def getDFrame(fileName):
  name=os.path.splitext(fileName)[0]
  filepath=zipfile.ZipFile('./samples/Lab 2/{}'.format(fileName), 'r').extract('{}.txt'.format(name), 'Lab2_unzipped')
  dFrame=pandas.read_csv(filepath,sep=" ",header=None,dtype=str,nrows=NROWS)
  os.remove(filepath)
  return dFrame

def logisticRegression():
  from sklearn.linear_model import LogisticRegression

  chartDataX=[]
  chartDataY=[]

  for fileName in os.listdir('samples/Lab 2/'):
    dFrame=getDFrame(fileName)
    X=[[int(c) for c in i] for i in dFrame.values[:,0]]
    Y=np.array(dFrame.values[:,1]).astype(int)
    trainingEdge=int(len(dFrame.values) * TRAINING_EDGE)

    logreg = LogisticRegression(solver="lbfgs")
    logreg.fit(X[:trainingEdge], Y[:trainingEdge])
    score=logreg.score(X[trainingEdge:], Y[trainingEdge:])

    chartDataX.append(int(fileName.split("Base")[1].split(".zip")[0]))
    chartDataY.append(int(score * 100))

  drawChart(chartDataX, chartDataY, "LogisticRegression, training set {}, test set {}".format(trainingEdge, NROWS - trainingEdge))

def supportVector():
  from sklearn import svm

  chartDataX=[]
  chartDataY=[]

  for fileName in os.listdir('samples/Lab 2/'):
    dFrame=getDFrame(fileName)
    X=[[int(c) for c in i] for i in dFrame.values[:,0]]
    Y=np.array(dFrame.values[:,1]).astype(int)
    trainingEdge=int(len(dFrame.values) * TRAINING_EDGE)

    clf=svm.SVC(gamma='auto')
    clf.fit(X[:trainingEdge], Y[:trainingEdge])
    score=clf.score(X[trainingEdge:], Y[trainingEdge:])

    chartDataX.append(int(fileName.split("Base")[1].split(".zip")[0]))
    chartDataY.append(int(score * 100))

  drawChart(chartDataX, chartDataY, "SupportVector, training set {}, test set {}".format(trainingEdge, NROWS - trainingEdge))

def gradientBoosting():
  from sklearn.ensemble import GradientBoostingRegressor

  chartDataX=[]
  chartDataY=[]

  for fileName in os.listdir('samples/Lab 2/'):
    dFrame=getDFrame(fileName)
    X=[[int(c) for c in i] for i in dFrame.values[:,0]]
    Y=np.array(dFrame.values[:,1]).astype(int)
    trainingEdge=int(len(dFrame.values) * TRAINING_EDGE)

    clf = GradientBoostingRegressor()
    clf.fit(X[:trainingEdge], Y[:trainingEdge])

    score=clf.score(X[trainingEdge:], Y[trainingEdge:])

    chartDataX.append(int(fileName.split("Base")[1].split(".zip")[0]))
    chartDataY.append(int(score * 100))

  drawChart(chartDataX, chartDataY, "GradientBoostingRegressor, training set {}, test set {}".format(trainingEdge, NROWS - trainingEdge))

gradientBoosting()