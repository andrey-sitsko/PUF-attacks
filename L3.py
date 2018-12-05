import zipfile
import pandas
import numpy as np
import os

TRAINING_EDGE=0.7
NROWS=200000

def drawChart(X,Y,title):
  import seaborn as sns
  import matplotlib.pyplot as plt
  chart=sns.regplot(x=X, y=Y, fit_reg=False)
  chart.set_title(title)
  plt.show()

def getDFrame(fileName):
  name=os.path.splitext(fileName)[0]
  filepath=zipfile.ZipFile('./samples/Lab 3/{}'.format(fileName), 'r').extract('{}.txt'.format(name), 'Lab2_unzipped')
  dFrame=pandas.read_csv(filepath,sep=" ",header=None,dtype=str,nrows=NROWS)
  os.remove(filepath)
  return dFrame

