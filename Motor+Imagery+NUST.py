
# coding: utf-8

# In[ ]:




# In[3]:

import matplotlib
import pandas as pd
import numpy as np
import math
import re
import scipy
#import pymysql as db
import os
import sklearn
import datetime
import scipy.io as sio
from matplotlib import pyplot as plt
#from sshtunnel import SSHTunnelForwarder
from sklearn import model_selection, svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, average_precision_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline

#BCI MNE Imports
import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

#Data Ingestion
raw=sio.loadmat('C:\\648 wala\\VIVEK SHARMA\\EDUCATIONAL\\Brain Computer Interface\\DataSets\\NUST Pakistan\\Subject1_1D.mat')

#Determining the Left and Right Motor Stimulus From Appropriate Channels
left=raw['left']
right=raw['right']

time=[i for i in range(64300)] #Time For the Complete Task

lc1=left[:][0] #Taking Values Only From the C1 Channels For Left Hand Movement
print(lc1.shape)
rc1=right[:][0] #Taking Values Only From the C1  Channels For the Right Hand Movement
print(rc1.shape)

lc3c4=left[:][4:6] #Only For Channel From C3 and C4 from left Hand
print(lc3c4.shape)
rc3c4=right[:][4:6] #Only For Channel From C3 and C4 from right Hand
print(rc3c4.shape)

#Design Butterworth Filter
import scipy.signal as signal
N=2 #Filter Order
Wn = 64/500 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')

#Applying the Filter to the 
leftf = signal.filtfilt(B,A, lc3c4)
rightf=signal.filtfilt(B,A, rc3c4)

#Converting Both the Numpy Array to DataFrame
R = {'C3': rightf[0], 'C4': rightf[1]}
L = {'C3': leftf[0], 'C4': leftf[1]}

l_data = pd.DataFrame(data=L)
r_data = pd.DataFrame(data=R)

l_data['label']=1 #IF Data is From LEft HAnd it is 1
r_data['label']=0 #IF Data is from Right Hand it is 0

main_data=pd.concat([l_data, r_data])

feat=main_data[['C3','C4']]
labe=main_data[['label']]

#Train and Test Split
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(feat, labe, test_size = 0.3, random_state=42, stratify=labe)

#Running The Model
model=RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(train_features,train_labels)
prediction=model.predict(test_features)
print(classification_report(test_labels, prediction))


# In[4]:

plt.magnitude_spectrum(rc3c4[:][0])
plt.magnitude_spectrum(rightf[:][0])
plt.show()


# In[5]:

plt.magnitude_spectrum(rc3c4[:][1])
plt.magnitude_spectrum(rightf[:][1])
plt.show()


# In[6]:

plt.magnitude_spectrum(lc3c4[:][1])
plt.magnitude_spectrum(leftf[:][1])
plt.show()


# In[7]:

plt.figure(figsize=(15,3))
plt.plot(time,lc3c4[:][0],'b')
#plt.plot(time,leftf[:][0],'y')
plt.show()


# In[8]:

plt.figure(figsize=(20,3))
plt.plot(time,leftf[:][0],'y')
plt.show()


# In[9]:

plt.figure(figsize=(20,3))
plt.plot(time,leftf[:][0],'y')
plt.plot(time,rightf[:][0],'b')
plt.show()


# In[10]:

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True


    return array(maxtab), array(mintab)


# In[11]:

rc3=rightf[:][0]
rc4=rightf[:][1]
lc3=leftf[:][0]
lc4=leftf[:][1]

from matplotlib.pyplot import plot, scatter, show
series = rc3.tolist()
series2= rc4.tolist()
seriesl1=lc3.tolist()
seriesl2=lc4.tolist()
maxtabc3, mintabc3 = peakdet(series,0.5)
maxtabc4,mintabc4 = peakdet(series2,0.5)
maxtablc3,mintablc3=peakdet(seriesl1,0.5)
maxtablc4,mintablc4=peakdet(seriesl2,0.5)
plt.figure(figsize=(20,3))
plot(series,'y')
scatter(array(maxtabc3)[:,0], array(maxtabc3)[:,1], color='b')
#scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
show()

plt.figure(figsize=(20,3))
plot(series,'y')
scatter(array(maxtabc4)[:,0], array(maxtabc4)[:,1], color='b')
#scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
show()


# In[12]:

maxrc3=maxtabc3[:,1]
maxrc4=maxtabc4[:,1]


# In[13]:

maxlc3=maxtablc3[:,1]
maxlc4=maxtablc4[:,1]
print(';eft',maxlc3.shape)
print(';eft4',maxlc4.shape)
maxrc3.shape
print('max r4',maxrc4.shape)


# In[14]:

mlc3=maxlc3[:6301]
mlc4=maxlc4[:6301]

mrc3=maxrc3[:6301]
mrc4=maxrc4[:6301]


# In[15]:

mR = {'C3': mrc3, 'C4': mrc4}
mL = {'C3': mlc3, 'C4': mlc4}

ml_data = pd.DataFrame(data=mL)
mr_data = pd.DataFrame(data=mR)

ml_data['label']=1 #IF Data is From LEft HAnd it is 1
mr_data['label']=0 #IF Data is from Right Hand it is 0

main_dataM=pd.concat([ml_data, mr_data])

mfeat=main_data[['C3','C4']]
mlabe=main_data[['label']]


# In[16]:

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(mfeat, mlabe, test_size = 0.3, random_state=42, stratify=labe)

#Running The Model
model=RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(train_features,train_labels)
prediction=model.predict(test_features)
print(classification_report(test_labels, prediction))
print("RF Accuracy is :",accuracy_score(test_labels,prediction)*100)


# In[17]:

from sklearn.neighbors import KNeighborsClassifier

my_classifier=KNeighborsClassifier(n_neighbors=50)

my_classifier.fit(train_features,train_labels)
prediction=my_classifier.predict(test_features)
print(classification_report(test_labels, prediction))
print("K NN Accuracy is :",accuracy_score(test_labels,prediction)*100)


# In[19]:

from sklearn.ensemble import GradientBoostingClassifier
my_classifier = GradientBoostingClassifier(n_estimators=100)
my_classifier.fit(train_features,train_labels)
prediction=my_classifier.predict(test_features)
print(classification_report(test_labels, prediction))
print("GBC is :",accuracy_score(test_labels,prediction)*100)


# In[ ]:



