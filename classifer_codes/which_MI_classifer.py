# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 21:16:18 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:10:29 2022

@author: user
"""
from scipy.io import loadmat
import numpy as np
from scipy.signal import butter, lfilter
from mne.decoding import CSP
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn import svm
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
#import novel_cnn as novel
import itertools

def plot_sig(t,x):
    plt.plot(t, x)
    plt.ylabel('Amplitude')
    plt.xlabel('Time [sec]')
    plt.show() 
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None, normalize=None):
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #return plt
    plt.tight_layout()
    plt.show()
    #plt.savefig('images/cf.png',pad_inches=0.1)
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    #b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

filename = 'BCICIV_calib_ds1d.mat'
lowcut = 4
highcut =40
fs= 100
order = 8
n_of_comp =16
n_of_ft =14
X=np.zeros(0).reshape(0,59,400)                              # creating the dataset variable
Y=np.zeros(0) 
matlabfile = loadmat(filename,struct_as_record=True, squeeze_me=False)            # loading matfile
data = matlabfile['cnt']         
print(data.shape)
                                           #selecting the run
        
marks = matlabfile["mrk"]                    # reconfiguring trail values to python Compatable
position = marks["pos"][0][0][0]
y = marks['y'][0][0][0]


i=0
while(i<len(position)):                             # for all remeaining trails extract the respective sample values 
    
    x_MI=data[position[i]+0:position[i]+400:1]  # Taking the values from the traul position to tvhe next 750 values
    """Reshaping the data"""
    x_MI= x_MI.transpose()
    x_MI = x_MI[np.newaxis,:]  

    "applying bandpass"
    x_MI= butter_bandpass_filter(x_MI, lowcut, highcut, fs, order)
    
    """Concarting the values"""    
    X=np.append(X,x_MI,axis=0)   #Axis 0 stack on top of another
    i+=1



Y =y
print(X.shape)
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.25, random_state=22, shuffle= True)

""" Feature engineering"""
csp_alt= CSP(n_of_comp,reg=None, log=None, norm_trace=False)
csp_alt.fit(X_train,y_train)

dire = 'F:/1_Projects/continuous classification on dataset 1/'
filename = dire+'which- mi_CSP'+'.pkl'
pickle.dump(csp_alt, open(filename, 'wb'))



ft_train= csp_alt.fit_transform(X_train,y_train)
ft_test = csp_alt.fit_transform(X_test,y_test)

"""Feature selection"""
ft_best = SelectKBest(mutual_info_classif, k=n_of_ft).fit(ft_train,y_train)

dire = 'F:/1_Projects/continuous classification on dataset 1/'
filename = dire+'which- mi_kbest'+'.pkl'
pickle.dump(ft_best, open(filename, 'wb'))

ft_train= ft_best.fit_transform(ft_train,y_train)
ft_test= ft_best.fit_transform(ft_test,y_test)

print(ft_train.shape)
print(ft_test.shape)
"""  Classifer"""
linear = svm.SVC(kernel='rbf', C=1, decision_function_shape='ovo').fit(ft_train, y_train)
accuracy_lin = linear.score(ft_test, y_test)
#accuracy_lin = linear.score(ft_train, y_train)


print("\n\nAccuracy Linear Kernel:", accuracy_lin)                        # mean accuracy results
linear_pred = linear.predict(ft_test)
#linear_pred = linear.predict(ft_train)

cm_lin = confusion_matrix(y_test, linear_pred)    
#cm_lin = confusion_matrix(y_train, linear_pred)                            # confusion matrix  
print(cm_lin)
# plot_confusion_matrix(cm_lin,["Left hand","Right hand"])

keppa_svm_lin=cohen_kappa_score(y_test, linear_pred)                      # Cohen Kappa score
#keppa_svm_lin=cohen_kappa_score(y_train, linear_pred)                      # Cohen Kappa score
print(keppa_svm_lin)

test1 = ft_test[0]
test1 = test1.reshape(1, -1)
print(test1.shape)
print(linear.predict(test1))


dire = 'F:/1_Projects/continuous classification on dataset 1/'
filename = dire+'which- mi'+str(accuracy_lin)+'.sav'
pickle.dump(linear, open(filename, 'wb'))

