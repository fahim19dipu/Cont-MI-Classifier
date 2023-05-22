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
import time

from moviepy.editor import VideoFileClip

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
highcut =35
fs= 100
order = 4

matlabfile = loadmat(filename,struct_as_record=True, squeeze_me=False)            # loading matfile
data = matlabfile['cnt']         

# print(y.shape)

i=0

base = "F:/1_Projects/continuous classification on dataset 1/"
filename = base +"mi-nomi_CSP.pkl"
csp_1 = pickle.load(open(filename, 'rb'))
filename = base +"which- mi_CSP.pkl"
csp_2 = pickle.load(open(filename, 'rb'))

filename =base + "mi-nomi_kbest.pkl"
kbest_1 = pickle.load(open(filename, 'rb'))
filename = base +"which- mi_kbest.pkl"
kbest_2 = pickle.load(open(filename, 'rb'))

filename = base +"mi1.0.sav"
cls_1 = pickle.load(open(filename, 'rb'))

filename = base +"which- mi0.92.sav"
cls_2 = pickle.load(open(filename, 'rb'))
total_vids=[]
while(i+400<=190400):                             # for all remeaining trails extract the respective sample values 
    
    print(i)
    x_MI=data[i:i+400:1]  # Taking the values from the traul position to the next 400 values
    
    x_MI= x_MI.transpose()
    x_MI = x_MI[np.newaxis,:]  
    
    "applying bandpass"
    x_MI= butter_bandpass_filter(x_MI, lowcut, highcut, fs, order)    

    """ Feature engineering"""
    
    #print("csp")
    # print(x_MI.shape)
    ft_train= csp_1.transform(x_MI)
    # print(ft_train.shape)
    ft_train= kbest_1.transform(ft_train)
    # print(ft_train.shape)
    
    ft_train = ft_train.reshape(1, -1)
    res1 = cls_1.predict(ft_train)
    #print(f"First Classifier result {res1} ")

    if res1==1.0:
        i+=200
        continue
    #print("Did not continue")
    ft_train= csp_2.transform(x_MI)
    ft_train= kbest_2.transform(ft_train)
    # print(ft_train.shape)
    
    ft_train = ft_train.reshape(1, -1)
    #print(ft_train.shape)
    res2 = cls_2.predict(ft_train)
    print(f"Second Classifier result {res2[0]} ")
    
    if res2[0] == 1:
        vid_filename = base + "waving_left.mkv"
    else:
        vid_filename = base + "waving_right_hand.mkv"
    total_vids.append(vid_filename)
    #clip = VideoFileClip(vid_filename)
    #clip.preview()
    
    i += 800
print(len(total_vids))
   # time.sleep(1)


    #time.sleep(1)


#print(len(X))