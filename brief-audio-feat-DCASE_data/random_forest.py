from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt
from features_functions import compute_features

from sklearn import preprocessing

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# import de la grid search
from sklearn.model_selection import GridSearchCV

# pickle
import pickle

# Set the paths to the files 
data_path = "Data/"

# Names of the classes
classes_paths = ["Cars/", "Trucks/"]
classes_names = ["car", "truck"]
cars_list = [4,5,7,9,10,15,20,21,23,26,30,38,39,44,46,48,51,52,53,57]
trucks_list = [2,4,10,11,13,20,22,25,27,30,31,32,33,35,36,39,40,45,47,48]
nbr_of_sigs = 20 # Nbr of sigs in each class
seq_length = 0.2 # Nbr of second of signal for one sequence
nbr_of_obs = int(nbr_of_sigs*10/seq_length) # Each signal is 10 s long

# Go to search for the files
learning_labels = []
for i in range(2*nbr_of_sigs):
    if i < nbr_of_sigs:
        name = f"{classes_names[0]}{cars_list[i]}.wav"
        class_path = classes_paths[0]
    else:
        name = f"{classes_names[1]}{trucks_list[i - nbr_of_sigs]}.wav"
        class_path = classes_paths[1]

    # Read the data and scale them between -1 and 1
    fs, data = sio.wavfile.read(data_path + class_path + name)
    data = data.astype(float)
    data = data/32768

    # Cut the data into sequences (we take off the last bits)
    data_length = data.shape[0]
    nbr_blocks = int((data_length/fs)/seq_length)
    seqs = data[:int(nbr_blocks*seq_length*fs)].reshape((nbr_blocks, int(seq_length*fs)))

    for k_seq, seq in enumerate(seqs):
        # Compute the signal in three domains
        sig_sq = seq**2
        sig_t = seq / np.sqrt(sig_sq.sum())
        sig_f = np.absolute(np.fft.fft(sig_t))
        sig_c = np.absolute(np.fft.fft(sig_f))

        # Compute the features and store them
        features_list = []
        N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2], fs)
        features_vector = np.array(features_list)[np.newaxis,:]

        if k_seq == 0 and i == 0:
            learning_features = features_vector
            learning_labels.append(classes_names[0])
        elif i < nbr_of_sigs:
            learning_features = np.vstack((learning_features, features_vector))
            learning_labels.append(classes_names[0])
        else:
            learning_features = np.vstack((learning_features, features_vector))
            learning_labels.append(classes_names[1])

print(learning_features.shape)
print(len(learning_labels))

# Separate data in train and test
X_train, X_test, y_train, y_test = train_test_split(learning_features, learning_labels, test_size=0.3, random_state=42)

# Standardize the labels
labelEncoder = preprocessing.LabelEncoder().fit(y_train)
learningLabelsStd = labelEncoder.transform(y_train)
testLabelsStd = labelEncoder.transform(y_test)


                        
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, learningLabelsStd)
print(clf.predict(X_test, testLabelsStd))


y_pred = gnb.fit(X_train, learningLabelsStd).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (testLabelsStd != y_pred).sum()))
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(testLabelsStd, y_pred)
print("Accuracy:", accuracy)