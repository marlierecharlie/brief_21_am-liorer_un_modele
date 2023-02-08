# vos données d'entraînement
# -*- coding: utf-8 -*-
"""
Example script

Script to perform some corrections in the brief audio project

Created on Fri Jan 27 09:08:40 2023

@author: ValBaron10
"""

# Import
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt
from features_functions import compute_features

from sklearn import preprocessing

# SVM
from sklearn import svm
from sklearn.svm import SVC

# cross-validation
from sklearn.model_selection import learning_curve, KFold


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

# permet de centraliser les données et les réduire 
scaler = preprocessing.StandardScaler(with_mean=True).fit(learning_features)
learningFeatures_scaled = scaler.transform(learning_features)


# initialisation de la validation croisée K-fold
kf = KFold(n_splits=4, shuffle=True, random_state=42)

X = learningFeatures_scaled
y = learning_labels


# Découper les données en ensemble d'entraînement et de validation
for train_index, test_index in kf.split(learning_features):
    X_train, X_test = learning_features[train_index], learning_features[test_index]
    y_train, y_test = learning_labels[train_index], learning_labels[test_index]

    # Entraîner le modèle SVM sur les données d'entraînement
    clf = SVC(C=10, gamma=0.001, kernel='linear')
    clf.fit(X_train, y_train)

    # Évaluer le modèle sur les données de validation
    accuracy = clf.score(X_test, y_test)
    print("Accuracy:", accuracy)

# Tracer la courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    SVC(kernel='linear', C=10), learning_features, learning_labels, cv=kf, scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score entraînement")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score validation")
plt.xlabel("Taille de l'ensemble d'entraînement")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show
