# Test d'optimisation numéro 1 : la grid search 

j'ai testé une première grid search : 
--py
param_grid = {
    'C' : [.001, .01, 1, 10, 100], 'gamma': [.001, .01, .07], 'kernel' : ['linear', 'sigmoid', 'rbf', 'poly']}

Le problème est que le kernel poly mets beaucoup de temps à tourner. Et ce, pour un résultats qui n'est pas optimal (cf le test ci-dessous.)

--py
model = svm.SVC(C=.001, kernel='poly',gamma=.01, class_weight=None, probability=False)

index.png

La grid search actuellement utilisée est la suivante (dans le fichier SVM_opti_1_gridSearch.py)
--py

param_grid = {
    'C' : [.001, .01, 1, 10, 100],
    'gamma': [.001, .01, .07],
    'kernel' : ['linear', 'sigmoid', 'rbf']
             }


Cepedant, cette gridSearch a mis beaucoup de temps à se terminer, dans l'attente, j'ai noté les résultats les plus probants :
'''
    [CV 2/5] END C=0.01, gamma=0.001, kernel=linear;, score=0.700 total time=  10.9s
    [CV 4/5] END ..C=100, gamma=0.01, kernel=linear;, score=0.719 total time= 2.6min
    [CV 1/5] END C=0.001, gamma=0.001, kernel=linear;, score=0.720 total time=   1.4s
    [CV 1/5] END C=0.001, gamma=0.01, kernel=linear;, score=0.720 total time=   1.4s
    [CV 1/5] END C=0.001, gamma=0.07, kernel=linear;, score=0.720 total time=   1.4s
    [CV 1/5] END ...C=1, gamma=0.001, kernel=linear;, score=0.720 total time= 2.2min


    [CV 5/5] END C=0.01, gamma=0.001, kernel=linear;, score=0.716 total time=  11.5s


    [CV 5/5] END ...C=1, gamma=0.001, kernel=linear;, score=0.726 total time= 1.7min
    [CV 1/5] END ..C=10, gamma=0.001, kernel=linear;, score=0.723 total time= 2.8min

    [CV 1/5] END .C=0.01, gamma=0.07, kernel=linear;, score=0.733 total time=  10.4s
    [CV 1/5] END ..C=100, gamma=0.01, kernel=linear;, score=0.748 total time= 3.2min
    [CV 1/5] END ..C=100, gamma=0.07, kernel=linear;, score=0.748 total time= 3.5min
'''

On voit ici que les résultats les plus précis sont obtenus avec un C élevé, un kernel linéaire... Mais qu'ils prennent beaucoup de temps à apprendre. Paradoxalement, quand le C est bas et le gamma également, la précision est un peu moins bonne (de l'ordre de .72) mais bien plus rapide (1.4 secondes). Il serait intéressant de voir si ce type de modèle n'overfit pas. 

Dans le cas ou on utilise un C d'une valeur élevée, j'aurai tendance à réduire le nombre de valeurs afin de rendre son exécution plus rapide. 

Suite à la gridSearch, les meilleurs paramètres sont les suivants : 
'''
    rgb(255, 0, 0)
    SVC(C=100, gamma=0.001, kernel='linear')
'''

# Cross-validation
Avec la méthode K-fold, l'accuracy obtenue pour ces paraètres est de :
'''
    Accuracy: 0.7887323943661971
    Accuracy: 0.795774647887324
    Accuracy: 0.7703180212014135
    Accuracy: 0.7667844522968198
    Accuracy: 0.7632508833922261
    Accuracy: 0.7915194346289752
    Accuracy: 0.773851590106007
    moyenne : 0.778
    équart-type : 0.012
'''
Cette méthode que le modèle est peu précis et de plus très long à tourner (30 minutes)

Les paramètres du k-fold ont été sauvegardés dans un .pkl. 

# test d'over fitting

La courbe d'overfitting est très longue à obtenir, avec le temps imparti pour compléter ce brief, j'ai décidé de ne pas aller au bout de se fichier.

# Autres modèles

Le SVM n'a pas été amélioré avec l'utilisation de la grid Search. Je décide donc de tester d'autres modèles de 
classification :

_**Régression logistique** : pour l'instant j'ai un message d'erreurs concernant le nombre d'iterations, 100, 200 et 1000 ne sont pas suffisantes, ni même 3000. J'ai essayé d'appliquer les solver *'sag'*, *'liblinear'* et *'lbfgs'*. Avec des paramètres adaptés aux données multiclasses, la max_iteration n'est pas nécessaire.
''' clf = LogisticRegression(random_state=0, solver='sag')
    accuraccy : 0.63

    clf = LogisticRegression(random_state=0, solver='liblinear')
    accuracy : 0.70
'''

_**Naïves Bayes** :

    le nb multinomial ne support pas les X négatifs, j'ai donc décidé de ne tester que le gaussien. 

    '''
    gaussien 
    Number of mislabeled points out of a total 397 points : 154
    Accuracy: 0.612
    '''

_**random forest** : 

j'aurai aimé tester le stakking ensuite 




