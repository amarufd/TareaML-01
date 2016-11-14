##!
import sys
sys.path.append('/Users/amaru/Desktop/Diplomado_bigData/Tarea_ML/CodigosPython')   
import AsaUtils

import time;

import initOptions
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt


try:
    pdf = sys.argv[7]
except IndexError:
    pdf = False

try:
    learning_rate_init = sys.argv[6]
except IndexError:
    learning_rate_init = '0.001'

try:
    capaOcultaB = sys.argv[5]
except IndexError:
    capaOcultaB = '100'

try:
    capaOcultaA = sys.argv[4]
except IndexError:
    capaOcultaA = '100'

solve =["lbfgs","sgd","adam"]

try:
    selSolve = int(sys.argv[3])
except IndexError:
    selSolve = 0

try:
    selNumSet = int(sys.argv[2])
except IndexError:
    selNumSet = 10

#Section 0. Define Path to seek features
RootDescPath='MIT-'+str(selNumSet)+'-DescriptorsPath'

option =["KNN","SVM","NNS"]

print "KNN -> 1\nSVM -> 2\nNNS -> 3"

if sys.argv[1].isdigit():
    var = int(sys.argv[1])
else:
    var = 0

while var < 1 or var > 3 :  # This constructs an infinite loop
	var = int(input("ESeleccione una opcion: "))


timeStart = time.time()
print "Opcion escogida: ", option[var-1], ' - Solver: ', solve[selSolve], ' - CapaOcultaA: ', capaOcultaA, ' - CapaOcultaB: ', capaOcultaB, ' - Learning Rate Init: ', learning_rate_init

#Section 1. Define classifiers to use in this activity
classifiersDefs = {'KNN':"KNeighborsClassifier(n_neighbors=1, metric='euclidean', weights='distance')",
                   'SVM':"svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr')",
                   'NNS':"MLPClassifier(solver='"+solve[selSolve]+"', alpha=1e-5, hidden_layer_sizes=("+capaOcultaA+", "+capaOcultaB+"), learning_rate='constant', learning_rate_init="+learning_rate_init+", shuffle=True, random_state=1)"}

#Section 2. Init libraries and main options
options=initOptions.selections()
 
#Section 3. Read training data
descriptorsPath=options[RootDescPath] + 'TrainSet'
classNames,trainFeats,trainLabels=AsaUtils.getMatlabDescriptors(descriptorsPath, 'mat',4096)

#Section 4. Train Classifier
print('******Training classifier, it can take time !, be patient !')
classifier = eval(classifiersDefs[option[var-1]])

classifier.fit(trainFeats,trainLabels)
    
#Section 5. Read test data
descriptorsPath=options[RootDescPath] + 'TestSet'
classNames,testFeats,testLabels=AsaUtils.getMatlabDescriptors(descriptorsPath, 'mat',4096)

#Section 6. Apply Classifier to test data and calculate accuracy 
print('******Testing classifier, it can take time !, be patient !')
predictedLabels=classifier.predict(testFeats)
accuracy=accuracy_score(testLabels, predictedLabels)
print 'Classification accuracy on test set: ' + str(100*round(accuracy,2)) + '%'

#Section 7. Apply Classifier to train data and calculate accuracy 
predictedTrainLabels=classifier.predict(trainFeats)
accuracy=accuracy_score(trainLabels, predictedTrainLabels)
print 'Classification accuracy on training set: ' + str(100*round(accuracy,2)) + '%'

if pdf:
    #Section 8. Computer and show confusion matrix on test set
    cnf_matrix = confusion_matrix(testLabels, predictedLabels)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    AsaUtils.plot_confusion_matrix(cnf_matrix, classes=range(1,len(classNames)+1),
                           title='Confusion matrix, without normalization')
    
    pp = PdfPages(option[var-1]+'/'+str(selNumSet)+'/NroTraining_'+str(selNumSet)+'-Machine_'+option[var-1]+'-Solve_'+solve[selSolve]+'-CapaOcultaA_'+capaOcultaA+'-CapaOcultaB_'+capaOcultaB+'-learning_rate_init_'+learning_rate_init+'.pdf')
    plt.savefig(pp, format='pdf')
     
    #Plot normalized confusion matrix
    plt.figure()
    AsaUtils.plot_confusion_matrix(cnf_matrix, classes=range(1,len(classNames)+1), normalize=True,
                           title='Normalized confusion matrix')
     
    #plt.show()
    
    plt.savefig(pp, format='pdf')
    pp.close()





time_total = (time.time() - timeStart) #En mili segundos


minuto = int(time_total / 60)
segundos = int(time_total % 60)

print "tiempo de ejecucion: "+str(time_total)+"\n"+str(minuto)+" Minutos - "+str(segundos)+" Segundos"

