import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd
import numpy as np


class Data:
	_X = None
	_Y = None
	
	def __init__(self,fileName,header = False,algo = 1):
		if header:
			data = pd.read_csv(fileName)
		
		else:
			data = pd.read_csv(fileName,header=None)
		# Extract the no of features
		self.noOfFeatures = data.shape[1]
		# Extract all the features to _X
		self._X = data.values[:,0:self.noOfFeatures-1]
		# Extract all the class labels to _Y
		self._Y = data.values[:,self.noOfFeatures-1]
		# Extract the Accuracy of the set
		self.accuracy = self.getAccuracy(self._X,algo)
		print('Accuracy on all the Attributes : ',self.accuracy)
	
	
	def getAccuracy(self,features,algo = 1):
		# Split the data into testing and training
		X_train,X_test,Y_train,Y_test = train_test_split(features,self._Y,test_size = 0.2)
		if algo == 1:
			# create classifier
			clf = DecisionTreeClassifier(criterion = 'gini')
			# Fit the data into classifier
			clf.fit(X_train,Y_train)
			# Predict the results  
			prediction = clf.predict(X_test)
			incorrectPrediction = (Y_test != prediction).sum()
			#print('Incorrect Prediction ',incorrectPrediction)
			return accuracy_score(Y_test,prediction) * 100
			
		elif algo == 2:
			gnb = GaussianNB()
			prediction = gnb.fit(X_train,Y_train).predict(X_test)
			incorrectPrediction = (Y_test != prediction).sum()
			#print('Incorrect Prediction ',incorrectPrediction)
			return accuracy_score(Y_test,prediction) * 100
		
		elif algo == 3:
			neigh = KNeighborsClassifier(n_neighbors = 3)
			neigh.fit(X_train,Y_train)
			prediction = neigh.predict(X_test)
			incorrectPrediction = (Y_test != prediction).sum()
			#print('Incorrect Prediction ',incorrectPrediction)
			return accuracy_score(Y_test,prediction) * 100
	
	def data_cleaning(self,vector):
		if np.count_nonzero(vector) == 0:
			X_subset = np.copy(self._X)
			
		else:
			X_subset = np.copy(self._X[:,vector==1])
		return X_subset
	

	def getDimension(self):
		return self.noOfFeatures-1