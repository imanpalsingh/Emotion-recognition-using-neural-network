'''
 *
 * Author : - Imanpal Singh <imanpalsingh@gmail.com>
 * Description : -  Emotion Recognition from live camera feed.
 * Technology stack :- Numpy, Pandas, OpenCV, Scikit-Learn
 * Date created : - 11-10-2019
 * Last modified : - 19-11-2019
 * Version : - 0.2.1
 *
'''

'''
 *
 * Change log :

   (0.0.1) : - Algorithm : Now using Logistic Regression algorithm.
   (0.1.0) : - Algorithm : Now using Multi Layer Perceptron algorithm.
   
 * File Description : - This file uses the extracted and preprocessed data to train the model
 
'''

# Driver function
def Apply():
################################## IMPORTS ###################################

    print("Importing required libraries.")

    # Pandas 0.25.0
    import pandas as pd

    # Scikit-learn 0.21.3
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score



    # Numpy version 1.17.3
    import numpy as np

    #Joblib 0.13.2
    import joblib

    print("Loaded successfully")

    ############################### VARIABLES/LOADING DATA #################################

    print("Loading variables, data such as feature matrix, vector of prediction etc.")

    # Loading the datasets
    DATASET = pd.read_csv('Datasets/images.csv')
    TARGET = pd.read_csv('Datasets/target.csv')

    # Creating Feature Matrix
    X = DATASET.iloc[:,1:].values

    # Creating vector of prediction
    y = TARGET.iloc[:,1].values

    # Splitting Data into train and test set
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

    print("Done")

    ############################## MODEL TRAINING AND SAVING ############################

    print("Training the algorithm ")

    #Creating object of the algorithm class
    model = MLPClassifier(hidden_layer_sizes=[100,100],activation='relu',max_iter=2000,learning_rate='adaptive')

    # Training
    model.fit(X_train,y_train)

    print("Done")

    # Freezing the trained model

    print("Saving model")
    joblib.dump(model,'Models/trained_algorithm.pkl')
    print("Done")

    ################################# MODEl EVALUATION #############################

    # Predictng a input
    y_pred = model.predict(X_test)
    #Checking accuracy
    print("Model's score on train data {}".format(model.score(X_train,y_train)))
    print("Model's score on test data {}".format(model.score(X_test,y_test)))
    print("Model's confusion matrix\n{}".format(confusion_matrix(y_pred,y_test)))
    print("Model's precision score {}".format(precision_score(y_pred,y_test,average="micro")))
    print("Model's recall score {}".format(recall_score(y_pred,y_test,average="micro")))
    print("Model's f1 score {}".format(f1_score(y_pred,y_test,average="micro")))



if __name__ == '__main__':

    print(" Warning ! This file is not supposed to be run as main file ")




