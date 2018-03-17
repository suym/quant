#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
DL_Package.py 
"""
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold

def make_reg_LSTM(nb_time_steps,data_dim):
    model = Sequential()
    model.add(LSTM(64,return_sequences=True,input_shape=(nb_time_steps,data_dim),kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(LSTM(64,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    return model

def make_reg_MFNN(data_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=data_dim,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    return model
                       
def GS_reg_LSTM(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    X_dim = X_train.shape[1]
    X_time_steps = 1
    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_time_steps, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_time_steps, X_test.shape[1]))
                                 
    my_lstm = KerasRegressor(build_fn=make_reg_LSTM)
                            
    tuned_parameters_1 = [{'nb_epoch':[30],'batch_size':[64,128,256],
                          'nb_time_steps': [X_time_steps],'data_dim':[X_dim]}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(my_lstm, tuned_parameters_1, cv =C_V, n_jobs=1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_bs = best_par_1['batch_size']
    
    tuned_parameters = [{'nb_epoch':[30,50],'batch_size':[best_par_bs],
                          'nb_time_steps': [X_time_steps],'data_dim':[X_dim]}
                        ]
    clf =GridSearchCV(my_lstm, tuned_parameters, cv =C_V, n_jobs=1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)  

def GS_reg_MFNN(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    X_dim = X_train.shape[1]
                                 
    my_mfnn = KerasRegressor(build_fn=make_reg_MFNN)
                            
    tuned_parameters_1 = [{'nb_epoch':[30],'batch_size':[64,128,256],
                          'data_dim':[X_dim]}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(my_mfnn, tuned_parameters_1, cv =C_V, n_jobs=1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_bs = best_par_1['batch_size']
    
    tuned_parameters = [{'nb_epoch':[30,50],'batch_size':[best_par_bs],
                         'data_dim':[X_dim]}
                        ]
    clf =GridSearchCV(my_mfnn, tuned_parameters, cv =C_V, n_jobs=1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)
                  
                            
    
