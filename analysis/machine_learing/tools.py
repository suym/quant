#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
My tools 
"""
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import os
import sys
import pandas as pd
import numpy as np
import tushare as ts
import datetime as dt
from feature import TY_SMA,TY_EWMA,TY_BBANDS,TY_CCI,TY_EVM,TY_ForceIndex,TY_ROC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC, LinearSVR, SVR
from sklearn.feature_selection import RFE, f_regression, chi2, f_classif 
from minepy import MINE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA, IncrementalPCA



# ---------------------------------------------
# Function 
# ---------------------------------------------

def get_hs300_symbols():
    """
    get a list of symbols of hs300
    from tushare.
    """
    symbol = ts.get_hs300s()
    
    return symbol['code'], symbol['name']

def readstkData(rootpath, stockcode):

    filename = rootpath+ '/' + stockcode + '.csv'
    returndata = pd.read_csv(filename)

    return returndata

def create_lagged_series(symbol, start_date, end_date):
    """
    This creates a pandas DataFrame that stores the 
    close value of a stock obtained from Tushare
    """

    # Obtain stock information from Tushare
    ts_hs = ts.get_hist_data(symbol,
                             str(start_date-dt.timedelta(days=365)),
                             str(end_date)
                            )
    # Descending by date
    ts_hs300 = ts_hs.sort_index()

    # Create the new returns DataFrame
    tsret = pd.DataFrame(index=ts_hs300.index)
    tsret["price_change"] = ts_hs300['price_change'].shift(-1)
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["cla_Direction"] = np.sign(tsret["price_change"])
    tsret["reg_Direction"] = ts_hs300['close'].shift(-1)
    # Create the other colum
    features = ['open','high','close','low','volume']
    for fe in features:
        tsret[fe] = ts_hs300[fe]
    indexs = [("SMA5", TY_SMA(ts_hs300,5)), ("SMA10", TY_SMA(ts_hs300,10)),("SMA20", TY_SMA(ts_hs300,20)),
              ("EWMA_20", TY_EWMA(ts_hs300,20)),("BBANDS", TY_BBANDS(ts_hs300,20)),("CCI", TY_CCI(ts_hs300,20)),
              ("EVM", TY_EVM(ts_hs300,20)), ("ForceIndex", TY_ForceIndex(ts_hs300,20)),("ROC", TY_ROC(ts_hs300,20))
                ]
    for ind in indexs:
        tsret_1 = ind[1]
        tsret=tsret.join(tsret_1)

    tsret = tsret[tsret.index >= str(start_date)]

    # To remove a null value
    tsret=tsret.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    
    return tsret

def store_data(symbol, start_date, end_date):
    sto_data = create_lagged_series(symbol, start_date, end_date)
    date1 = start_date.strftime("%Y%m%d")
    date2 = end_date.strftime("%Y%m%d")
    filename = symbol+'/'+date1+'__'+date2
    filepath = '/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/tushare_data'
    path = filepath+'/'+filename
    if path != '' and not os.access(path, os.F_OK) :
        sys.stdout.write('Creating dir %s ...\n'  % path)
        os.makedirs(path)
    sto_data.to_csv('%s/%s.csv'%(path,symbol))
    
    #A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
        
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

def rank_to_list(ranks, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    
    return ranks

def features_com_reg(X, Y):
    ranks=pd.DataFrame(index=X.columns)

    lr = LinearRegression(normalize=True,n_jobs=-1)
    lr.fit(X, Y)
    ranks["Linear reg"] = rank_to_list(np.abs(lr.coef_))

    f, pval  = f_regression(X, Y, center=True)
    ranks["f_regression Corr."] = rank_to_list(f)

    mine = MINE()
    mic_scores = []
    for i in range(X.shape[1]):
        mine.compute_score(X.iloc[:,i], Y)
        m = mine.mic()
        mic_scores.append(m)

    ranks["MIC"] = rank_to_list(mic_scores)
    
    return ranks

def features_com_cla(X, Y):
    ranks=pd.DataFrame(index=X.columns)

    f_1, pval_1  = chi2(X, Y)
    ranks["chi2 Corr."] = rank_to_list(f_1)

    f_2, pval_2  = f_classif(X, Y)
    ranks["f_claaif Corr."] = rank_to_list(f_2)

    mine = MINE()
    mic_scores = []
    for i in range(X.shape[1]):
        mine.compute_score(X.iloc[:,i], Y)
        m = mine.mic()
        mic_scores.append(m)

    ranks["MIC"] = rank_to_list(mic_scores)
    
    return ranks

# ---------------------------------------------
# Function of classification
# ---------------------------------------------

def GS_LogisticRegression(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'penalty':['l1','l2'], 'C':[0.01,1],
                        'solver':['liblinear']}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(LogisticRegression(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_penalty = best_par_1['penalty']
    
    tuned_parameters = [{'penalty':[best_par_penalty], 'C':[0.001,0.01,0.2,1,8,50],
                        'solver':['liblinear']}
                        ]
    clf =GridSearchCV(LogisticRegression(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
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

def GS_LinearSVC(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'penalty':['l2'], 'C':[1,0.1],
                        'loss':['hinge','squared_hinge']}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(LinearSVC(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_loss = best_par_1['loss']
    
    tuned_parameters = [{'penalty':['l2'], 'C':[0.001,0.01,0.2,1,8,50],
                        'loss':[best_par_loss]}
                        ]
    clf =GridSearchCV(LinearSVC(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
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

def GS_SVC_linear(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'C':[0.001,0.01,0.2,1,8,50],
                        'kernel':['linear']}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf =GridSearchCV(SVC(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_SVC_rbf(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'C':[1],'kernel':['rbf'],
                        'gamma':[0.001,0.01]}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(SVC(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_gamma = best_par_1['gamma']
    
    tuned_parameters = [{'C':[0.1,1],'kernel':['rbf'],
                        'gamma':[best_par_gamma]}
                        ]
    clf =GridSearchCV(SVC(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
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

def GS_RandomForestClassifier(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'n_estimators':[300],
                        'max_features':[0.5,0.8,1]}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(RandomForestClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_f = best_par_1['max_features']
    
    tuned_parameters = [{'n_estimators':[400],'max_depth':[best_par_d],
                        'max_features':[best_par_f]}
                        ]
    clf =GridSearchCV(RandomForestClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
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

def GS_GradientBoostingClassifier(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'n_estimators':[200],'max_depth':[5],
                        'max_features':[0.5],'loss':['deviance','exponential']}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(GradientBoostingClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_loss = best_par_1['loss']
    
    tuned_parameters_2 = [{'n_estimators':[200],'max_depth':[5,10,15,20],
                        'max_features':[0.5],'loss':[best_par_loss]}
                        ]
    clf_2 =GridSearchCV(GradientBoostingClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_d = best_par_2['max_depth']
    
    tuned_parameters_3 = [{'n_estimators':[200],'max_depth':[best_par_d],
                        'max_features':[0.2,0.5,0.8],'loss':[best_par_loss]}
                        ]
    clf_3 =GridSearchCV(GradientBoostingClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_3, cv =C_V, n_jobs=-1)
    clf_3.fit(X_train,y_train)
    best_par_3 = clf_3.best_params_
    best_par_f = best_par_3['max_features']
    
    tuned_parameters = [{'n_estimators':[400],'max_depth':[best_par_d],
                        'max_features':[best_par_f],'loss':[best_par_loss]}
                        ]
    clf =GridSearchCV(GradientBoostingClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_3, mean_score_3, scores_3, in clf_3.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_3, scores_3.std()*2,params_3)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)

# ---------------------------------------------
# Function of regression
# ---------------------------------------------

def GS_Lasso(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'alpha':[0.0005,0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,1,5,10,20,30,50,70]}]
    
    C_V = KFold(n_splits=5,random_state=0)
    clf =GridSearchCV(Lasso(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_Ridge(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'alpha':[0.0005,0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,1,5,10,20,30,50,70]}]
    
    C_V = KFold(n_splits=5,random_state=0)
    clf =GridSearchCV(Ridge(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_LinearSVR(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'epsilon':[0.06], 'C':[1],
                        'loss':['epsilon_insensitive','squared_epsilon_insensitive']}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(LinearSVR(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_loss = best_par_1['loss']
    
    tuned_parameters_2 = [{'epsilon':[0.06], 'C':[0.001,0.01,0.1,1,10,50],
                        'loss':[best_par_loss]}
                        ]
    clf_2 =GridSearchCV(LinearSVR(tol = 1e-6), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_c = best_par_2['C']
    
    tuned_parameters = [{'epsilon':[0.005,0.06,0.1,0.8,5], 'C':[best_par_c],
                        'loss':[best_par_loss]}
                        ]
    clf =GridSearchCV(LinearSVR(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_SVR_linear(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'epsilon':[0.005,0.06],'C':[1],
                        'kernel':['linear']}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(SVR(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_eps = best_par_1['epsilon']
    
    tuned_parameters = [{'epsilon':[best_par_eps],'C':[0.1,1],
                        'kernel':['linear']}
                        ]
    clf =GridSearchCV(SVR(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
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

def GS_SVR_rbf(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    
    tuned_parameters_2 = [{'epsilon':[0.005,0.06],'C':[1],'kernel':['rbf'],
                        'gamma':[0.001]}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_2 =GridSearchCV(SVR(tol = 1e-6), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_eps = best_par_2['epsilon']
    
    tuned_parameters = [{'epsilon':[best_par_eps],'C':[0.1,1],'kernel':['rbf'],
                        'gamma':[0.001]}
                        ]
    clf =GridSearchCV(SVR(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_RandomForestRegressor(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    
    tuned_parameters_1 = [{'n_estimators':[300],
                        'max_features':[0.5,0.8,1]}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(RandomForestRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_f = best_par_1['max_features']
    
    tuned_parameters = [{'n_estimators':[400],
                        'max_features':[best_par_f]}
                        ]
    clf =GridSearchCV(RandomForestRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
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

def GS_GradientBoostingRegressor_lslad(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'n_estimators':[200],'max_depth':[5],
                        'max_features':[0.5],'loss':['ls','lad']}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_loss = best_par_1['loss']
    
    tuned_parameters_2 = [{'n_estimators':[200],'max_depth':[5,10,15,20],
                        'max_features':[0.5],'loss':[best_par_loss]}
                        ]
    clf_2 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_d = best_par_2['max_depth']
    
    tuned_parameters_3 = [{'n_estimators':[200],'max_depth':[best_par_d],
                        'max_features':[0.2,0.5,0.8],'loss':[best_par_loss]}
                        ]
    clf_3 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_3, cv =C_V, n_jobs=-1)
    clf_3.fit(X_train,y_train)
    best_par_3 = clf_3.best_params_
    best_par_f = best_par_3['max_features']
    
    tuned_parameters = [{'n_estimators':[400],'max_depth':[best_par_d],
                        'max_features':[best_par_f],'loss':[best_par_loss]}
                        ]
    clf =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_3, mean_score_3, scores_3, in clf_3.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_3, scores_3.std()*2,params_3)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_GradientBoostingRegressor_huber(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'n_estimators':[200],'max_depth':[5,10,15,20],
                        'max_features':[0.5],'loss':['huber'],'alpha':[0.9]}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_d = best_par_1['max_depth']
    
    tuned_parameters_2 = [{'n_estimators':[200],'max_depth':[best_par_d],
                        'max_features':[0.2,0.5,0.8],'loss':['huber'],'alpha':[0.9]}
                        ]
    clf_2 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_f = best_par_2['max_features']
    
    tuned_parameters_3 = [{'n_estimators':[200],'max_depth':[best_par_d],
                        'max_features':[best_par_f],'loss':['huber'],'alpha':[0.2,0.5,0.9]}
                        ]
    clf_3 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_3, cv =C_V, n_jobs=-1)
    clf_3.fit(X_train,y_train)
    best_par_3 = clf_3.best_params_
    best_par_alpha = best_par_3['alpha']
    
    tuned_parameters = [{'n_estimators':[400],'max_depth':[best_par_d],
                        'max_features':[best_par_f],'loss':['huber'],'alpha':[best_par_alpha]}
                        ]
    clf =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_3, mean_score_3, scores_3, in clf_3.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_3, scores_3.std()*2,params_3)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_PCA(data):
    X = data
    num1 = 0.99
    num2 = 0.98
    num3 = 0.97
    num4 = 0.95
    sum_t = 0
    count = 0
    ret = {}
    pca = PCA(n_components=None)
    pca.fit(X)
    ratios = pca.explained_variance_ratio_
    for ratio in ratios:
        sum_t = sum_t + ratio
        count = count + 1
        if sum_t <= num4:
            ret['95%'] = count
        if sum_t <= num3:
            ret['97%'] = count
        if sum_t <= num2:
            ret['98%'] = count
        if sum_t <= num1:
            ret['99%'] = count
    return pca.n_components_, pca.explained_variance_ratio_, ret

def GS_IncrementalPCA(data):
    X = data
    num1 = 0.99
    num2 = 0.98
    num3 = 0.97
    num4 = 0.95
    sum_t = 0
    count = 0
    ret = {}
    pca = IncrementalPCA(n_components=None)
    pca.fit(X)
    ratios = pca.explained_variance_ratio_
    for ratio in ratios:
        sum_t = sum_t + ratio
        count = count + 1
        if sum_t <= num4:
            ret['95%'] = count
        if sum_t <= num3:
            ret['97%'] = count
        if sum_t <= num2:
            ret['98%'] = count
        if sum_t <= num1:
            ret['99%'] = count
    return pca.n_components_, pca.explained_variance_ratio_, ret

