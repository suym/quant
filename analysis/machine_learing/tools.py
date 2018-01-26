#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
My tools 
"""
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"


import pandas as pd
import numpy as np
import tushare as ts
import datetime as dt
from feature import SMA,EWMA,BBANDS,CCI,EVM,ForceIndex,ROC
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso,LogisticRegression)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import RFE, f_regression
from minepy import MINE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
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
    ts_hs = ts.get_hist_data(
                            symbol,
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
    indexs = [("SMA5", SMA(ts_hs300,5)),
              ("SMA10", SMA(ts_hs300,10)),
              ("SMA20", SMA(ts_hs300,20)),
              ("EWMA_20", EWMA(ts_hs300,20)),
              ("BBANDS", BBANDS(ts_hs300,20)),
              ("CCI", CCI(ts_hs300,20)),
              ("EVM", EVM(ts_hs300,20)),
              ("ForceIndex", ForceIndex(ts_hs300,20)),
              ("ROC", ROC(ts_hs300,20)),
                ]
    for ind in indexs:
        tsret_1 = ind[1]
        tsret=tsret.join(tsret_1)

    tsret = tsret[tsret.index >= str(start_date)]

    # To remove a null value
    tsret=tsret.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    return tsret

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

def features_com(X, Y):

    ranks=pd.DataFrame(index=X.columns)

    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)
    ranks["Linear reg"] = rank_to_list(np.abs(lr.coef_))

    ridge = Ridge(alpha=7)
    ridge.fit(X, Y)
    ranks["Ridge"] = rank_to_list(np.abs(ridge.coef_))

    lasso = Lasso(alpha=.05)
    lasso.fit(X, Y)
    ranks["Lasso"] = rank_to_list(np.abs(lasso.coef_))

    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X, Y)
    ranks["Stability"] = rank_to_list(np.abs(rlasso.scores_))

    #stop the search when 5 features are left (they will get equal scores)
    rfe = RFE(lr, n_features_to_select=5)
    rfe.fit(X,Y)
    ranks["RFE"] = rank_to_list(map(float, rfe.ranking_), order=-1)

    rf = RandomForestRegressor()
    rf.fit(X,Y)
    ranks["RF"] = rank_to_list(rf.feature_importances_)

    f, pval  = f_regression(X, Y, center=True)
    ranks["Corr."] = rank_to_list(f)

    mine = MINE()
    mic_scores = []
    for i in range(X.shape[1]):
        mine.compute_score(X.iloc[:,i], Y)
        m = mine.mic()
        mic_scores.append(m)

    ranks["MIC"] = rank_to_list(mic_scores)
    
    return ranks

def GS_LogisticRegression(*data):
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'penalty':['l1','l2'], 'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                        'solver':['liblinear']},

                        {'penalty':['l2',], 'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                        'solver':['lbfgs']}
                        ]
    clf =GridSearchCV(LogisticRegression(tol = 1e-6), tuned_parameters, cv =10, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)
    print "Detailed classification report: "
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true,y_pred)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_LinearSVC(*data):
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'penalty':['l2'], 'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                        'loss':['hinge','squared_hinge']}
                        ]
    clf =GridSearchCV(LinearSVC(tol = 1e-6), tuned_parameters, cv =10, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)
    print "Detailed classification report: "
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true,y_pred)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_SVC(*data):
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                        'kernel':['linear']},

                        {'C':[0.01,0.1,0.5,1,5,10,50,100],'kernel':['rbf'],
                        'gamma':[0.01,0.1,0.5,5,10,50]}
                        ]
    clf =GridSearchCV(SVC(tol = 1e-6), tuned_parameters, cv =10, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)
    print "Detailed classification report: "
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true,y_pred)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_RandomForestRegressor(*data):
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'n_estimators':[10,50,200,500,1000],'max_depth':[1,3,5,10],
                        'max_features':[0.01,0.1,0.6,1]}
                        ]
    clf =GridSearchCV(RandomForestRegressor(), tuned_parameters, cv =10, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)
    print "Detailed classification report: "
    y_true, y_pred = y_test, clf.predict(X_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_Lasso(*data):
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'alpha':[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,50,100,200,500,10000]}]
    
    clf =GridSearchCV(Lasso(tol = 1e-6), tuned_parameters, cv =10, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)
    print "Detailed classification report: "
    y_true, y_pred = y_test, clf.predict(X_test)

    return clf.best_params_, clf.score(X_test,y_test)

def GS_Ridge(*data):
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'alpha':[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,50,100,200,500,10000]}]
    
    clf =GridSearchCV(Ridge(tol = 1e-6), tuned_parameters, cv =10, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)
    print "Detailed classification report: "
    y_true, y_pred = y_test, clf.predict(X_test)

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

