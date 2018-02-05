#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
store_data_reg_model.py
'''

import sys
sys.path.append('../job')

from sklearn.model_selection import cross_val_predict,KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import LinearSVR, SVR
from DIY_data import data_from_input
import pandas as pd


if __name__ == "__main__":
    X,Y = data_from_input()
    
    path = '/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/ensemble_data/HFT_data'
    filename = 'first_htf_data'
    
    with open('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/inf_log/reg_log.dat','r') as f:
        lines = f.readlines()
        for line in lines:
            if 'GradientBoostingRegressor_huber' in line:
                gbr_hu_par = line.split('found:')[1]
                gbr_hu_par = eval(gbr_hu_par)
            if 'GradientBoostingRegressor_lslad' in line:
                gbr_ld_par = line.split('found:')[1]
                gbr_ld_par = eval(gbr_ld_par)
            if 'Lasso' in lines:
                lasso_par = line.split('found:')[1]
                lasso_par = eval(lasso_par)
            if 'LinearSVR' in line:
                lrsvr_par = line.split('found:')[1]
                lrsvr_par = eval(lrsvr_par)
            if 'RandomForestRegressor' in line:
                rfr_par = line.split('found:')[1]
                rfr_par = eval(rfr_par)
            if 'Ridge' in line:
                ridge_par = line.split('found:')[1]
                ridge_par = eval(ridge_par)
            if 'SVR_linear' in line:
                svr_lr_par = line.split('found:')[1]
                svr_lr_par = eval(svr_lr_par)
            if 'SVR_rbf' in line:
                svr_rbf_par = line.split('found:')[1]
                svr_rbf_par = eval(svr_rbf_par)
    
    C_V = KFold(n_splits=5,random_state=0)
    reg_gbr_hu = GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8, loss=gbr_hu_par['loss'],
                                          n_estimators=gbr_hu_par['n_estimators'],max_depth=gbr_hu_par['max_depth'],
                                           max_features=gbr_hu_par['max_features'],alpha=gbr_hu_par['alpha']
                                          )
    reg_gbr_ld = GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8, loss=gbr_ld_par['loss'],
                                          n_estimators=gbr_ld_par['n_estimators'],max_depth=gbr_ld_par['max_depth'],
                                           max_features=gbr_ld_par['max_features']
                                          )
    reg_lasso = Lasso(alpha=lasso_par['alpha'])
    reg_lrsvr = LinearSVR(epsilon=lrsvr_par['epsilon'],C=lrsvr_par['C'],loss=lrsvr_par['loss'])
    reg_rfr = RandomForestRegressor(min_samples_split = 20, min_samples_leaf = 8,
                                    max_features=rfr_par['max_features'],n_estimators=rfr_par['n_estimators'])
    reg_ridge = Ridge(alpha=ridge_par['alpha'])
    reg_svr_lr = SVR(epsilon=svr_lr_par['epsilon'],C=svr_lr_par['C'],kernel=svr_lr_par['kernel'])
    reg_svr_rbf = SVR(epsilon=svr_rbf_par['epsilon'],C=svr_rbf_par['C'],kernel=svr_rbf_par['kernel'],gamma=svr_rbf_par['gamma'])
    
    gbr_hu_pred=cross_val_predict(reg_gbr_hu,X,Y,cv=C_V)
    gbr_ld_pred=cross_val_predict(reg_gbr_ld,X,Y,cv=C_V)
    lasso_pred=cross_val_predict(reg_lasso,X,Y,cv=C_V)
    lrsvr_pred=cross_val_predict(reg_lrsvr,X,Y,cv=C_V)
    rfr_pred=cross_val_predict(reg_rfr,X,Y,cv=C_V)
    ridge_pred=cross_val_predict(reg_ridge,X,Y,cv=C_V)
    svr_lr_pred=cross_val_predict(reg_svr_lr,X,Y,cv=C_V)
    svr_rbf_pred=cross_val_predict(reg_svr_rbf,X,Y,cv=C_V)
    
    total_ret = pd.DataFrame(index=Y.index)
    feature_names = ['gbr_hu_pred','gbr_ld_pred','lasso_pred','lrsvr_pred',
                     'rfr_pred','ridge_pred','svr_lr_pred','svr_rbf_pred','Y']
    for fn in feature_names:
        total_ret[fn]=eval(fn)
    
    total_ret.to_csv('%s/%s.csv'%(path,filename))
    
    
    
   
