#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_reg_pipeline.py
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import LinearSVR, SVR
from DIY_data import data_from_input
from sklearn.decomposition import PCA, IncrementalPCA


if __name__ == "__main__":
    X,Y = data_from_input()
    
    with open('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/chk_jobs/cla_log.dat','r') as f:
        lines = f.readlines()
        for line in lines:
            if 'GradientBoostingRegressor_huber' in line:
                gbr_hu_par = line.split('found:')[1]
                gbr_hu_par = eval(gbr_hu_par)
            if 'GradientBoostingRegressor_lslad' in line:
                gbr_ld_par = line.split('found:')[1]
                gbr_ld_par = eval(gbr_ld_par)
            if 'Lasso' in lines:
                laso_par = line.split('found:')[1]
                laso_par = eval(laso_par)
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
            if 'IncrementalPCA' in line:
                ipca_par = line.split('DIY')[1]
                ipca_par = eval(ipca_par)
                
    pca = IncrementalPCA(n_components=ipca_par['99%'])
    gbc = GradientBoostingClassifier()
    lg =  LogisticRegression()
    rfc = RandomForestClassifier(rfc_par)
    svc = SVC(svc_par)
    lrsvc = LinearSVC(lrsvc_par)
    steps = [('PCA',pca)
            ]