#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_cla_pipeline.py
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from DIY_data import data_from_input
from sklearn.decomposition import PCA, IncrementalPCA


if __name__ == "__main__":
    X,Y = data_from_input()
    
    with open('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/inf_log/cla_log.dat','r') as f:
        lines = f.readlines()
        for line in lines:
            if 'GradientBoostingClassifier' in line:
                gbc_par = line.split('found:')[1]
                gbc_par = eval(gbc_par)
            if ' LogisticRegression' in line:
                lg_par = line.split('found:')[1]
                lg_par = eval(lg_par)
            if 'RandomForestClassifier' in lines:
                rfc_par = line.split('found:')[1]
                rfc_par = eval(rfc_par)
            if 'SVC' in line:
                svc_par = line.split('found:')[1]
                svc_par = eval(svc_par)
            if 'LinearSVC' in line:
                lrsvc_par = line.split('found:')[1]
                lrsvc_par = eval(lrsvc_par)
            if 'IncrementalPCA' in line:
                ipca_par = line.split('DIY')[1]
                ipca_par = eval(ipca_par)
                
    pca = IncrementalPCA(n_components=nums)
    gbc = GradientBoostingClassifier()
    lg =  LogisticRegression()
    rfc = RandomForestClassifier(rfc_par)
    svc = SVC(svc_par)
    lrsvc = LinearSVC(lrsvc_par)
    steps = [('PCA',pca)
            ]
