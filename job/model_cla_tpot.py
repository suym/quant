#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_cla_tpot.py
'''

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from DIY_data import data_from_input

if __name__ == "__main__":
    X,Y = data_from_input()
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25,stratify=Y)
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)

    tpot.export('./TPOT_report/tpot_class_pipeline.py')
    print 'The scores of test set %s'%tpot.score(X_test, y_test)
    
    