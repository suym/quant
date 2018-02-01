#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_reg_Ridge.py
'''

from sklearn.model_selection import train_test_split
from tools import GS_Ridge
from DIY_data import data_from_input

if __name__ == "__main__":
    X,Y = data_from_input()
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25)
    results = GS_Ridge(X_train, X_test, y_train, y_test)
    
    print 'The best parameters %s'%results[0]
    print 'The scores of test set %s'%results[1]
    print 'Mark the finish line'
    
    
