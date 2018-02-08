#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_reg_LSTM.py
'''

from DL_Package import GS_reg_LSTM
from DIY_data import data_from_input

if __name__ == "__main__":
    X,Y = data_from_input()
    
    X_va=X.values
    Y_va=Y.values
    nums = X_va.shape[0]
    start_test = int(0.75*nums)
                 
    X_train = X_va[0:start_test,:]
    X_test = X_va[start_test:,:]
    y_train = Y_va[0:start_test]
    y_test = Y_va[start_test:]
    
    results = GS_reg_LSTM(X_train, X_test, y_train, y_test)
    
    print 'The best parameters %s'%results[0]
    print 'The scores of test set %s'%results[1]
    print 'Mark the finish line'
    
    
