#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_reg_featureComp.py
'''

from tools import features_com_reg
from DIY_data import data_from_input

if __name__ == "__main__":
    X,Y = data_from_input()
    results = features_com_reg(X,Y)
    
    print 'Results: %s'%results
    print 'Mark the finish line'
    
