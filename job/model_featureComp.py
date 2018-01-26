#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_featureComp.py
'''

from tools import features_com
from DIY_data import data_from_input

if __name__ == "__main__":
    X,Y = data_from_input()
    results = features_com(X,Y)
    
    print 'Results: %s'%results
    
