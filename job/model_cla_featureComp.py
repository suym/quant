#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_cla_featureComp.py
'''

from tools import features_com_cla
from DIY_data import data_from_input

if __name__ == "__main__":
    X,Y = data_from_input()
    results = features_com_cla(X,Y)
    
    print 'Results: %s'%results
    print 'Mark the finish line'
    
