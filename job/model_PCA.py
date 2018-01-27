#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_PCA.py
'''

from tools import GS_PCA
from DIY_data import data_from_input

if __name__ == "__main__":
    X,Y = data_from_input()
    results = GS_PCA(X)
    
    print 'The number of features %s'%results[0]
    print 'Explained variances %s'%results[1]
    print 'To DIY %s'%results[2]
    print 'Mark the finish line'
    
    