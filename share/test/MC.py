from __future__ import division
import numpy as np
from time import time
import sys
a = []
b = []
k=0
time_start = time()
for num in xrange(10000000):
    i = np.random.random_integers(1,6)
    a.append(i)
while len(a)>=(k+3+1):
    if a[k]==a[k+1]==a[k+2]==a[k+3]:
        b.append(k+3+1)
        del a[0:k+3+1]
        k=-1
    k=k+1
dur = (time()-time_start)
sys.stdout.write(' \nDone in %s. \n' % dur)
print 'EX:',float(sum(b)/len(b)) 
print 'total number of times:',len(b) 


