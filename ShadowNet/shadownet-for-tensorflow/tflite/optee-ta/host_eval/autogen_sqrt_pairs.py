#!/usr/bin/env python
import random
import math 
x_arr = []
y_arr = []
s_arr = []
z_arr = []
for i in range(32):
    x = random.uniform(0, 10)
    y = random.uniform(0, 10)
    s = random.uniform(0, 10)
    x_arr.append(x)
    y_arr.append(y)
    s_arr.append(s)
    z_arr.append(x*s+y)
print x_arr
print y_arr
print s_arr
print z_arr
    
    

