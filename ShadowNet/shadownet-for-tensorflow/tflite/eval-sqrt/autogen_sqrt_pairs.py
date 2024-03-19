#!/usr/bin/env python
import random
import math 
x_arr = []
y_arr = []
for i in range(20):
    x = random.uniform(0, 10)
    x_arr.append(x)
    y_arr.append(math.sqrt(x))
print x_arr
print y_arr
    
    

