#!/usr/bin/env python
import pandas as pd
#a = {3:'a',5:'b'}
#pd.to_pickle(a,'a.dic.pkl')

cache_a = pd.read_pickle('a.dic.pkl')
print(cache_a[3])
