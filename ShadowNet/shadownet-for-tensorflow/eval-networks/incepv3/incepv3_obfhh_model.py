#!/usr/bin/env python
from inception_v3_obfhh import InceptionV3ObfHH 

scalar_stack = []
model = InceptionV3ObfHH(scalar_stack, weights=None)
model.summary()
model.save('inception_v3_obfhh.h5')
