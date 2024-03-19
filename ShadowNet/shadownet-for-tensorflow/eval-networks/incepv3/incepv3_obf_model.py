#!/usr/bin/env python
from inception_v3_obf import InceptionV3Obf 

scalar_stack = []
model = InceptionV3Obf(scalar_stack, weights=None)
model.summary()
model.save('inception_v3_obf.h5')
