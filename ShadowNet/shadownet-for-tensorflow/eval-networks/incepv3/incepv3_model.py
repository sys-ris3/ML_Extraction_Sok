#!/usr/bin/env python
from inception_v3 import InceptionV3 

model = InceptionV3()
model.summary()
model.save('inception_v3.h5')
