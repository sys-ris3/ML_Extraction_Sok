#!/usr/bin/env python
from inception_v3_split import InceptionV3Split 

model = InceptionV3Split(weights=None)
model.summary()
model.save('inception_v3_split.h5')
