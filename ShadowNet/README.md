# ShadowNet: A Secure and Efficient Scheme for On-device Model Inference

## Overview
ShadowNet is a security schme designed for on-device machine learning. It
aims at protecting the model privacy without sacrifing performance. 
It solved two challenges faced by secure model inference on mobile device:

- How to use GPU for model inference without leaking model weights?
- How to use TEE(e.g. Arm TrustZone) without exhausting its limited resource(both CPU and Memory)? 

In detail, ShadowNet transforms machine learning models(e.g., tflite models) 
and splits them into two parts: an obfuscated part(O-part) and a secret part(S-part).
The O-part contains transformed weights for linear layers and can be stored in 
plaintext. This part runs in the Normal World(e.g. Android) and can be offloaded to
GPU for acceleration without leaking the original weights. 
The S-part is smaller. It contains weights for non-linear layers and 
only runs inside TEE(e.g. ARM TrustZone) with ShadowNet model inference engine for TEE.


## ML Framework Support
ShadowNet supports two modes: *TEE-emulated* mode and *TEE-enabled* mode. 
*TEE-emulated* mode runs both O-part and S-part in the Normal World.
It's used for quick testing. We have extended both TensorFlow (Lite) and Darknet framework
to support ShadowNet in *TEE-emulated* mode. *TEE-enabled* mode is only supported on TensorFlow(Lite).

## Transform Your Model into ShadowNet-enabled Model
ShadowNet offers a set of tools to help developer transform TensorFlow models into ShadowNet
enabled models. Existing Android apps using TensorFlow Lite models can switch to
to ShadowNet-enabled models seemlessly. So far, we have included transformed examples 
for MobileNet, AlexNet and MiniVGG.

Model transforming pipeline contains the following steps:
- Preparation: Original model description script using python Keras interface(model).
- Obfuscation: Following a set of rules to transform the original model description into an obfuscated model(model.obf)
 It corresponds to the *TEE-emulated* mode.
- Split: Following a set of rules to transform original model description into ShadowNet enabled model, you will get
the O-part of the model(model.split). It corresponds to the *TEE-enabled* mode.
- ShadowWeights: Using our tool to generate weights for TEE part, the S-part of the model(model.tee.weights).
- OP-TEE CA/TA: Writing the CA(client application) and TA(trusted application) for the model with our libraries. CA
 connects the TensorFlow Lite to the TA inside TrustZone.
