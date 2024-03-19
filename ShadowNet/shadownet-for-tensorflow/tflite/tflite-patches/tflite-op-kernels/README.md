## Patches on Tensorflow

## Repo info
```
remote: https://github.com/tensorflow/tensorflow
tag: r2.2
commit: b34c246
```

## Patch Details
Add custom ops for tensorflow lite to allow running obfuscated
models with partial model offloaded to TEE for secure inference.

## Build Tips
For python interface test on x86-64, use build script `ModelSafe-Code/scripts/build_tensorflow.sh`.

For android aar build, use `ModelSafe-Code/scripts/build_tflite_aar.sh`.


## Testing Tips
When run the modified tflite lite demo app with AAR built here, please
change mode of `/dev/tee0` on Android device to `0666` to allow app's
access, and turn off selinux checks with`su;setenforce 0`.
