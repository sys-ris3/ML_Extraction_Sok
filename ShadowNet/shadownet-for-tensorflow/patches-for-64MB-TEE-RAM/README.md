# Set up AOSP + OPTEE on HiKey960 Board

## clone repo
optee_android_manifest_v390dirty: this repo is used to build android and optee that can run hikey960. It's current version is v3.9.0dirty which can boot Android 9. The optee os is 3.9.0, which support dynamic shared memory range yet. Based on this [post](https://github.com/OP-TEE/optee_os/issues/4087), this branch should work and Android 9.0 should boot.

```
git clone https://github.com/linaro-swg/optee_android_manifest.git -b master-darty optee_android_manifest_v390dirty
```

## Apply patches to support 64MB RAM for TEE OS

Apply the following patches in the corresponding directory under `optee_android_manifest_v390dirty/optee/`

```
optee_android_manifest_v390dirty/optee/
atf-fastboot  OpenPlatformPkg  optee_os   trusted-firmware-a

patches are below:

./atf-fastboot:
0001-atf-fastboot-add-build-option-for-64-MB-memory.patch

./OpenPlatformPkg:
0001-OpenPlatformPkg-add-build-option-for-Hikey960-TZRAM-.patch

./optee_os:
0001-optee-os-add-build-option-for-64-MB-TEE-memory.patch

./trusted-firmware-a:
0001-trusted-firmware-a-add-build-option-for-64-MB-TEE-me.patch
```

## Tips for compilation

- repo should be placed in a non-NFS partition to avoid aquire file lock error
- python should be updated to python3, use virtual env (source ~/work/myenv36/bin/activate)
