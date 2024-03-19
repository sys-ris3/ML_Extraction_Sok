###################### optee-hello-world ######################
LOCAL_PATH := $(call my-dir)

OPTEE_CLIENT_EXPORT = $(LOCAL_PATH)/../../optee_client/out/export

include $(CLEAR_VARS)
LOCAL_CFLAGS += -DANDROID_BUILD
#LOCAL_CFLAGS += -Wall
LOCAL_CFLAGS += -Wno-error
LOCAL_CFLAGS += -O1 -fvectorize

LOCAL_SRC_FILES += host/fake_ca.c
LOCAL_SRC_FILES += host/ca_cmds.c
LOCAL_SRC_FILES += host/shadownet_ca.c

#LOCAL_C_INCLUDES := $(LOCAL_PATH)/ta/include 
#LOCAL_C_INCLUDES := $(LOCAL_PATH)/ta_minivgg/include 
LOCAL_C_INCLUDES := $(LOCAL_PATH)/ta_alexnet/include  \
		$(OPTEE_CLIENT_EXPORT)/include \
		$(LOCAL_PATH)/host/include \

LOCAL_SHARED_LIBRARIES := libteec
LOCAL_MODULE := optee_example_hello_world
LOCAL_VENDOR_MODULE := true
LOCAL_MODULE_TAGS := optional
include $(BUILD_EXECUTABLE)

#include $(LOCAL_PATH)/ta/Android.mk
include $(LOCAL_PATH)/ta_alexnet/Android.mk
#include $(LOCAL_PATH)/ta_minivgg/Android.mk
