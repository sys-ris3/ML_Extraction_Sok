global-incdirs-y += include
global-incdirs-y += darknet_ta/include

srcs-y += darknet_ta/src/activation_layer.c
srcs-y += darknet_ta/src/activations.c
srcs-y += darknet_ta/src/add_mask_layer.c
srcs-y += darknet_ta/src/avgpool_layer.c
srcs-y += darknet_ta/src/batchnorm_layer.c
srcs-y += darknet_ta/src/blas.c
srcs-y += darknet_ta/src/dropout_layer.c
srcs-y += darknet_ta/src/layer.c
srcs-y += darknet_ta/src/linear_transform_layer.c
srcs-y += darknet_ta/src/list.c
srcs-y += darknet_ta/src/network.c
srcs-y += darknet_ta/src/option_list.c
srcs-y += darknet_ta/src/parser.c
srcs-y += darknet_ta/src/shuffle_channel_layer.c
srcs-y += darknet_ta/src/softmax_layer.c
srcs-y += darknet_ta/src/ta_cmds.c
srcs-y += darknet_ta/src/utils.c
srcs-y += darknet_ta/src/math_TA.c

# To remove a certain compiler flag, add a line like this
cflags-template_ta.c-y += -Wno-strict-prototypes
cflags-darknet_ta/src/parser.c-y += -Wno-aggregate-return
cflags-darknet_ta/src/parser.c-y += -Wno-missing-prototypes
cflags-darknet_ta/src/parser.c-y += -Wno-missing-declarations
cflags-darknet_ta/src/parser.c-y += -Wno-unused-parameter
cflags-darknet_ta/src/parser.c-y += -Wno-shadow
cflags-darknet_ta/src/softmax_layer.c-y += -Wno-aggregate-return
cflags-darknet_ta/src/network.c-y += -Wno-aggregate-return
cflags-darknet_ta/src/add_mask_layer.c-y += -Wno-aggregate-return
