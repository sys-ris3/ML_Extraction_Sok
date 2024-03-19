global-incdirs-y += include
global-incdirs-y += shadownet_ta/include

srcs-y += shadownet_ta/src/shadownet.c
srcs-y += shadownet_ta/src/forward_layers.c
srcs-y += shadownet_ta/src/math.c
srcs-y += shadownet_ta/src/ta_cmds.c
srcs-y += shadownet_ta/src/sqrtf.c
srcs-y += shadownet_ta/src/neon_sqrt.c
srcs-y += shadownet_ta/src/muladd.c

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
#cflags-shadownet_ta/src/parser.c-y += -Wno-aggregate-return
#cflags-shadownet_ta/src/parser.c-y += -Wno-missing-prototypes
#cflags-shadownet_ta/src/parser.c-y += -Wno-missing-declarations
#cflags-shadownet_ta/src/parser.c-y += -Wno-unused-parameter
#cflags-shadownet_ta/src/parser.c-y += -Wno-shadow
#cflags-shadownet_ta/src/softmax_layer.c-y += -Wno-aggregate-return
#cflags-shadownet_ta/src/network.c-y += -Wno-aggregate-return
#cflags-shadownet_ta/src/add_mask_layer.c-y += -Wno-aggregate-return
