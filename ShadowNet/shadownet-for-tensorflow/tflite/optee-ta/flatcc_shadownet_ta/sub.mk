global-incdirs-y += include
global-incdirs-y += ../../../../../../../usr/include/
#srcs-y += hello_world_ta.c
srcs-y += ta_cmds.c 
srcs-y += math.c 
srcs-y += shadownet.c 
srcs-y += muladd.c 

libdeps += library/libflatccrt.a
libdeps += library/libflatcc.a

# To remove a certain compiler flag, add a line like this
cflags-shadownet.c-y += -Wno-strict-prototypes
cflags-shadownet.c-y += -Wno-cast-align
cflags-shadownet.c-y += -Wno-asm-operand-widths
