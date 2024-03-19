cc_library(
    name = "tee_delegate",
    hdrs = [
            "tee_delegate.h",
            "include/tee_client_api.h"
            ],
    srcs = ["tee_delegate.cc"],
    visibility = ["//visibility:public"],
    deps = [
            "@optee_client//:libteec",
            ],
)

cc_library(
    name = "tee_emulated",
    hdrs = [
            "tee_delegate.h",
            "include/tee_client_api.h"
            ],
    srcs = ["tee_emulated_delegate.cc"],
    visibility = ["//visibility:public"],
    deps = [
            "@darknet//:libdarknet",
            ],
)

cc_library(
    name = "tee_shadow_net",
    hdrs = [
            "tee_shadow_net.h",
#            "alexnet_ca/include/shadownet_ca.h"
#            "minivgg_ca/include/shadownet_ca.h"
#            "shadownet_ca/shadownet_ca.h"
            ],
    srcs = ["tee_shadow_net.c"],
    visibility = ["//visibility:public"],
    deps = [
#            "@alexnet//:libalexnet",
            "@shadownet//:libshadownet",
#            "@minivgg//:libminivgg",
            ],
)
