cc_library(
    name = "libalexnet",
    srcs = [
		"src/shadownet_ca.c",
		"src/ca_cmds.c",
    ],
    hdrs = [
        "include/shadownet_ca.h",
        "include/tee_client_api.h",
		"src/ca_cmds.h",
    ],
    copts = [
        "-DDEBUG",
        "-fPIC",
        "-c",
        "-Wall",
        "-Wno-unused-result",
        "-Wno-unknown-pragmas",
        "-Wfatal-errors",
    ],
    includes = [
        "include",
        "src",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [
            "-lm",
        ],
        "//conditions:default": [
            "-lm",
        ],
    }),
    deps = [
            "@optee_client//:libteec",
            ],
    visibility = ["//visibility:public"],
)
