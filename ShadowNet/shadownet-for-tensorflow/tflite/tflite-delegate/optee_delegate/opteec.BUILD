cc_library(
    name = "libteec",
    srcs = [
        "libteec/src/tee_client_api.c",
        "libteec/src/teec_trace.c",
    ],
    hdrs = [
        "libteec/include/linux/tee.h",
    ],
    copts = [
        "-D_GNU_SOURCE",
        "-DDEBUGLEVEL_1",
        "-D_FILE_OFFSET_BITS=64",
        '-DBINARY_PREFIX=\\"TEEC\\"',
        "-fPIC",
        "-c",
        "-Wall",
        "-Wbad-function-cast",
        "-Wcast-align",
    ],
    includes = [
        "libteec/include",
        "public/",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [],
        "//conditions:default": [
            "-lpthread",
        ],
    }),
    visibility = ["//visibility:public"],
)
