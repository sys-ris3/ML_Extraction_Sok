licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_binary(
    name = 'python/ops/_linear_transform_generic_ops.so',
    srcs = [
        "cc/kernels/linear_transform_generic_kernels.cc",
        "cc/ops/linear_transform_generic_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    copts = select({
        ":windows": ["/DEIGEN_STRONG_INLINE=inline", "-DTENSORFLOW_MONOLITHIC_BUILD", "/DPLATFORM_WINDOWS", "/DEIGEN_HAS_C99_MATH", "/DTENSORFLOW_USE_EIGEN_THREADPOOL", "/DEIGEN_AVOID_STL_ARRAY", "/Iexternal/gemmlowp", "/wd4018", "/wd4577", "/DNOGDI", "/UTF_COMPILE_LIBRARY"],
        "//conditions:default": ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    }),
)

py_library(
    name = "linear_transform_generic_ops_py",
    srcs = ([
        "python/ops/linear_transform_generic_ops.py",
    ]),
    data = [
        ":python/ops/_linear_transform_generic_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "linear_transform_generic_ops_py_test",
    srcs = [
        "python/ops/linear_transform_generic_ops_test.py"
    ],
    main = "python/ops/linear_transform_generic_ops_test.py",
    deps = [
        ":linear_transform_generic_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "linear_transform_generic_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":linear_transform_generic_ops_py"
    ],
    srcs_version = "PY2AND3",
)
