# Description:
# TensorFlow Ranking research code for published papers.

# Placeholder: load py_library
# Placeholder: load py_test
# Placeholder: load py_binary

package(
    default_visibility = [
        "//tensorflow_ranking:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "dasalc_lib",
    srcs = ["dasalc.py"],
    srcs_version = "PY3",
    deps = [
        # py/absl/flags dep,
        # py/tensorflow dep,
        # py/tensorflow:tensorflow_estimator dep,
        "//tensorflow_ranking",
    ],
)

py_binary(
    name = "dasalc_py_binary",
    srcs = ["dasalc.py"],
    main = "dasalc.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dasalc_lib",
    ],
)

py_test(
    name = "dasalc_test",
    size = "large",
    srcs = ["dasalc_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":dasalc_lib",
        # py/absl/flags dep,
        # py/absl/testing:flagsaver dep,
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
