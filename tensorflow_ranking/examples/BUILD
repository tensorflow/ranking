# Description:
# TensorFlow Ranking examples.
package(
    default_visibility = [
        "//tensorflow_ranking:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# Placeholder for internal Python version compatibility macro.

py_library(
    name = "tf_ranking_libsvm_lib",
    srcs = ["tf_ranking_libsvm.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/absl/flags dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
    ],
)

py_binary(
    name = "tf_ranking_libsvm_py_binary",
    srcs = ["tf_ranking_libsvm.py"],
    main = "tf_ranking_libsvm.py",
    srcs_version = "PY2AND3",
    deps = [
        ":tf_ranking_libsvm_lib",
    ],
)

py_test(
    name = "tf_ranking_libsvm_test",
    size = "large",
    srcs = ["tf_ranking_libsvm_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":tf_ranking_libsvm_lib",
        # py/absl/flags dep,
        # py/absl/testing:flagsaver dep,
        # py/tensorflow dep,
    ],
)
