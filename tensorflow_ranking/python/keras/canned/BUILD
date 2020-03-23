# TensorFlow Ranking Keras canned models.

# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "canned",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":dnn",
    ],
)

py_library(
    name = "dnn",
    srcs = ["dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_ranking/python:utils",
        "//tensorflow_ranking/python/keras:feature",
        "//tensorflow_ranking/python/keras:network",
    ],
)

py_test(
    name = "dnn_test",
    size = "small",
    srcs = ["dnn_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":dnn",
        # py/tensorflow dep,
        "//tensorflow_ranking/python/keras:losses",
        "//tensorflow_ranking/python/keras:metrics",
        "//tensorflow_ranking/python/keras:model",
    ],
)
