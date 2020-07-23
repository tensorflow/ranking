# TensorFlow Ranking Keras canned models.

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
        ":gam",
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
    python_version = "PY3",
    srcs_version = "PY3",
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

py_library(
    name = "gam",
    srcs = ["gam.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_ranking/python:utils",
        "//tensorflow_ranking/python/keras:feature",
        "//tensorflow_ranking/python/keras:model",
    ],
)

py_test(
    name = "gam_test",
    size = "small",
    srcs = ["gam_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":gam",
        # py/tensorflow dep,
        "//tensorflow_ranking/python/keras:losses",
        "//tensorflow_ranking/python/keras:metrics",
        "//tensorflow_ranking/python/keras:network",
    ],
)
