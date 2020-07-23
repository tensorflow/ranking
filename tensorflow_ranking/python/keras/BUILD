# TensorFlow Ranking Keras API.

package(
    default_visibility = [
        "//tensorflow_ranking:__subpackages__",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":estimator",
        ":feature",
        ":losses",
        ":metrics",
        ":model",
        "//tensorflow_ranking/python/keras/canned",
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
        # py/tensorflow dep,
        "//tensorflow_ranking/python:utils",
    ],
)

py_test(
    name = "estimator_test",
    size = "large",
    srcs = ["estimator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":estimator",
        ":losses",
        ":metrics",
        ":model",
        ":network",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        "//tensorflow_ranking/python:data",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_library(
    name = "network",
    srcs = ["network.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":feature",
        # py/tensorflow dep,
        "//tensorflow_ranking/python:utils",
    ],
)

py_test(
    name = "network_test",
    size = "small",
    srcs = ["network_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":losses",
        ":metrics",
        ":network",
        # py/tensorflow dep,
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":feature",
        # py/tensorflow dep,
        "//tensorflow_ranking/python:utils",
    ],
)

py_test(
    name = "model_test",
    size = "small",
    srcs = ["model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":feature",
        ":losses",
        ":metrics",
        ":model",
        ":network",
        # py/tensorflow dep,
        "//tensorflow_ranking/python:data",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_ranking/python:utils",
    ],
)

py_test(
    name = "feature_test",
    size = "small",
    srcs = ["feature_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":feature",
        # py/tensorflow dep,
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_ranking/python:losses_impl",
    ],
)

py_test(
    name = "losses_test",
    size = "medium",
    srcs = ["losses_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":losses",
        # py/tensorflow dep,
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_ranking/python:metrics_impl",
    ],
)

py_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":metrics",
        # py/tensorflow dep,
        "//tensorflow_ranking/python:metrics_impl",
    ],
)
