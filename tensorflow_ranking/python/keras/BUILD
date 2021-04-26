# TensorFlow Ranking Keras API.

package(
    default_visibility = [
        "//tensorflow_ranking:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":estimator",
        ":feature",
        ":layers",
        ":losses",
        ":metrics",
        ":model",
        ":network",
        ":pipeline",
        ":saved_model",
        ":strategy_utils",
        ":utils",
        "//tensorflow_ranking/python/keras/canned",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":utils",
        # py/tensorflow dep,
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
        ":layers",
        # py/absl/logging dep,
        # py/tensorflow dep,
        # py/typing dep,
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
        # py/absl/testing:parameterized dep,
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
    srcs_version = "PY3",
    deps = [
        ":utils",
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
        ":utils",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
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
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
        # tensorflow_models/official/nlp/modeling dep,
        "//tensorflow_ranking/python:utils",
    ],
)

py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    python_version = "PY3",
    deps = [
        ":layers",
        # py/tensorflow dep,
    ],
)

py_library(
    name = "saved_model",
    srcs = ["saved_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
        # py/typing dep,
        "//tensorflow_ranking/python:data",
    ],
)

py_test(
    name = "saved_model_test",
    size = "small",
    srcs = ["saved_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":saved_model",
        # py/numpy dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_library(
    name = "strategy_utils",
    srcs = ["strategy_utils.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
    ],
)

py_test(
    name = "strategy_utils_test",
    srcs = ["strategy_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":strategy_utils",
        # py/tensorflow dep,
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    srcs_version = "PY3",
    deps = [
        ":losses",
        ":metrics",
        ":model",
        ":saved_model",
        ":strategy_utils",
        # py/dataclasses dep,
        # py/tensorflow dep,
        "//tensorflow_ranking/python:data",
    ],
)

py_test(
    name = "pipeline_test",
    size = "large",
    srcs = ["pipeline_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model",
        ":pipeline",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
