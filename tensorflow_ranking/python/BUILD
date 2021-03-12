# Description:
# TensorFlow Ranking, the project to build ranking models on TensorFlow.
package(
    default_visibility = ["//tensorflow_ranking:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "python",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow_ranking:__pkg__"],
    deps = [
        ":data",
        ":estimator",
        ":feature",
        ":head",
        ":losses",
        ":metrics",
        ":model",
        "//tensorflow_ranking/python/keras",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "metrics_impl",
    srcs = ["metrics_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # py/six dep,
        # py/tensorflow dep,
    ],
)

py_test(
    name = "metrics_impl_test",
    size = "medium",
    srcs = ["metrics_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":metrics_impl",
        # py/tensorflow dep,
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_impl",
        ":utils",
        # py/tensorflow dep,
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
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
    ],
)

py_test(
    name = "utils_test",
    size = "medium",
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
    name = "losses_impl",
    srcs = ["losses_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # py/tensorflow dep,
    ],
)

py_test(
    name = "losses_impl_test",
    size = "medium",
    srcs = ["losses_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":losses_impl",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        ":utils",
        # py/tensorflow dep,
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
    name = "head",
    srcs = ["head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
        # py/tensorflow dep,
    ],
)

py_test(
    name = "head_test",
    size = "medium",
    srcs = ["head_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":head",
        ":losses",
        ":metrics",
        # py/tensorflow dep,
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # py/tensorflow dep,
    ],
)

py_test(
    name = "data_test",
    srcs = ["data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":data",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # py/tensorflow dep,
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
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":feature",
        ":utils",
        # py/tensorflow dep,
    ],
)

py_test(
    name = "model_test",
    size = "medium",
    srcs = ["model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":feature",
        ":head",
        ":losses",
        ":model",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":feature",
        ":head",
        ":losses",
        ":metrics",
        ":model",
        # py/tensorflow dep,
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
        ":data",
        ":estimator",
        ":feature",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
