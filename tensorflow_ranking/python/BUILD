# Description:
# TensorFlow Ranking, the project to build ranking models on TensorFlow.
package(
    default_visibility = [
        "//tensorflow_ranking:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# Placeholder for internal Python version compatibility macro.

py_library(
    name = "python",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow_ranking:__pkg__"],
    deps = [
        ":data",
        ":feature",
        ":head",
        ":losses",
        ":metrics",
        ":model",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # py/tensorflow dep,
    ],
)

py_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
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
    name = "losses",
    srcs = ["losses.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # py/tensorflow dep,
    ],
)

py_test(
    name = "losses_test",
    size = "medium",
    srcs = ["losses_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
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
        # py/tensorflow dep,
    ],
)

py_test(
    name = "data_test",
    srcs = ["data_test.py"],
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":data",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
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
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":feature",
        ":head",
        ":losses",
        ":model",
        # py/tensorflow dep,
    ],
)
