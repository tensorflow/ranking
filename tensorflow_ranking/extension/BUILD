# Description:
# Folder for TensorFlow ranking extensions.

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

py_library(
    name = "extension",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow_ranking:__pkg__"],
    deps = [
        ":pipeline",
        "//tensorflow_ranking/extension/premade",
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
        # py/tensorflow:tensorflow_compat_v1_estimator dep,
        # py/tensorflow:tensorflow_estimator dep,
        "//tensorflow_ranking/python:data",
    ],
)

py_test(
    name = "pipeline_test",
    size = "large",
    srcs = ["pipeline_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":pipeline",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        # py/tensorflow:tensorflow_estimator dep,
        "//tensorflow_ranking",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**"]),
)

py_library(
    name = "task",
    srcs = ["task.py"],
    srcs_version = "PY3",
    deps = [
        # py/absl/logging dep,
        # py/orbit dep,
        # py/tensorflow dep,
        # tensorflow_models/official/core:base_task dep,
        # tensorflow_models/official/core:config_definitions dep,
        # tensorflow_models/official/core:input_reader dep,
        # tensorflow_models/official/core:task_factory dep,
        # tensorflow_models/official/modeling:tf_utils dep,
        # tensorflow_models/official/nlp/data:data_loader dep,
        # tensorflow_models/official/nlp/modeling dep,
        "//tensorflow_ranking/python:data",
        "//tensorflow_ranking/python/keras:losses",
        "//tensorflow_ranking/python/keras:metrics",
        "//tensorflow_ranking/python/keras:model",
    ],
)

py_test(
    name = "task_test",
    size = "large",
    srcs = ["task_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":task",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        "//tensorflow_ranking/python/keras:model",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
