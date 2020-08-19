# Description:
# Folder for TensorFlow ranking extensions.

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "extension",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow_ranking:__pkg__"],
    deps = [
        ":pipeline",
        ":tfrbert",
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
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
        # py/tensorflow dep,
        "//tensorflow_ranking",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**"]),
)

py_library(
    name = "tfrbert",
    srcs = ["tfrbert.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/tensorflow dep,
        # tensorflow_models/official/modeling/activations dep,
        # tensorflow_models/official/nlp:optimization dep,
        # tensorflow_models/official/nlp/bert:configs dep,
        # tensorflow_models/official/nlp/bert:tokenization dep,
        # tensorflow_models/official/nlp/modeling/networks dep,
        "//tensorflow_ranking/python:estimator",
        "//tensorflow_ranking/python:head",
        "//tensorflow_ranking/python:losses",
        "//tensorflow_ranking/python:model",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_test(
    name = "tfrbert_test",
    size = "large",
    srcs = ["tfrbert_test.py"],
    data = [":testdata"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":tfrbert",
        # py/absl/flags dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
