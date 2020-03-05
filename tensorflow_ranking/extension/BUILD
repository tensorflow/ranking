# Description:
# Folder for TensorFlow ranking extensions.
# Placeholder for internal Python version compatibility macro.

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
    shard_count = 2,
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
