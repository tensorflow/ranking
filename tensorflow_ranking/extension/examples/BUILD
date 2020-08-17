# Description:
# Example usage of tensorflow ranking extensions.

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "pipeline_example_lib",
    srcs = ["pipeline_example.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/absl/flags dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
        "//tensorflow_ranking/extension:pipeline",
    ],
)

py_binary(
    name = "pipeline_example_py_binary",
    srcs = ["pipeline_example.py"],
    main = "pipeline_example.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pipeline_example_lib",
    ],
)

py_test(
    name = "pipeline_example_test",
    size = "large",
    srcs = ["pipeline_example_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":pipeline_example_lib",
        # py/absl/flags dep,
        # py/absl/testing:flagsaver dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_library(
    name = "tfrbert_example_lib",
    srcs = ["tfrbert_example.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/absl/flags dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
        "//tensorflow_ranking/extension:tfrbert",
    ],
)

py_binary(
    name = "tfrbert_example_py_binary",
    srcs = ["tfrbert_example.py"],
    main = "tfrbert_example.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":tfrbert_example_lib",
    ],
)

py_test(
    name = "tfrbert_example_test",
    size = "large",
    srcs = ["tfrbert_example_test.py"],
    data = ["//tensorflow_ranking/extension:testdata"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":tfrbert_example_lib",
        # py/absl/flags dep,
        # py/absl/testing:flagsaver dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
