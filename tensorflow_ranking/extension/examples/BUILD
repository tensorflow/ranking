# Description:
# Example usage of tensorflow ranking extensions.

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "pipeline_example_lib",
    srcs = ["pipeline_example.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/absl/flags dep,
        # py/tensorflow dep,
        # py/tensorflow:tensorflow_estimator dep,
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
