# Description:
# TensorFlow Ranking Keras examples.

package(
    default_visibility = [
        "//tensorflow_ranking:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "keras_m2e_tfrecord_lib",
    srcs = ["keras_m2e_tfrecord.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/absl/flags dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
    ],
)

py_binary(
    name = "keras_m2e_tfrecord_py_binary",
    srcs = ["keras_m2e_tfrecord.py"],
    main = "keras_m2e_tfrecord.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":keras_m2e_tfrecord_lib",
    ],
)

py_test(
    name = "keras_m2e_tfrecord_test",
    size = "large",
    srcs = ["keras_m2e_tfrecord_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":keras_m2e_tfrecord_lib",
        # py/absl/flags dep,
        # py/absl/testing:flagsaver dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_library(
    name = "keras_m2e_gam_tfrecord_lib",
    srcs = ["keras_m2e_gam_tfrecord.py"],
    srcs_version = "PY2AND3",
    deps = [
        # py/absl/flags dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
    ],
)

py_binary(
    name = "keras_m2e_gam_tfrecord_py_binary",
    srcs = ["keras_m2e_gam_tfrecord.py"],
    main = "keras_m2e_gam_tfrecord.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":keras_m2e_gam_tfrecord_lib",
    ],
)

py_test(
    name = "keras_m2e_gam_tfrecord_test",
    size = "large",
    srcs = ["keras_m2e_gam_tfrecord_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":keras_m2e_gam_tfrecord_lib",
        # py/absl/flags dep,
        # py/absl/testing:flagsaver dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
