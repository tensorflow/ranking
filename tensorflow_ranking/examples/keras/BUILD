# Description:
# TensorFlow Ranking Keras examples.

package(
    default_visibility = [
        "//tensorflow_ranking:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "antique_kpl_din",
    srcs = ["antique_kpl_din.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/absl/flags dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
    ],
)

py_test(
    name = "antique_kpl_din_test",
    size = "large",
    srcs = ["antique_kpl_din_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":antique_kpl_din",
        # py/absl/testing:flagsaver dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_binary(
    name = "antique_ragged",
    srcs = ["antique_ragged.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/absl:app dep,
        # py/absl/flags dep,
        # py/absl/logging dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
    ],
)

py_test(
    name = "antique_ragged_test",
    srcs = ["antique_ragged_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":antique_ragged",
        # py/absl/testing:flagsaver dep,
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

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

py_binary(
    name = "keras_dnn_tfrecord",
    srcs = ["keras_dnn_tfrecord.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/absl:app dep,
        # py/absl/flags dep,
        # py/tensorflow dep,
        "//tensorflow_ranking",
    ],
)

py_test(
    name = "keras_dnn_tfrecord_test",
    srcs = ["keras_dnn_tfrecord_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_dnn_tfrecord",
        # py/absl/testing:flagsaver dep,
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
