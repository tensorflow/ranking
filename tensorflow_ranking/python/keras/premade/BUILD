"""TFR-BERT."""

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "tfrbert_task",
    srcs = ["tfrbert_task.py"],
    deps = [
        # py/absl/logging dep,
        # py/dataclasses dep,
        # py/numpy dep,
        # py/orbit dep,
        # py/scipy dep,
        # py/sklearn dep,
        # py/tensorflow dep,
        # tensorflow_models/official/core:config_definitions dep,
        # tensorflow_models/official/core:task_factory dep,
        # tensorflow_models/official/modeling/hyperparams:base_config dep,
        # tensorflow_models/official/nlp/configs:encoders dep,
        # tensorflow_models/official/nlp/modeling dep,
        "//tensorflow_ranking/python/keras:metrics",
        "//tensorflow_ranking/python/keras:model",
        "//tensorflow_ranking/python/keras:task",
    ],
)

py_test(
    name = "tfrbert_task_test",
    srcs = ["tfrbert_task_test.py"],
    python_version = "PY3",
    deps = [
        ":tfrbert_task",
        # py/absl/flags dep,
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        # tensorflow_models/official/nlp/configs:bert dep,
        # tensorflow_models/official/nlp/configs:encoders dep,
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
