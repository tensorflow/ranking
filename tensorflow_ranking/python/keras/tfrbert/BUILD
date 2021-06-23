"""TFR-BERT."""

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "tfrbert_task",
    srcs = ["tfrbert_task.py"],
    deps = [
        ":tfrbert_dataloader",
        ":tfrbert_model",
        # py/absl/logging dep,
        # py/dataclasses dep,
        # py/numpy dep,
        # py/orbit dep,
        # py/scipy dep,
        # py/sklearn dep,
        # py/tensorflow dep,
        # tensorflow_models/official/core:base_task dep,
        # tensorflow_models/official/core:config_definitions dep,
        # tensorflow_models/official/core:task_factory dep,
        # tensorflow_models/official/modeling:tf_utils dep,
        # tensorflow_models/official/modeling/hyperparams:base_config dep,
        # tensorflow_models/official/nlp/configs:encoders dep,
        # tensorflow_models/official/nlp/modeling dep,
        "//tensorflow_ranking/python/keras:losses",
        "//tensorflow_ranking/python/keras:metrics",
        "//tensorflow_ranking/python/keras:model",
    ],
)

py_test(
    name = "tfrbert_task_test",
    srcs = ["tfrbert_task_test.py"],
    python_version = "PY3",
    deps = [
        ":tfrbert_task",
        # py/absl/flags dep,
        # py/tensorflow dep,
        # tensorflow_models/official/nlp/configs:bert dep,
        # tensorflow_models/official/nlp/configs:encoders dep,
    ],
)

py_library(
    name = "experiments",
    srcs = ["experiments.py"],
    deps = [
        ":tfrbert_dataloader",
        ":tfrbert_task",
        # tensorflow_models/official/core:config_definitions dep,
        # tensorflow_models/official/core:exp_factory dep,
        # tensorflow_models/official/modeling/optimization dep,
    ],
)

py_library(
    name = "tfrbert_dataloader",
    srcs = ["tfrbert_dataloader.py"],
    deps = [
        # py/dataclasses dep,
        # py/tensorflow dep,
        # tensorflow_models/official/core:config_definitions dep,
        # tensorflow_models/official/core:input_reader dep,
        # tensorflow_models/official/nlp/data:data_loader dep,
        "//tensorflow_ranking/python:data",
    ],
)

py_test(
    name = "tfrbert_dataloader_test",
    timeout = "long",
    srcs = ["tfrbert_dataloader_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tfrbert_dataloader",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        "//tensorflow_ranking/python:data",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)

py_library(
    name = "tfrbert_model",
    srcs = ["tfrbert_model.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # py/absl/logging dep,
        # py/tensorflow dep,
        # py/tensorflow_ranking dep,
        "//tensorflow_ranking/python/keras:model",
    ],
)

py_test(
    name = "tfrbert_model_test",
    timeout = "long",
    srcs = ["tfrbert_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tfrbert_model",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        # tensorflow_models/official/nlp/configs:bert dep,
        # tensorflow_models/official/nlp/configs:encoders dep,
        "//tensorflow_ranking/python:data",
        "//tensorflow_ranking/python/keras:model",
        # tensorflow_serving/apis:input_proto_py_pb2 dep,
    ],
)
