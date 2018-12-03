# Description:
#  Tools for building the TensorFlow Ranking pip package.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

COMMON_PIP_DEPS = [
    "//tensorflow_ranking",
]

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = COMMON_PIP_DEPS,
)
