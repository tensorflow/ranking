FROM tensorflow/tensorflow:latest-gpu

RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add -

RUN apt-get update && apt-get install -y \
    openjdk-8-jdk \
    python-setuptools \ 
    bazel

COPY . ranking

WORKDIR ranking

RUN bazel build //tensorflow_ranking/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow_ranking/tools/pip_package/build_pip_package /tmp/ranking_pip && \
    pip install /tmp/ranking_pip/tensorflow_ranking*.whl
