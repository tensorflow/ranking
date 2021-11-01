# Install TensorFlow Ranking

There are several ways to set up your environment to use the TensorFlow Ranking
library.

+   The easiest way to learn and use TensorFlow Ranking is run any of the 
    tutorials Google Colab. Select the link at the top of the
    [Quickstart tutorial](/ranking/tutorials/quickstart)).
+   To use the Ranking library on a local machine, install the
    `tensorflow_ranking` pip package.
+   If you have a unique machine configuration, you can build the package
    from source, using the [Build from source](#build-source) instructions.

## Install TensorFlow Ranking using pip

Install using pip.

```posix-terminal
pip install --upgrade tensorflow_ranking
```

## Build from source {:#build-source}

You can also install from source, which requires the
[Bazel](https://bazel.build/) build system.

1.  Install Bazel, Git and Pip.
    ```posix-terminal
    sudo apt-get update

    sudo apt-get install bazel git python3-pip python3-venv
    ```
1.  Clone the TensorFlow Ranking repository.
    <pre class="devsite-terminal prettyprint lang-bsh">
    git clone https://github.com/tensorflow/ranking.git</pre>
1.  Build TensorFlow Ranking wheel file and store them in a
    `/tmp/ranking_pip` folder.
    ```posix-terminal
    cd ranking  # folder cloned in Step 2.

    bazel build //tensorflow_ranking/tools/pip_package:build_pip_package

    bazel-bin/tensorflow_ranking/tools/pip_package/build_pip_package \
        /tmp/ranking_pip
    ```
1.  Activate a `venv` environment.
    ```posix-terminal
    python3 -m venv --system-site-packages venv

    source venv/bin/activate
    ```
1.  Install the wheel package in your `venv` environment.
    ```devsite-terminal {:.tfo-terminal-venv}
    pip install /tmp/ranking_pip/tensorflow_ranking*.whl
    ```
1.  Optionally, run all TensorFlow Ranking tests.
    ```devsite-terminal {:.tfo-terminal-venv}
    bazel test //tensorflow_ranking/...
    ```

For more information about installing Python, pip, TensorFlow, and working with
Python virtual environments, see
[Install TensorFlow with pip](/install/pip#2.-create-a-virtual-environment-recommended).
