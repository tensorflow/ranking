# TensorFlow Ranking

TensorFlow Ranking is a library for Learning-to-Rank (LTR) techniques on the
TensorFlow platform. It contains the following components:

*   Commonly used loss functions including pointwise, pairwise, and listwise
    losses.
*   Commonly used ranking metrics like Mean Reciprocal Rank (MRR) and Normalized
    Discounted Cumulative Gain (NDCG).
*   [Multi-item (also known as groupwise) scoring functions](https://arxiv.org/abs/1811.04415).
*   [LambdaLoss](https://ai.google/research/pubs/pub47258) implementation for
    direct ranking metric optimization.
*   [Unbiased Learning-to-Rank](http://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf)
    from biased feedback data.

We envision that this library will provide a convenient open platform for
hosting and advancing state-of-the-art ranking models based on deep learning
techniques, and thus facilitate both academic research as well as industrial
applications.

A quick demo for a ranker on dummy dataset (no setup required): [![Run in Google Colab](https://www.tensorflow.org/images/colab_logo_32px.png "")](https://colab.research.google.com/github/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_libsvm.ipynb)

For more details on this code and data, look at
 the section on [Example Code](https://github.com/tensorflow/ranking#example-code).

## Linux Installation

### Stable Builds

To install the latest version from [PyPI](https://pypi.org/project/tensorflow-ranking/), run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tensorflow_ranking
```

To force a Python 3-specific install, replace `pip` with `pip3` in the above
commands. For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the [TensorFlow
installation guide](https://www.tensorflow.org/install).

Note: Since TensorFlow is *not* included as a dependency of the TensorFlow
Ranking package (in `setup.py`), you must explicitly install the TensorFlow
package (`tensorflow` or `tensorflow-gpu`). This allows us to maintain one
package instead of separate packages for CPU and GPU-enabled TensorFlow.

### Installing from Source

1.  To build TensorFlow Ranking locally, you will need to install:

    *   [Bazel](https://docs.bazel.build/versions/master/install.html), an open
        source build tool.

        ```shell
        $ sudo apt-get update && sudo apt-get install bazel
        ```

    *   [Pip](https://pypi.org/project/pip/), a Python package manager.

        ```shell
        $ sudo apt-get install python-pip
        ```

    *   [VirtualEnv](https://virtualenv.pypa.io/en/stable/installation/), a tool
        to create isolated Python environments.

        ```shell
        $ pip install --user virtualenv
        ```

2.  Clone the TensorFlow Ranking repository.

    ```shell
    $ git clone https://github.com/tensorflow/ranking.git
    ```

3.  Build TensorFlow Ranking wheel file and store them in `/tmp/ranking_pip`
    folder.

    ```shell
    $ cd ranking  # The folder which was cloned in Step 2.
    $ bazel build //tensorflow_ranking/tools/pip_package:build_pip_package
    $ bazel-bin/tensorflow_ranking/tools/pip_package/build_pip_package /tmp/ranking_pip
    ```

4.  Install the wheel package using pip. Test in virtualenv, to avoid clash with
    any system dependencies.

    ```shell
    $ ~/.local/bin/virtualenv -p python3 /tmp/tfr
    $ source /tmp/tfr/bin/activate
    (tfr) $ pip install tensorflow  #  or tensorflow-gpu, if GPU support is needed.
    (tfr) $ pip install /tmp/ranking_pip/tensorflow_ranking*.whl
    ```

5.  Run all TensorFlow Ranking tests.

    ```shell
    (tfr) $ bazel test //tensorflow_ranking/...
    ```

6.  Invoke TensorFlow Ranking package in python (within virtualenv).

    ```shell
    (tfr) $ python -c "import tensorflow_ranking"
    ```

## Example Code

The repository has a running script over a dummy data set in
[the LIBSVM format](https://sourceforge.net/p/lemur/wiki/RankLib%20File%20Format).

### Running Script

1.  Set up the data and directory.

    ```shell
    OUTPUT_DIR=/tmp/output && \
    TRAIN=tensorflow_ranking/examples/data/train.txt && \
    VALI=tensorflow_ranking/examples/data/vali.txt && \
    TEST=tensorflow_ranking/examples/data/test.txt
    ```

2.  Build and run.

    ```shell
    rm -rf $OUTPUT_DIR && \
    bazel build -c opt \
    tensorflow_ranking/examples/tf_ranking_libsvm_py_binary && \
    ./bazel-bin/tensorflow_ranking/examples/tf_ranking_libsvm_py_binary \
    --train_path=$TRAIN \
    --vali_path=$VALI \
    --test_path=$TEST \
    --output_dir=$OUTPUT_DIR \
    --num_features=136 \
    --num_train_steps=100
    ```

### TensorBoard

The training results such as loss and metrics can be visualized using
[Tensorboard](https://github.com/tensorflow/tensorboard/blob/master/README.md).

1.  (Optional) If you are working on remote server, set up port forwarding with
    this command.

    ```shell
    $ ssh <remote-server> -L 8888:127.0.0.1:8888
    ```

2.  Install Tensorboard and invoke it with the following commands.

    ```shell
    (tfr) $ pip install tensorboard
    (tfr) $ tensorboard --logdir $OUTPUT_DIR
    ```

### Jupyter Notebook

An example jupyter notebook using
[the LIBSVM format](https://sourceforge.net/p/lemur/wiki/RankLib%20File%20Format)
is available in `tensorflow_ranking/examples/tf_ranking_libsvm.ipynb`.

1.  To run this notebook, first follow the steps in installation to set up
    `virtualenv` environment with tensorflow_ranking package installed.

2.  Install jupyter within virtualenv.

    ```shell
    (tfr) $ pip install jupyter
    ```

3.  Start a jupyter notebook instance on remote server.

    ```shell
    (tfr) $ jupyter notebook tensorflow_ranking/examples/tf_ranking_libsvm.ipynb \
            --NotebookApp.allow_origin='https://colab.research.google.com' \
            --port=8888
    ```

4.  (Optional) If you are working on remote server, set up port forwarding with
    this command.

    ```shell
    $ ssh <remote-server> -L 8888:127.0.0.1:8888
    ```

5.  Running the notebook.

    *   Start jupyter notebook on your local machine at
        [http://localhost:8888/](http://localhost:8888/) and browse to the
        ipython notebook.

    *   An alternative is to use colaboratory notebook via
        [colab.research.google.com](http://colab.research.google.com) and open
        the notebook in the browser. Choose local runtime and link to port 8888.

## References

+   Rama Kumar Pasumarthi, Xuanhui Wang, Cheng Li, Sebastian Bruch, Michael
    Bendersky, Marc Najork, Jan Pfeifer, Nadav Golbandi, Rohan Anil, Stephan
    Wolf. _TF-Ranking: Scalable TensorFlow Library for Learning-to-Rank._
    [CoRR abs/1812.00073 (2018)](https://arxiv.org/abs/1812.00073)


+   Qingyao Ai, Xuanhui Wang, Nadav Golbandi, Michael Bendersky, Marc Najork.
    _Learning Groupwise Scoring Functions Using Deep Neural Networks._
    [CoRR abs/1811.04415 (2018)](https://arxiv.org/abs/1811.04415)


+   Xuanhui Wang, Michael Bendersky, Donald Metzler, and Marc Najork.
    _Learning to Rank with Selection Bias in Personal Search._
    [SIGIR 2016.](https://ai.google/research/pubs/pub45286)


+   Xuanhui Wang, Cheng Li, Nadav Golbandi, Mike Bendersky, Marc Najork.
    _The LambdaLoss Framework for Ranking Metric Optimization_.
    [CIKM 2018.](https://ai.google/research/pubs/pub47258)

### Citation

If you use TensorFlow Ranking in your research and would like to cite it, we
suggest you use the following citation:

       @misc{TensorflowRanking2018,
       author = {Rama Kumar Pasumarthi and Xuanhui Wang and Cheng Li and Sebastian Bruch and Michael Bendersky and Marc Najork and Jan Pfeifer and Nadav Golbandi and Rohan Anil and Stephan Wolf},
       title = {TF-Ranking: Scalable TensorFlow Library for Learning-to-Rank},
       year = {2018},
       eprint = {arXiv:1812.00073},
       }
