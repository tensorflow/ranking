# TensorFlow Ranking

TensorFlow Ranking is a library for Learning-to-Rank (LTR) techniques on the
TensorFlow platform. It contains the following components:

*   Commonly used loss functions including pointwise, pairwise, and listwise
    losses.
*   Commonly used ranking metrics like
    [Mean Reciprocal Rank (MRR)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
    and
    [Normalized Discounted Cumulative Gain (NDCG)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain).
*   [Multi-item (also known as groupwise) scoring functions](https://arxiv.org/abs/1811.04415).
*   [LambdaLoss](https://ai.google/research/pubs/pub47258) implementation for
    direct ranking metric optimization.
*   [Unbiased Learning-to-Rank](http://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf)
    from biased feedback data.

We envision that this library will provide a convenient open platform for
hosting and advancing state-of-the-art ranking models based on deep learning
techniques, and thus facilitate both academic research and industrial
applications.

## Tutorial Slides

TF-Ranking was presented at premier conferences in Information Retrieval,
[SIGIR 2019](https://sigir.org/sigir2019/program/tutorials/) and
[ICTIR 2019](http://ictir2019.org/program/#tutorials)! The slides are available
[here](http://bendersky.github.io/res/TF-Ranking-ICTIR-2019.pdf).

## Demos

We provide a demo, with no installation required, to get started on using
TF-Ranking. This demo runs on a
[colaboratory notebook](https://research.google.com/colaboratory/faq.html), an
interactive Python environment. Using sparse features and embeddings in
TF-Ranking
[![Run in Google Colab](https://www.tensorflow.org/images/colab_logo_32px.png)](https://colab.research.google.com/github/tensorflow/ranking/blob/master/tensorflow_ranking/examples/handling_sparse_features.ipynb).
This demo demonstrates how to:

*   Use sparse/embedding features
*   Process data in TFRecord format
*   Tensorboard integration in colab notebook, for Estimator API

Also see [Running Scripts](#running-scripts) for executable scripts.

## Linux Installation

### Stable Builds

To install the latest version from
[PyPI](https://pypi.org/project/tensorflow-ranking/), run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tensorflow_ranking
```

To force a Python 3-specific install, replace `pip` with `pip3` in the above
commands. For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the
[TensorFlow installation guide](https://www.tensorflow.org/install).

Note: Since TensorFlow is now included as a dependency of the TensorFlow Ranking
package (in `setup.py`). If you wish to use different versions of TensorFlow
(e.g., `tensorflow-gpu`), you may need to uninstall the existing verison and
then install your desired version:

```shell
$ pip uninstall tensorflow
$ pip install tensorflow-gpu
```

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
    (tfr) $ pip install /tmp/ranking_pip/tensorflow_ranking*.whl
    ```

    In some cases, you may want to install a specific version of tensorflow,
    e.g., `tensorflow-gpu` or `tensorflow==2.0.0`. To do so you can either

    ```shell
    (tfr) $ pip uninstall tensorflow
    (tfr) $ pip install tensorflow==2.0.0
    ```

    or

    ```shell
    (tfr) $ pip uninstall tensorflow
    (tfr) $ pip install tensorflow-gpu
    ```

5.  Run all TensorFlow Ranking tests.

    ```shell
    (tfr) $ bazel test //tensorflow_ranking/...
    ```

6.  Invoke TensorFlow Ranking package in python (within virtualenv).

    ```shell
    (tfr) $ python -c "import tensorflow_ranking"
    ```

## Running Scripts

For ease of experimentation, we also provide
[a TFRecord example](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_tfrecord.py)
and
[a LIBSVM example](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_libsvm.py)
in the form of executable scripts. This is particularly useful for
hyperparameter tuning, where the hyperparameters are supplied as flags to the
script.

### TFRecord Example

1.  Set up the data and directory.

    ```shell
    MODEL_DIR=/tmp/tf_record_model && \
    TRAIN=tensorflow_ranking/examples/data/train_elwc.tfrecord && \
    EVAL=tensorflow_ranking/examples/data/eval_elwc.tfrecord && \
    VOCAB=tensorflow_ranking/examples/data/vocab.txt
    ```

2.  Build and run.

    ```shell
    rm -rf $MODEL_DIR && \
    bazel build -c opt \
    tensorflow_ranking/examples/tf_ranking_tfrecord_py_binary && \
    ./bazel-bin/tensorflow_ranking/examples/tf_ranking_tfrecord_py_binary \
    --train_path=$TRAIN \
    --eval_path=$EVAL \
    --vocab_path=$VOCAB \
    --model_dir=$MODEL_DIR \
    --data_format=example_list_with_context
    ```

### LIBSVM Example

1.  Set up the data and directory.

    ```shell
    OUTPUT_DIR=/tmp/libsvm && \
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

An example jupyter notebook is available in
`third_party/tensorflow_ranking/examples/handling_sparse_features.ipynb`.

1.  To run this notebook, first follow the steps in installation to set up
    `virtualenv` environment with tensorflow_ranking package installed.

2.  Install jupyter within virtualenv.

    ```shell
    (tfr) $ pip install jupyter
    ```

3.  Start a jupyter notebook instance on remote server.

    ```shell
    (tfr) $ jupyter notebook third_party/tensorflow_ranking/examples/handling_sparse_features.ipynb \
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

+   Rama Kumar Pasumarthi, Sebastian Bruch, Xuanhui Wang, Cheng Li, Michael
    Bendersky, Marc Najork, Jan Pfeifer, Nadav Golbandi, Rohan Anil, Stephan
    Wolf. _TF-Ranking: Scalable TensorFlow Library for Learning-to-Rank._
    [KDD 2019.](https://ai.google/research/pubs/pub48160)

+   Qingyao Ai, Xuanhui Wang, Sebastian Bruch, Nadav Golbandi, Michael
    Bendersky, Marc Najork. _Learning Groupwise Scoring Functions Using Deep
    Neural Networks._ [ICTIR 2019](https://ai.google/research/pubs/pub48348)

+   Xuanhui Wang, Michael Bendersky, Donald Metzler, and Marc Najork. _Learning
    to Rank with Selection Bias in Personal Search._
    [SIGIR 2016.](https://ai.google/research/pubs/pub45286)

+   Xuanhui Wang, Cheng Li, Nadav Golbandi, Mike Bendersky, Marc Najork. _The
    LambdaLoss Framework for Ranking Metric Optimization_.
    [CIKM 2018.](https://ai.google/research/pubs/pub47258)

### Citation

If you use TensorFlow Ranking in your research and would like to cite it, we
suggest you use the following citation:

    @inproceedings{TensorflowRankingKDD2019,
       author = {Rama Kumar Pasumarthi and Sebastian Bruch and Xuanhui Wang and Cheng Li and Michael Bendersky and Marc Najork and Jan Pfeifer and Nadav Golbandi and Rohan Anil and Stephan Wolf},
       title = {TF-Ranking: Scalable TensorFlow Library for Learning-to-Rank},
       booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
       year = {2019},
       pages = {2970--2978},
       location = {Anchorage, AK}
    }
