<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.estimator" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.estimator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/estimator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Provides an `EstimatorBuilder` for creating a TF-Ranking model estimator.

This class contains the boilerplate that is required to create an estimator for
a TF-Ranking model. The goal is to reduce replicated setups (e.g., transform
function, scoring function) for adopting TF-Ranking. Advanced users can also
derive from this class and further tailor for their needs.

## Classes

[`class EstimatorBuilder`](../tfr/estimator/EstimatorBuilder.md): Builds a
tf.estimator.Estimator for a TF-Ranking model.

[`class GAMEstimatorBuilder`](../tfr/estimator/GAMEstimatorBuilder.md): Builds a
TFR estimator with subscore signatures of GAM models.

## Functions

[`make_dnn_ranking_estimator(...)`](../tfr/estimator/make_dnn_ranking_estimator.md):
Builds an `Estimator` instance with DNN scoring function.

[`make_gam_ranking_estimator(...)`](../tfr/estimator/make_gam_ranking_estimator.md):
Builds an `Estimator` instance with GAM scoring function.
