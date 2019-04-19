<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="ranking.model.make_groupwise_ranking_fn" />
<meta itemprop="path" content="Stable" />
</div>

# ranking.model.make_groupwise_ranking_fn

``` python
ranking.model.make_groupwise_ranking_fn(
    group_score_fn,
    group_size,
    ranking_head,
    transform_fn=None
)
```



Defined in [`python/model.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/model.py).

<!-- Placeholder for "Used in" -->

Builds an `Estimator` model_fn for groupwise comparison ranking models.

#### Args:

* <b>`group_score_fn`</b>: Scoring function for a group of examples with `group_size`
    that returns a score per example. It has to follow signature:
    * Args:
      `context_features`: A dict of `Tensor`s with shape [batch_size, ...].
      `per_example_features`: A dict of `Tensor`s with shape [batch_size,
        group_size, ...]
      `mode`: Optional. Specifies if this is training, evaluation or
        inference. See `ModeKeys`.
      `params`: Optional dict of hyperparameters, same value passed in the
        `Estimator` constructor.
      `config`: Optional configuration object, same value passed in the
        `Estimator` constructor.
    * Returns: Tensor of shape [batch_size, group_size] containing per-example
      scores.
* <b>`group_size`</b>: An integer denoting the number of examples in `group_score_fn`.
* <b>`ranking_head`</b>: A `head._RankingHead` object.
* <b>`transform_fn`</b>: Function transforming the raw features into dense tensors. It
    has the following signature:
    * Args:
      `features`: A dict of `Tensor`s contains the raw input.
      `mode`: Optional. See estimator `ModeKeys`.
    * Returns:
      `context_features`: A dict of `Tensor`s with shape [batch_size, ...]
      `per_example_features`: A dict of `Tensor`s with shape [batch_size,
        list_size, ...]


#### Returns:

An `Estimator` `model_fn` (see estimator.py) with the following signature:
* Args:
  * `features`: dict of Tensors of shape [batch_size, list_size, ...] for
  per-example features and shape [batch_size, ...] for non-example context
  features.
  * `labels`: Tensor with shape [batch_size, list_size] denoting relevance.
  * `mode`: No difference.
  * `params`: No difference.
  * `config`: No difference..
* Returns:
  `EstimatorSpec`

#### Raises:

* <b>`ValueError`</b>: when group_size is invalid.