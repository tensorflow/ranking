<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.model.make_groupwise_ranking_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.model.make_groupwise_ranking_fn

Builds an `Estimator` model_fn for groupwise comparison ranking models.

```python
tfr.model.make_groupwise_ranking_fn(
    group_score_fn,
    group_size,
    ranking_head,
    transform_fn=None
)
```

Defined in
[`python/model.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/model.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`group_score_fn`</b>: See `_GroupwiseRankingModel`.
*   <b>`group_size`</b>: See `_GroupwiseRankingModel`.
*   <b>`ranking_head`</b>: A `head._RankingHead` object.
*   <b>`transform_fn`</b>: See `_GroupwiseRankingModel`.

#### Returns:

An `Estimator` `model_fn` with the following signature: * Args: `features`: The
raw features from input_fn. `labels`: A Tensor with shape [batch_size,
list_size]. `mode`: No difference. `params`: No difference. `config`: No
difference.. * Returns: `EstimatorSpec`.
