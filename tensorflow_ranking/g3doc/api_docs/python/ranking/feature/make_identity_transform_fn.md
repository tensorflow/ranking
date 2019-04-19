<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="ranking.feature.make_identity_transform_fn" />
<meta itemprop="path" content="Stable" />
</div>

# ranking.feature.make_identity_transform_fn

``` python
ranking.feature.make_identity_transform_fn(context_feature_names)
```



Defined in [`python/feature.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/feature.py).

<!-- Placeholder for "Used in" -->

Returns transform fn that split the features.

  The make_identity_transform_fn generates a transform_fn which handles only
  non-prefixed features. The per-example features need to have shape
  [batch_size, input_size, ...] and the context features need to have shape
  [batch_size, ...].

#### Args:

* <b>`context_feature_names`</b>: A list of strings representing the context feature
    names.


#### Returns:

An identity transform function that splits into context and per example
features.