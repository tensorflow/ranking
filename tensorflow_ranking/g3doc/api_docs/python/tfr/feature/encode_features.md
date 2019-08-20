<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.feature.encode_features" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.feature.encode_features

Returns dense tensors from features using feature columns.

```python
tfr.feature.encode_features(
    features,
    feature_columns,
    mode=tf.estimator.ModeKeys.TRAIN,
    scope=None
)
```

Defined in
[`python/feature.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/feature.py).

<!-- Placeholder for "Used in" -->

This function encodes the feature column transformation on the 'raw' `features`.

#### Args:

*   <b>`features`</b>: (dict) mapping feature names to feature values, possibly
    obtained from input_fn.
*   <b>`feature_columns`</b>: (list) list of feature columns.
*   <b>`mode`</b>: (`estimator.ModeKeys`) Specifies if this is training,
    evaluation or inference. See `ModeKeys`.
*   <b>`scope`</b>: (str) variable scope for the per column input layers.

#### Returns:

(dict) A mapping from columns to dense tensors.
