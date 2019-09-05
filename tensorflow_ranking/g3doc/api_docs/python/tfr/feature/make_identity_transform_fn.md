<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.feature.make_identity_transform_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.feature.make_identity_transform_fn

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/feature.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Returns transform fn that split the features.

```python
tfr.feature.make_identity_transform_fn(context_feature_names)
```

<!-- Placeholder for "Used in" -->

The make_identity_transform_fn generates a transform_fn which handles only
non-prefixed features. The per-example features need to have shape [batch_size,
input_size, ...] and the context features need to have shape [batch_size, ...].

#### Args:

*   <b>`context_feature_names`</b>: A list of strings representing the context
    feature names.

#### Returns:

An identity transform function that splits into context and per example
features.
