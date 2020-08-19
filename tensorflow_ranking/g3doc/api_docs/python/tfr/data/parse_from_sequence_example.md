<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.parse_from_sequence_example" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.parse_from_sequence_example

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Parses SequenceExample to feature maps.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.data.parse_from_sequence_example(
    serialized, list_size=None, context_feature_spec=None,
    example_feature_spec=None, size_feature_name=None, shuffle_examples=False,
    seed=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

The `FixedLenFeature` in `example_feature_spec` is converted to
`FixedLenSequenceFeature` to parse `feature_list` in SequenceExample. We keep
track of the non-trivial default_values (e.g., -1 for labels) for features in
`example_feature_spec` and use them to replace the parsing defaults of the
SequenceExample (i.e., 0 for numbers and "" for strings). Due to this
complexity, we only allow scalar non-trivial default values for numbers.

When `list_size` is None, the 2nd dim of the output Tensors are not fixed and
vary from batch to batch. When `list_size` is specified as a positive integer,
truncation or padding is applied so that the 2nd dim of the output Tensors is
the specified `list_size`.

#### Example:

```
serialized = [
  sequence_example {
    context {
      feature {
        key: "query_length"
        value { int64_list { value: 3 } }
      }
    }
    feature_lists {
      feature_list {
        key: "unigrams"
        value {
          feature { bytes_list { value: "tensorflow" } }
          feature { bytes_list { value: ["learning" "to" "rank"] } }
        }
      }
      feature_list {
        key: "utility"
        value {
          feature { float_list { value: 0.0 } }
          feature { float_list { value: 1.0 } }
        }
      }
    }
  }
  sequence_example {
    context {
      feature {
        key: "query_length"
        value { int64_list { value: 2 } }
      }
    }
    feature_lists {
      feature_list {
        key: "unigrams"
        value {
          feature { bytes_list { value: "gbdt" } }
          feature { }
        }
      }
      feature_list {
        key: "utility"
        value {
          feature { float_list { value: 0.0 } }
          feature { float_list { value: 0.0 } }
        }
      }
    }
  }
]
```

#### We can use arguments:

```
context_feature_spec: {
  "query_length": tf.io.FixedenFeature([1], dtypes.int64)
}
example_feature_spec: {
  "unigrams": tf.io.VarLenFeature(dtypes.string),
  "utility": tf.io.FixedLenFeature([1], dtypes.float32,
    default_value=[0.])
}
```

And the expected output is:

```python
{
  "unigrams": SparseTensor(
    indices=array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [1,
    1, 0], [1, 1, 1]]),
    values=["tensorflow", "learning", "to", "rank", "gbdt"],
    dense_shape=array([2, 2, 3])),
  "utility": [[[ 0.], [ 1.]], [[ 0.], [ 0.]]],
  "query_length": [[3], [2]],
}
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`serialized`
</td>
<td>
(Tensor) A string Tensor for a batch of serialized
SequenceExample.
</td>
</tr><tr>
<td>
`list_size`
</td>
<td>
(int) The number of frames to keep for a SequenceExample. If
specified, truncation or padding may happen. Otherwise, the output Tensors
have a dynamic list size.
</td>
</tr><tr>
<td>
`context_feature_spec`
</td>
<td>
(dict) A mapping from feature keys to
`FixedLenFeature` or `VarLenFeature` values for context.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
(dict) A mapping from feature keys to
`FixedLenFeature` or `VarLenFeature` values for the list of examples.
These features are stored in the `feature_lists` field in SequenceExample.
`FixedLenFeature` is translated to `FixedLenSequenceFeature` to parse
SequenceExample. Note that no missing value in the middle of a
`feature_list` is allowed for frames.
</td>
</tr><tr>
<td>
`size_feature_name`
</td>
<td>
(str) Name of feature for example list sizes. Populates
the feature dictionary with a `tf.int32` Tensor of shape [batch_size] for
this feature name. If None, which is default, this feature is not
generated.
</td>
</tr><tr>
<td>
`shuffle_examples`
</td>
<td>
(bool) A boolean to indicate whether examples within a
list are shuffled before the list is trimmed down to list_size elements
(when list has more than list_size elements).
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
(int) A seed passed onto random_ops.uniform() to shuffle examples.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A mapping from feature keys to `Tensor` or `SparseTensor`.
</td>
</tr>

</table>
