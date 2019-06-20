<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.parse_from_sequence_example" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.parse_from_sequence_example

Parses SequenceExample to feature maps.

```python
tfr.data.parse_from_sequence_example(
    serialized,
    list_size=None,
    context_feature_spec=None,
    example_feature_spec=None
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

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

#### Args:

*   <b>`serialized`</b>: (Tensor) A string Tensor for a batch of serialized
    SequenceExample.
*   <b>`list_size`</b>: (int) The number of frames to keep for a
    SequenceExample. If specified, truncation or padding may happen. Otherwise,
    the output Tensors have a dynamic list size.
*   <b>`context_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values for context.
*   <b>`example_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values for the list of examples. These
    features are stored in the `feature_lists` field in SequenceExample.
    `FixedLenFeature` is translated to `FixedLenSequenceFeature` to parse
    SequenceExample. Note that no missing value in the middle of a
    `feature_list` is allowed for frames.

#### Returns:

A mapping from feature keys to `Tensor` or `SparseTensor`.
