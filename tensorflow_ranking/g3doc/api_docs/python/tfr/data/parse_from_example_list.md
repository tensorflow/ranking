<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.parse_from_example_list" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.parse_from_example_list

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Parses an `ExampleListWithContext` batch to a feature map.

```python
tfr.data.parse_from_example_list(
    serialized, list_size=None, context_feature_spec=None,
    example_feature_spec=None, size_feature_name=None, shuffle_examples=False,
    seed=None
)
```

<!-- Placeholder for "Used in" -->

#### Example:

```
serialized = [
  example_list_with_context = {
    context {
      features {
        feature {
          key: "query_length"
          value { int64_list { value: 3 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: "tensorflow" } }
        }
        feature {
          key: "utility"
          value { float_list { value: 0.0 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: ["learning" "to" "rank" } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
      }
    }
  }
  example_list_with_context = {
    context {
      features {
        feature {
          key: "query_length"
          value { int64_list { value: 2 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: ["gbdt"] } }
        }
        feature {
          key: "utility"
          value { float_list { value: 0.0 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: ["neural", "networks"] } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
      }
    }
  }
]
```

#### We can use arguments:

```
context_feature_spec: {
  "query_length": tf.io.FixedenFeature([1], dtypes.int64),
}
example_feature_spec: {
  "unigrams": tf.io.VarLenFeature(dtypes.string),
  "utility": tf.io.FixedLenFeature([1], dtypes.float32),
}
```

And the expected output is:

```python
{
  "unigrams": SparseTensor(
    indices=array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0],
      [1, 1, 0], [1, 1, 1]]),
    values=["tensorflow", "learning", "to", "rank", "gbdt", "neural" ,
      "network"],
    dense_shape=array([2, 2, 3])),
  "utility": [[[ 0.], [ 1.]], [[ 0.], [ 1.]]],
  "query_length": [[3], [2]],
}
```

#### Args:

*   <b>`serialized`</b>: (Tensor) 1-D Tensor and each entry is a serialized
    `ExampleListWithContext` proto that contains context and example list.
*   <b>`list_size`</b>: (int) The number of examples for each list. If
    specified, truncation or padding is applied to make 2nd dim of output
    Tensors aligned to `list_size`. Otherwise, the 2nd dim of the output Tensors
    is dynamic.
*   <b>`context_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values for context in
    `ExampleListWithContext` proto.
*   <b>`example_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values for examples in
    `ExampleListWithContext` proto.
*   <b>`size_feature_name`</b>: (str) Name of feature for example list sizes.
    Populates the feature dictionary with a `tf.int32` Tensor of shape
    [batch_size] for this feature name. If None, which is default, this feature
    is not generated.
*   <b>`shuffle_examples`</b>: (bool) A boolean to indicate whether examples
    within a list are shuffled before the list is trimmed down to list_size
    elements (when list has more than list_size elements).
*   <b>`seed`</b>: (int) A seed passed onto random_ops.uniform() to shuffle
    examples.

#### Returns:

A mapping from feature keys to `Tensor` or `SparseTensor`.
