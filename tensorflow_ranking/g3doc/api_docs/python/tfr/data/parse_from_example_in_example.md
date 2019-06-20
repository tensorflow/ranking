<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.parse_from_example_in_example" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.parse_from_example_in_example

Parses an ExampleInExample batch to a feature map.

```python
tfr.data.parse_from_example_in_example(
    serialized,
    list_size=None,
    context_feature_spec=None,
    example_feature_spec=None
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

An ExampleInExample is a tf.train.Example that has two fields: -
`serialized_context` is a scalar of bytes. The value is a serialized
tf.train.Example that contains context features. - `serialized_examples` is a
repeated field of bytes. The value is a list of serialized tf.train.Example with
each representing an example that contains example features.

#### For example:

```
serialized_context_string = Serialize({
  features {
    feature {
      key: "query_length"
      value { int64_list { value: 3 } }
    }
  }
})

serialized_examples_string = [
  Serialize({
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
  }),

  Serialize({
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
  })
]

serialized_context_string_2 = Serialize({
  features {
    feature {
      key: "query_length"
      value { int64_list { value: 2 } }
    }
  }
})

serialized_examples_string_2 = [
  Serialize({
    features {
      feature {
        key: "unigrams"
        value { bytes_list { value: "gbdt" } }
      }
      feature {
        key: "utility"
        value { float_list { value: 0.0 } }
      }
    }
  }),

  Serialize({
    features {
      feature {
        key: "unigrams"
        value { bytes_list { value: ["neural" "network" } }
      }
      feature {
        key: "utility"
        value { float_list { value: 1.0 } }
      }
    }
  })
]

serialized = [
  {
    serialized_context: serialized_context_string,
    serialized_examples: serialized_examples_string,
  },
  {
    serialized_context: serialized_context_string_2,
    serialized_examples: serialized_examples_string_2,
  },
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

#### Returns:

A mapping from feature keys to `Tensor` or `SparseTensor`.
