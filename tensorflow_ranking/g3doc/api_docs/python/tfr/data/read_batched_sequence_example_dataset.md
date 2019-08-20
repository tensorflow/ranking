<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.read_batched_sequence_example_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.read_batched_sequence_example_dataset

Returns a `Dataset` of features from `SequenceExample`.

```python
tfr.data.read_batched_sequence_example_dataset(
    file_pattern,
    batch_size,
    list_size,
    context_feature_spec,
    example_feature_spec,
    reader=tf.data.TFRecordDataset,
    reader_args=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=1000,
    shuffle_seed=None,
    prefetch_buffer_size=32,
    reader_num_threads=10,
    sloppy_ordering=True,
    drop_final_batch=False
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

#### Example:

```
data = [
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
context_features: {
  "query_length": parsing_ops.FixedenFeature([1], dtypes.int64)
}
example_features: {
  "unigrams": parsing_ops.VarLenFeature(dtypes.string),
  "utility": parsing_ops.FixedLenFeature([1], dtypes.float32,
  default_value=[0.])
}
batch_size: 2
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

*   <b>`file_pattern`</b>: (str | list(str)) List of files or patterns of file
    paths containing tf.SequenceExample protos. See `tf.gfile.Glob` for pattern
    rules.
*   <b>`batch_size`</b>: (int) Number of records to combine in a single batch.
*   <b>`list_size`</b>: (int) The number of frames to keep in a SequenceExample.
    If specified, truncation or padding may happen. Otherwise, set it to None to
    allow dynamic list size.
*   <b>`context_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values.
*   <b>`example_feature_spec`</b>: (dict) A mapping feature keys to
    `FixedLenFeature` or `VarLenFeature` values.
*   <b>`reader`</b>: A function or class that can be called with a `filenames`
    tensor and (optional) `reader_args` and returns a `Dataset`. Defaults to
    `tf.data.TFRecordDataset`.
*   <b>`reader_args`</b>: (list) Additional argument list to pass to the reader
    class.
*   <b>`num_epochs`</b>: (int) Number of times to read through the dataset. If
    None, cycles through the dataset forever. Defaults to `None`.
*   <b>`shuffle`</b>: (bool) Indicates whether the input should be shuffled.
    Defaults to `True`.
*   <b>`shuffle_buffer_size`</b>: (int) Buffer size of the ShuffleDataset. A
    large capacity ensures better shuffling but would increase memory usage and
    startup time.
*   <b>`shuffle_seed`</b>: (int) Randomization seed to use for shuffling.
*   <b>`prefetch_buffer_size`</b>: (int) Number of feature batches to prefetch
    in order to improve performance. Recommended value is the number of batches
    consumed per training step (default is 1).
*   <b>`reader_num_threads`</b>: (int) Number of threads used to read records.
    If greater than 1, the results will be interleaved.
*   <b>`sloppy_ordering`</b>: (bool) If `True`, reading performance will be
    improved at the cost of non-deterministic ordering. If `False`, the order of
    elements produced is deterministic prior to shuffling (elements are still
    randomized if `shuffle=True`. Note that if the seed is set, then order of
    elements after shuffling is deterministic). Defaults to `False`.
*   <b>`drop_final_batch`</b>: (bool) If `True`, and the batch size does not
    evenly divide the input dataset size, the final smaller batch will be
    dropped. Defaults to `True`. If `True`, the batch_size can be statically
    inferred.

#### Returns:

A dataset of `dict` elements. Each `dict` maps feature keys to `Tensor` or
`SparseTensor` objects. The context features are mapped to a rank-2 tensor of
shape [batch_size, feature_size], and the example features are mapped to a
rank-3 tensor of shape [batch_size, list_size, feature_size], where list_size is
the number of examples.
