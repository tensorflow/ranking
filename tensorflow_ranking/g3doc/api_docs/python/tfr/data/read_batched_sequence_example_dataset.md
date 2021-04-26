description: Returns a Dataset of features from SequenceExample.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.read_batched_sequence_example_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.read_batched_sequence_example_dataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py#L1083-L1244">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a `Dataset` of features from `SequenceExample`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.data.read_batched_sequence_example_dataset(
    file_pattern, batch_size, list_size, context_feature_spec, example_feature_spec,
    reader=tfr.keras.pipeline.DatasetHparams.dataset_reader, reader_args=None,
    num_epochs=None, shuffle=True, shuffle_buffer_size=1000, shuffle_seed=None,
    prefetch_buffer_size=32, reader_num_threads=10, sloppy_ordering=True,
    drop_final_batch=False
)
</code></pre>

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

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`file_pattern`
</td>
<td>
(str | list(str)) List of files or patterns of file paths
containing tf.SequenceExample protos. See `tf.gfile.Glob` for pattern
rules.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
(int) Number of records to combine in a single batch.
</td>
</tr><tr>
<td>
`list_size`
</td>
<td>
(int) The number of frames to keep in a SequenceExample. If
specified, truncation or padding may happen. Otherwise, set it to None to
allow dynamic list size.
</td>
</tr><tr>
<td>
`context_feature_spec`
</td>
<td>
(dict) A mapping from  feature keys to
`FixedLenFeature` or `VarLenFeature` values.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
(dict) A mapping feature keys to `FixedLenFeature` or
`VarLenFeature` values.
</td>
</tr><tr>
<td>
`reader`
</td>
<td>
A function or class that can be called with a `filenames` tensor and
(optional) `reader_args` and returns a `Dataset`. Defaults to
`tf.data.TFRecordDataset`.
</td>
</tr><tr>
<td>
`reader_args`
</td>
<td>
(list) Additional argument list to pass to the reader class.
</td>
</tr><tr>
<td>
`num_epochs`
</td>
<td>
(int) Number of times to read through the dataset. If None,
cycles through the dataset forever. Defaults to `None`.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
(bool) Indicates whether the input should be shuffled. Defaults to
`True`.
</td>
</tr><tr>
<td>
`shuffle_buffer_size`
</td>
<td>
(int) Buffer size of the ShuffleDataset. A large
capacity ensures better shuffling but would increase memory usage and
startup time.
</td>
</tr><tr>
<td>
`shuffle_seed`
</td>
<td>
(int) Randomization seed to use for shuffling.
</td>
</tr><tr>
<td>
`prefetch_buffer_size`
</td>
<td>
(int) Number of feature batches to prefetch in order
to improve performance. Recommended value is the number of batches
consumed per training step (default is 1).
</td>
</tr><tr>
<td>
`reader_num_threads`
</td>
<td>
(int) Number of threads used to read records. If greater
than 1, the results will be interleaved.
</td>
</tr><tr>
<td>
`sloppy_ordering`
</td>
<td>
(bool) If `True`, reading performance will be improved at
the cost of non-deterministic ordering. If `False`, the order of elements
produced is deterministic prior to shuffling (elements are still
randomized if `shuffle=True`. Note that if the seed is set, then order of
elements after shuffling is deterministic). Defaults to `False`.
</td>
</tr><tr>
<td>
`drop_final_batch`
</td>
<td>
(bool) If `True`, and the batch size does not evenly
divide the input dataset size, the final smaller batch will be dropped.
Defaults to `False`. If `True`, the batch_size can be statically inferred.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dataset of `dict` elements. Each `dict` maps feature keys to
`Tensor` or `SparseTensor` objects. The context features are mapped to a
rank-2 tensor of shape [batch_size, feature_size], and the example features
are mapped to a rank-3 tensor of shape [batch_size, list_size,
feature_size], where list_size is the number of examples.
</td>
</tr>

</table>
