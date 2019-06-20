<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_ranking_dataset_with_parsing_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_ranking_dataset_with_parsing_fn

Builds a ranking tf.dataset using the provided `parsing_fn`.

```python
tfr.data.build_ranking_dataset_with_parsing_fn(
    file_pattern,
    parsing_fn,
    batch_size,
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

#### Args:

*   <b>`file_pattern`</b>: (str | list(str)) List of files or patterns of file
    paths containing serialized data. See `tf.gfile.Glob` for pattern rules.
*   <b>`parsing_fn`</b>: (function) It has a single argument
    parsing_fn(serialized). Users can customize this for their own data formats.
*   <b>`batch_size`</b>: (int) Number of records to combine in a single batch.
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
`SparseTensor` objects.
