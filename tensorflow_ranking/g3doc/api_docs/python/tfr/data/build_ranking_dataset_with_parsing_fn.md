description: Builds a ranking tf.dataset using the provided parsing_fn.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_ranking_dataset_with_parsing_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_ranking_dataset_with_parsing_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py#L857-L951">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a ranking tf.dataset using the provided `parsing_fn`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.data.build_ranking_dataset_with_parsing_fn(
    file_pattern, parsing_fn, batch_size,
    reader=tfr.keras.pipeline.DatasetHparams.dataset_reader, reader_args=None,
    num_epochs=None, shuffle=True, shuffle_buffer_size=10000, shuffle_seed=None,
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
    reader_num_threads=tf.data.experimental.AUTOTUNE, sloppy_ordering=False,
    drop_final_batch=False, num_parser_threads=tf.data.experimental.AUTOTUNE
)
</code></pre>

<!-- Placeholder for "Used in" -->

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
containing serialized data. See `tf.gfile.Glob` for pattern rules.
</td>
</tr><tr>
<td>
`parsing_fn`
</td>
<td>
(function) It has a single argument parsing_fn(serialized).
Users can customize this for their own data formats.
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
consumed per training step. Defaults to auto-tune.
</td>
</tr><tr>
<td>
`reader_num_threads`
</td>
<td>
(int) Number of threads used to read records. If greater
than 1, the results will be interleaved. Defaults to auto-tune.
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
</tr><tr>
<td>
`num_parser_threads`
</td>
<td>
(int) Optional number of threads to be used with
dataset.map() when invoking parsing_fn. Defaults to auto-tune.
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
`Tensor` or `SparseTensor` objects.
</td>
</tr>

</table>
