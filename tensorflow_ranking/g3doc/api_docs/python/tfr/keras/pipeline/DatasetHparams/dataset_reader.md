description: A Dataset comprising records from one or more TFRecord files.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.DatasetHparams.dataset_reader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__nonzero__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="as_numpy_iterator"/>
<meta itemprop="property" content="batch"/>
<meta itemprop="property" content="bucket_by_sequence_length"/>
<meta itemprop="property" content="cache"/>
<meta itemprop="property" content="cardinality"/>
<meta itemprop="property" content="concatenate"/>
<meta itemprop="property" content="enumerate"/>
<meta itemprop="property" content="filter"/>
<meta itemprop="property" content="flat_map"/>
<meta itemprop="property" content="from_generator"/>
<meta itemprop="property" content="from_tensor_slices"/>
<meta itemprop="property" content="from_tensors"/>
<meta itemprop="property" content="get_single_element"/>
<meta itemprop="property" content="group_by_window"/>
<meta itemprop="property" content="interleave"/>
<meta itemprop="property" content="list_files"/>
<meta itemprop="property" content="map"/>
<meta itemprop="property" content="options"/>
<meta itemprop="property" content="padded_batch"/>
<meta itemprop="property" content="prefetch"/>
<meta itemprop="property" content="random"/>
<meta itemprop="property" content="range"/>
<meta itemprop="property" content="reduce"/>
<meta itemprop="property" content="repeat"/>
<meta itemprop="property" content="shard"/>
<meta itemprop="property" content="shuffle"/>
<meta itemprop="property" content="skip"/>
<meta itemprop="property" content="take"/>
<meta itemprop="property" content="unbatch"/>
<meta itemprop="property" content="window"/>
<meta itemprop="property" content="with_options"/>
<meta itemprop="property" content="zip"/>
</div>

# tfr.keras.pipeline.DatasetHparams.dataset_reader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

A `Dataset` comprising records from one or more TFRecord files.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.DatasetHparams.dataset_reader(
    filenames, compression_type=None, buffer_size=None, num_parallel_reads=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

This dataset loads TFRecords from the files as bytes, exactly as they were
written.`TFRecordDataset` does not do any parsing or decoding on its own.
Parsing and decoding can be done by applying `Dataset.map` transformations after
the `TFRecordDataset`.

A minimal example is given below:

```
>>> import tempfile
>>> example_path = os.path.join(tempfile.gettempdir(), "example.tfrecords")
>>> np.random.seed(0)
```

```
>>> # Write the records to a file.
... with tf.io.TFRecordWriter(example_path) as file_writer:
...   for _ in range(4):
...     x, y = np.random.random(), np.random.random()
...
...     record_bytes = tf.train.Example(features=tf.train.Features(feature={
...         "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
...         "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
...     })).SerializeToString()
...     file_writer.write(record_bytes)
```

```
>>> # Read the data back out.
>>> def decode_fn(record_bytes):
...   return tf.io.parse_single_example(
...       # Data
...       record_bytes,
...
...       # Schema
...       {"x": tf.io.FixedLenFeature([], dtype=tf.float32),
...        "y": tf.io.FixedLenFeature([], dtype=tf.float32)}
...   )
```

```
>>> for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn):
...   print("x = {x:.4f},  y = {y:.4f}".format(**batch))
x = 0.5488,  y = 0.7152
x = 0.6028,  y = 0.5449
x = 0.4237,  y = 0.6459
x = 0.4376,  y = 0.8918
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filenames`
</td>
<td>
A `tf.string` tensor or `tf.data.Dataset` containing one or
more filenames.
</td>
</tr><tr>
<td>
`compression_type`
</td>
<td>
(Optional.) A `tf.string` scalar evaluating to one of
`""` (no compression), `"ZLIB"`, or `"GZIP"`.
</td>
</tr><tr>
<td>
`buffer_size`
</td>
<td>
(Optional.) A `tf.int64` scalar representing the number of
bytes in the read buffer. If your input pipeline is I/O bottlenecked,
consider setting this parameter to a value 1-100 MBs. If `None`, a
sensible default for both local and remote file systems is used.
</td>
</tr><tr>
<td>
`num_parallel_reads`
</td>
<td>
(Optional.) A `tf.int64` scalar representing the
number of files to read in parallel. If greater than one, the records of
files read in parallel are outputted in an interleaved order. If your
input pipeline is I/O bottlenecked, consider setting this parameter to a
value greater than one to parallelize the I/O. If `None`, files will be
read sequentially.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If any argument does not have the expected type.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If any argument does not have the expected shape.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `element_spec` </td> <td> The type specification of an element of this
dataset.

```
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset.element_spec
TensorSpec(shape=(), dtype=tf.int32, name=None)
```

For more information,
read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).
</td>
</tr>
</table>

## Methods

<h3 id="apply"><code>apply</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply(
    transformation_func
)
</code></pre>

Applies a transformation function to this dataset.

`apply` enables chaining of custom `Dataset` transformations, which are
represented as functions that take one `Dataset` argument and return a
transformed `Dataset`.

```
>>> dataset = tf.data.Dataset.range(100)
>>> def dataset_fn(ds):
...   return ds.filter(lambda x: x < 5)
>>> dataset = dataset.apply(dataset_fn)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2, 3, 4]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`transformation_func`
</td>
<td>
A function that takes one `Dataset` argument and
returns a `Dataset`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
The `Dataset` returned by applying `transformation_func` to this
dataset.
</td>
</tr>
</table>

<h3 id="as_numpy_iterator"><code>as_numpy_iterator</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_numpy_iterator()
</code></pre>

Returns an iterator which converts all elements of the dataset to numpy.

Use `as_numpy_iterator` to inspect the content of your dataset. To see element
shapes and types, print dataset elements directly instead of using
`as_numpy_iterator`.

```
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> for element in dataset:
...   print(element)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
```

This method requires that you are running in eager mode and the dataset's
element_spec contains only `TensorSpec` components.

```
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> for element in dataset.as_numpy_iterator():
...   print(element)
1
2
3
```

```
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> print(list(dataset.as_numpy_iterator()))
[1, 2, 3]
```

`as_numpy_iterator()` will preserve the nested structure of dataset elements.

```
>>> dataset = tf.data.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]),
...                                               'b': [5, 6]})
>>> list(dataset.as_numpy_iterator()) == [{'a': (1, 3), 'b': 5},
...                                       {'a': (2, 4), 'b': 6}]
True
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An iterable over the elements of the dataset, with their tensors converted
to numpy arrays.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
if an element contains a non-`Tensor` value.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
if eager execution is not enabled.
</td>
</tr>
</table>

<h3 id="batch"><code>batch</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch(
    batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None
)
</code></pre>

Combines consecutive elements of this dataset into batches.

```
>>> dataset = tf.data.Dataset.range(8)
>>> dataset = dataset.batch(3)
>>> list(dataset.as_numpy_iterator())
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]
```

```
>>> dataset = tf.data.Dataset.range(8)
>>> dataset = dataset.batch(3, drop_remainder=True)
>>> list(dataset.as_numpy_iterator())
[array([0, 1, 2]), array([3, 4, 5])]
```

The components of the resulting element will have an additional outer dimension,
which will be `batch_size` (or `N % batch_size` for the last element if
`batch_size` does not divide the number of input elements `N` evenly and
`drop_remainder` is `False`). If your program depends on the batches having the
same outer dimension, you should set the `drop_remainder` argument to `True` to
prevent the smaller batch from being produced.

Note: If your program requires data to have a statically known shape (e.g., when
using XLA), you should use `drop_remainder=True`. Without `drop_remainder=True`
the shape of the output dataset will have an unknown leading dimension due to
the possibility of a smaller final batch.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch_size`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the number of
consecutive elements of this dataset to combine in a single batch.
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
(Optional.) A `tf.bool` scalar `tf.Tensor`, representing
whether the last batch should be dropped in the case it has fewer than
`batch_size` elements; the default behavior is not to drop the smaller
batch.
</td>
</tr><tr>
<td>
`num_parallel_calls`
</td>
<td>
(Optional.) A `tf.int64` scalar `tf.Tensor`,
representing the number of batches to compute asynchronously in
parallel.
If not specified, batches will be computed sequentially. If the value
`tf.data.AUTOTUNE` is used, then the number of parallel
calls is set dynamically based on available resources.
</td>
</tr><tr>
<td>
`deterministic`
</td>
<td>
(Optional.) When `num_parallel_calls` is specified, if this
boolean is specified (`True` or `False`), it controls the order in which
the transformation produces elements. If set to `False`, the
transformation is allowed to yield elements out of order to trade
determinism for performance. If not specified, the
`tf.data.Options.experimental_deterministic` option
(`True` by default) controls the behavior.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="bucket_by_sequence_length"><code>bucket_by_sequence_length</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>bucket_by_sequence_length(
    element_length_func, bucket_boundaries, bucket_batch_sizes, padded_shapes=None,
    padding_values=None, pad_to_bucket_boundary=False, no_padding=False,
    drop_remainder=False
)
</code></pre>

A transformation that buckets elements in a `Dataset` by length.

Elements of the `Dataset` are grouped together by length and then are padded and
batched.

This is useful for sequence tasks in which the elements have variable length.
Grouping together elements that have similar lengths reduces the total fraction
of padding in a batch which increases training step efficiency.

Below is an example to bucketize the input data to the 3 buckets "[0, 3), [3,
5), [5, inf)" based on sequence length, with batch size 2.

```
>>> elements = [
...   [0], [1, 2, 3, 4], [5, 6, 7],
...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]
>>> dataset = tf.data.Dataset.from_generator(
...     lambda: elements, tf.int64, output_shapes=[None])
>>> dataset = dataset.bucket_by_sequence_length(
...         element_length_func=lambda elem: tf.shape(elem)[0],
...         bucket_boundaries=[3, 5],
...         bucket_batch_sizes=[2, 2, 2])
>>> for elem in dataset.as_numpy_iterator():
...   print(elem)
[[1 2 3 4]
[5 6 7 0]]
[[ 7  8  9 10 11  0]
[13 14 15 16 19 20]]
[[ 0  0]
[21 22]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`element_length_func`
</td>
<td>
function from element in `Dataset` to `tf.int32`,
determines the length of the element, which will determine the bucket it
goes into.
</td>
</tr><tr>
<td>
`bucket_boundaries`
</td>
<td>
`list<int>`, upper length boundaries of the buckets.
</td>
</tr><tr>
<td>
`bucket_batch_sizes`
</td>
<td>
`list<int>`, batch size per bucket. Length should be
`len(bucket_boundaries) + 1`.
</td>
</tr><tr>
<td>
`padded_shapes`
</td>
<td>
Nested structure of `tf.TensorShape` to pass to
`tf.data.Dataset.padded_batch`. If not provided, will use
`dataset.output_shapes`, which will result in variable length dimensions
being padded out to the maximum length in each batch.
</td>
</tr><tr>
<td>
`padding_values`
</td>
<td>
Values to pad with, passed to
`tf.data.Dataset.padded_batch`. Defaults to padding with 0.
</td>
</tr><tr>
<td>
`pad_to_bucket_boundary`
</td>
<td>
bool, if `False`, will pad dimensions with unknown
size to maximum length in batch. If `True`, will pad dimensions with
unknown size to bucket boundary minus 1 (i.e., the maximum length in
each bucket), and caller must ensure that the source `Dataset` does not
contain any elements with length longer than `max(bucket_boundaries)`.
</td>
</tr><tr>
<td>
`no_padding`
</td>
<td>
`bool`, indicates whether to pad the batch features (features
need to be either of type `tf.sparse.SparseTensor` or of same shape).
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
(Optional.) A `tf.bool` scalar `tf.Tensor`, representing
whether the last batch should be dropped in the case it has fewer than
`batch_size` elements; the default behavior is not to drop the smaller
batch.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Dataset` transformation function, which can be passed to
`tf.data.Dataset.apply`.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
</td>
</tr>
</table>

<h3 id="cache"><code>cache</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cache(
    filename=&#x27;&#x27;
)
</code></pre>

Caches the elements in this dataset.

The first time the dataset is iterated over, its elements will be cached either
in the specified file or in memory. Subsequent iterations will use the cached
data.

Note: For the cache to be finalized, the input dataset must be iterated through
in its entirety. Otherwise, subsequent iterations will not use cached data.

```
>>> dataset = tf.data.Dataset.range(5)
>>> dataset = dataset.map(lambda x: x**2)
>>> dataset = dataset.cache()
>>> # The first time reading through the data will generate the data using
>>> # `range` and `map`.
>>> list(dataset.as_numpy_iterator())
[0, 1, 4, 9, 16]
>>> # Subsequent iterations read from the cache.
>>> list(dataset.as_numpy_iterator())
[0, 1, 4, 9, 16]
```

When caching to a file, the cached data will persist across runs. Even the first
iteration through the data will read from the cache file. Changing the input
pipeline before the call to `.cache()` will have no effect until the cache file
is removed or the filename is changed.

```python
dataset = tf.data.Dataset.range(5)
dataset = dataset.cache("/path/to/file")
list(dataset.as_numpy_iterator())
# [0, 1, 2, 3, 4]
dataset = tf.data.Dataset.range(10)
dataset = dataset.cache("/path/to/file")  # Same file!
list(dataset.as_numpy_iterator())
# [0, 1, 2, 3, 4]
```

Note: `cache` will produce exactly the same elements during each iteration
through the dataset. If you wish to randomize the iteration order, make sure to
call `shuffle` *after* calling `cache`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`filename`
</td>
<td>
A `tf.string` scalar `tf.Tensor`, representing the name of a
directory on the filesystem to use for caching elements in this Dataset.
If a filename is not provided, the dataset will be cached in memory.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="cardinality"><code>cardinality</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cardinality()
</code></pre>

Returns the cardinality of the dataset, if known.

`cardinality` may return `tf.data.INFINITE_CARDINALITY` if the dataset contains
an infinite number of elements or `tf.data.UNKNOWN_CARDINALITY` if the analysis
fails to determine the number of elements in the dataset (e.g. when the dataset
source is a file).

```
>>> dataset = tf.data.Dataset.range(42)
>>> print(dataset.cardinality().numpy())
42
>>> dataset = dataset.repeat()
>>> cardinality = dataset.cardinality()
>>> print((cardinality == tf.data.INFINITE_CARDINALITY).numpy())
True
>>> dataset = dataset.filter(lambda x: True)
>>> cardinality = dataset.cardinality()
>>> print((cardinality == tf.data.UNKNOWN_CARDINALITY).numpy())
True
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A scalar `tf.int64` `Tensor` representing the cardinality of the dataset.
If the cardinality is infinite or unknown, `cardinality` returns the
named constants `tf.data.INFINITE_CARDINALITY` and
`tf.data.UNKNOWN_CARDINALITY` respectively.
</td>
</tr>

</table>

<h3 id="concatenate"><code>concatenate</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>concatenate(
    dataset
)
</code></pre>

Creates a `Dataset` by concatenating the given dataset with this dataset.

```
>>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
>>> b = tf.data.Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]
>>> ds = a.concatenate(b)
>>> list(ds.as_numpy_iterator())
[1, 2, 3, 4, 5, 6, 7]
>>> # The input dataset and dataset to be concatenated should have
>>> # compatible element specs.
>>> c = tf.data.Dataset.zip((a, b))
>>> a.concatenate(c)
Traceback (most recent call last):
TypeError: Two datasets to concatenate have different types
<dtype: 'int64'> and (tf.int64, tf.int64)
>>> d = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
>>> a.concatenate(d)
Traceback (most recent call last):
TypeError: Two datasets to concatenate have different types
<dtype: 'int64'> and <dtype: 'string'>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`dataset`
</td>
<td>
`Dataset` to be concatenated.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="enumerate"><code>enumerate</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>enumerate(
    start=0
)
</code></pre>

Enumerates the elements of this dataset.

It is similar to python's `enumerate`.

```
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset = dataset.enumerate(start=5)
>>> for element in dataset.as_numpy_iterator():
...   print(element)
(5, 1)
(6, 2)
(7, 3)
```

```
>>> # The (nested) structure of the input dataset determines the
>>> # structure of elements in the resulting dataset.
>>> dataset = tf.data.Dataset.from_tensor_slices([(7, 8), (9, 10)])
>>> dataset = dataset.enumerate()
>>> for element in dataset.as_numpy_iterator():
...   print(element)
(0, array([7, 8], dtype=int32))
(1, array([ 9, 10], dtype=int32))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`start`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the start value for
enumeration.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="filter"><code>filter</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>filter(
    predicate
)
</code></pre>

Filters this dataset according to `predicate`.

```
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset = dataset.filter(lambda x: x < 3)
>>> list(dataset.as_numpy_iterator())
[1, 2]
>>> # `tf.math.equal(x, y)` is required for equality comparison
>>> def filter_fn(x):
...   return tf.math.equal(x, 1)
>>> dataset = dataset.filter(filter_fn)
>>> list(dataset.as_numpy_iterator())
[1]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`predicate`
</td>
<td>
A function mapping a dataset element to a boolean.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
The `Dataset` containing the elements of this dataset for which
`predicate` is `True`.
</td>
</tr>
</table>

<h3 id="flat_map"><code>flat_map</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flat_map(
    map_func
)
</code></pre>

Maps `map_func` across this dataset and flattens the result.

Use `flat_map` if you want to make sure that the order of your dataset stays the
same. For example, to flatten a dataset of batches into a dataset of their
elements:

```
>>> dataset = tf.data.Dataset.from_tensor_slices(
...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> dataset = dataset.flat_map(
...     lambda x: tf.data.Dataset.from_tensor_slices(x))
>>> list(dataset.as_numpy_iterator())
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

`tf.data.Dataset.interleave()` is a generalization of `flat_map`, since
`flat_map` produces the same output as
`tf.data.Dataset.interleave(cycle_length=1)`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`map_func`
</td>
<td>
A function mapping a dataset element to a dataset.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="from_generator"><code>from_generator</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_generator(
    generator, output_types=None, output_shapes=None, args=None,
    output_signature=None
)
</code></pre>

Creates a `Dataset` whose elements are generated by `generator`. (deprecated
arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(output_shapes, output_types)`. They
will be removed in a future version. Instructions for updating: Use
output_signature instead

The `generator` argument must be a callable object that returns an object that
supports the `iter()` protocol (e.g. a generator function).

The elements generated by `generator` must be compatible with either the given
`output_signature` argument or with the given `output_types` and (optionally)
`output_shapes` arguments, whichever was specified.

The recommended way to call `from_generator` is to use the `output_signature`
argument. In this case the output will be assumed to consist of objects with the
classes, shapes and types defined by `tf.TypeSpec` objects from
`output_signature` argument:

```
>>> def gen():
...   ragged_tensor = tf.ragged.constant([[1, 2], [3]])
...   yield 42, ragged_tensor
>>>
>>> dataset = tf.data.Dataset.from_generator(
...      gen,
...      output_signature=(
...          tf.TensorSpec(shape=(), dtype=tf.int32),
...          tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))
>>>
>>> list(dataset.take(1))
[(<tf.Tensor: shape=(), dtype=int32, numpy=42>,
<tf.RaggedTensor [[1, 2], [3]]>)]
```

There is also a deprecated way to call `from_generator` by either with
`output_types` argument alone or together with `output_shapes` argument. In this
case the output of the function will be assumed to consist of `tf.Tensor`
objects with the types defined by `output_types` and with the shapes which are
either unknown or defined by `output_shapes`.

Note: The current implementation of `Dataset.from_generator()` uses
`tf.numpy_function` and inherits the same constraints. In particular, it
requires the dataset and iterator related operations to be placed on a device in
the same process as the Python program that called `Dataset.from_generator()`.
The body of `generator` will not be serialized in a `GraphDef`, and you should
not use this method if you need to serialize your model and restore it in a
different environment.

Note: If `generator` depends on mutable global variables or other external
state, be aware that the runtime may invoke `generator` multiple times (in order
to support repeating the `Dataset`) and at any time between the call to
`Dataset.from_generator()` and the production of the first element from the
generator. Mutating global variables or external state can cause undefined
behavior, and we recommend that you explicitly cache any external state in
`generator` before calling `Dataset.from_generator()`.

Note: While the `output_signature` parameter makes it possible to yield
`Dataset` elements, the scope of `Dataset.from_generator()` should be limited to
logic that cannot be expressed through tf.data operations. Using tf.data
operations within the generator function is an anti-pattern and may result in
incremental memory growth.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`generator`
</td>
<td>
A callable object that returns an object that supports the
`iter()` protocol. If `args` is not specified, `generator` must take no
arguments; otherwise it must take as many arguments as there are values
in `args`.
</td>
</tr><tr>
<td>
`output_types`
</td>
<td>
(Optional.) A (nested) structure of `tf.DType` objects
corresponding to each component of an element yielded by `generator`.
</td>
</tr><tr>
<td>
`output_shapes`
</td>
<td>
(Optional.) A (nested) structure of `tf.TensorShape`
objects corresponding to each component of an element yielded by
`generator`.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
(Optional.) A tuple of `tf.Tensor` objects that will be evaluated
and passed to `generator` as NumPy-array arguments.
</td>
</tr><tr>
<td>
`output_signature`
</td>
<td>
(Optional.) A (nested) structure of `tf.TypeSpec`
objects corresponding to each component of an element yielded by
`generator`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="from_tensor_slices"><code>from_tensor_slices</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_tensor_slices(
    tensors
)
</code></pre>

Creates a `Dataset` whose elements are slices of the given tensors.

The given tensors are sliced along their first dimension. This operation
preserves the structure of the input tensors, removing the first dimension of
each tensor and using it as the dataset dimension. All input tensors must have
the same size in their first dimensions.

```
>>> # Slicing a 1D tensor produces scalar tensor elements.
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> list(dataset.as_numpy_iterator())
[1, 2, 3]
```

```
>>> # Slicing a 2D tensor produces 1D tensor elements.
>>> dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
>>> list(dataset.as_numpy_iterator())
[array([1, 2], dtype=int32), array([3, 4], dtype=int32)]
```

```
>>> # Slicing a tuple of 1D tensors produces tuple elements containing
>>> # scalar tensors.
>>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
>>> list(dataset.as_numpy_iterator())
[(1, 3, 5), (2, 4, 6)]
```

```
>>> # Dictionary structure is also preserved.
>>> dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
>>> list(dataset.as_numpy_iterator()) == [{'a': 1, 'b': 3},
...                                       {'a': 2, 'b': 4}]
True
```

```
>>> # Two tensors can be combined into one Dataset object.
>>> features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
>>> labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
>>> dataset = Dataset.from_tensor_slices((features, labels))
>>> # Both the features and the labels tensors can be converted
>>> # to a Dataset object separately and combined after.
>>> features_dataset = Dataset.from_tensor_slices(features)
>>> labels_dataset = Dataset.from_tensor_slices(labels)
>>> dataset = Dataset.zip((features_dataset, labels_dataset))
>>> # A batched feature and label set can be converted to a Dataset
>>> # in similar fashion.
>>> batched_features = tf.constant([[[1, 3], [2, 3]],
...                                 [[2, 1], [1, 2]],
...                                 [[3, 3], [3, 2]]], shape=(3, 2, 2))
>>> batched_labels = tf.constant([['A', 'A'],
...                               ['B', 'B'],
...                               ['A', 'B']], shape=(3, 2, 1))
>>> dataset = Dataset.from_tensor_slices((batched_features, batched_labels))
>>> for element in dataset.as_numpy_iterator():
...   print(element)
(array([[1, 3],
       [2, 3]], dtype=int32), array([[b'A'],
       [b'A']], dtype=object))
(array([[2, 1],
       [1, 2]], dtype=int32), array([[b'B'],
       [b'B']], dtype=object))
(array([[3, 3],
       [3, 2]], dtype=int32), array([[b'A'],
       [b'B']], dtype=object))
```

Note that if `tensors` contains a NumPy array, and eager execution is not
enabled, the values will be embedded in the graph as one or more `tf.constant`
operations. For large datasets (> 1 GB), this can waste memory and run into byte
limits of graph serialization. If `tensors` contains one or more large NumPy
arrays, consider the alternative described in
[this guide](https://tensorflow.org/guide/data#consuming_numpy_arrays).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensors`
</td>
<td>
A dataset element, whose components have the same first
dimension. Supported values are documented
[here](https://www.tensorflow.org/guide/data#dataset_structure).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="from_tensors"><code>from_tensors</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_tensors(
    tensors
)
</code></pre>

Creates a `Dataset` with a single element, comprising the given tensors.

`from_tensors` produces a dataset containing only a single element. To slice the
input tensor into multiple elements, use `from_tensor_slices` instead.

```
>>> dataset = tf.data.Dataset.from_tensors([1, 2, 3])
>>> list(dataset.as_numpy_iterator())
[array([1, 2, 3], dtype=int32)]
>>> dataset = tf.data.Dataset.from_tensors(([1, 2, 3], 'A'))
>>> list(dataset.as_numpy_iterator())
[(array([1, 2, 3], dtype=int32), b'A')]
```

```
>>> # You can use `from_tensors` to produce a dataset which repeats
>>> # the same example many times.
>>> example = tf.constant([1,2,3])
>>> dataset = tf.data.Dataset.from_tensors(example).repeat(2)
>>> list(dataset.as_numpy_iterator())
[array([1, 2, 3], dtype=int32), array([1, 2, 3], dtype=int32)]
```

Note that if `tensors` contains a NumPy array, and eager execution is not
enabled, the values will be embedded in the graph as one or more `tf.constant`
operations. For large datasets (> 1 GB), this can waste memory and run into byte
limits of graph serialization. If `tensors` contains one or more large NumPy
arrays, consider the alternative described in
[this guide](https://tensorflow.org/guide/data#consuming_numpy_arrays).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensors`
</td>
<td>
A dataset "element". Supported values are documented
[here](https://www.tensorflow.org/guide/data#dataset_structure).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="get_single_element"><code>get_single_element</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_single_element()
</code></pre>

Returns the single element of the `dataset` as a nested structure of tensors.

The function enables you to use a `tf.data.Dataset` in a stateless "tensor-in
tensor-out" expression, without creating an iterator. This facilitates the ease
of data transformation on tensors using the optimized `tf.data.Dataset`
abstraction on top of them.

For example, lets consider a `preprocessing_fn` which would take as an input the
raw features and returns the processed feature along with it's label.

```python
def preprocessing_fn(raw_feature):
  # ... the raw_feature is preprocessed as per the use-case
  return feature

raw_features = ...  # input batch of BATCH_SIZE elements.
dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
          .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
          .batch(BATCH_SIZE))

processed_features = dataset.get_single_element()
```

In the above example, the `raw_features` tensor of length=BATCH_SIZE was
converted to a `tf.data.Dataset`. Next, each of the `raw_feature` was mapped
using the `preprocessing_fn` and the processed features were grouped into a
single batch. The final `dataset` contains only one element which is a batch of
all the processed features.

NOTE: The `dataset` should contain only one element.

Now, instead of creating an iterator for the `dataset` and retrieving the batch
of features, the `tf.data.get_single_element()` function is used to skip the
iterator creation process and directly output the batch of features.

This can be particularly useful when your tensor transformations are expressed
as `tf.data.Dataset` operations, and you want to use those transformations while
serving your model.

# Keras

```python

model = ... # A pre-built or custom model

class PreprocessingModel(tf.keras.Model):
  def __init__(self, model):
    super().__init__(self)
    self.model = model

  @tf.function(input_signature=[...])
  def serving_fn(self, data):
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
    ds = ds.batch(batch_size=BATCH_SIZE)
    return tf.argmax(self.model(ds.get_single_element()), axis=-1)

preprocessing_model = PreprocessingModel(model)
your_exported_model_dir = ... # save the model to this path.
tf.saved_model.save(preprocessing_model, your_exported_model_dir,
              signatures={'serving_default': preprocessing_model.serving_fn}
              )
```

# Estimator

In the case of estimators, you need to generally define a `serving_input_fn`
which would require the features to be processed by the model while inferencing.

```python
def serving_input_fn():

  raw_feature_spec = ... # Spec for the raw_features
  input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  )
  serving_input_receiver = input_fn()
  raw_features = serving_input_receiver.features

  def preprocessing_fn(raw_feature):
    # ... the raw_feature is preprocessed as per the use-case
    return feature

  dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
            .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
            .batch(BATCH_SIZE))

  processed_features = dataset.get_single_element()

  # Please note that the value of `BATCH_SIZE` should be equal to
  # the size of the leading dimension of `raw_features`. This ensures
  # that `dataset` has only element, which is a pre-requisite for
  # using `dataset.get_single_element()`.

  return tf.estimator.export.ServingInputReceiver(
      processed_features, serving_input_receiver.receiver_tensors)

estimator = ... # A pre-built or custom estimator
estimator.export_saved_model(your_exported_model_dir, serving_input_fn)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A nested structure of `tf.Tensor` objects, corresponding to the single
element of `dataset`.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`InvalidArgumentError`
</td>
<td>
(at runtime) if `dataset` does not contain exactly
one element.
</td>
</tr>
</table>

<h3 id="group_by_window"><code>group_by_window</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>group_by_window(
    key_func, reduce_func, window_size=None, window_size_func=None
)
</code></pre>

Groups windows of elements by key and reduces them.

This transformation maps each consecutive element in a dataset to a key using
`key_func` and groups the elements by key. It then applies `reduce_func` to at
most `window_size_func(key)` elements matching the same key. All except the
final window for each key will contain `window_size_func(key)` elements; the
final window may be smaller.

You may provide either a constant `window_size` or a window size determined by
the key through `window_size_func`.

```
>>> dataset = tf.data.Dataset.range(10)
>>> window_size = 5
>>> key_func = lambda x: x%2
>>> reduce_func = lambda key, dataset: dataset.batch(window_size)
>>> dataset = dataset.group_by_window(
...           key_func=key_func,
...           reduce_func=reduce_func,
...           window_size=window_size)
>>> for elem in dataset.as_numpy_iterator():
...   print(elem)
[0 2 4 6 8]
[1 3 5 7 9]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`key_func`
</td>
<td>
A function mapping a nested structure of tensors (having shapes
and types defined by `self.output_shapes` and `self.output_types`) to a
scalar `tf.int64` tensor.
</td>
</tr><tr>
<td>
`reduce_func`
</td>
<td>
A function mapping a key and a dataset of up to `window_size`
consecutive elements matching that key to another dataset.
</td>
</tr><tr>
<td>
`window_size`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the number of
consecutive elements matching the same key to combine in a single batch,
which will be passed to `reduce_func`. Mutually exclusive with
`window_size_func`.
</td>
</tr><tr>
<td>
`window_size_func`
</td>
<td>
A function mapping a key to a `tf.int64` scalar
`tf.Tensor`, representing the number of consecutive elements matching
the same key to combine in a single batch, which will be passed to
`reduce_func`. Mutually exclusive with `window_size`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.data.Dataset`
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if neither or both of {`window_size`, `window_size_func`} are
passed.
</td>
</tr>
</table>

<h3 id="interleave"><code>interleave</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>interleave(
    map_func, cycle_length=None, block_length=None, num_parallel_calls=None,
    deterministic=None
)
</code></pre>

Maps `map_func` across this dataset, and interleaves the results.

For example, you can use `Dataset.interleave()` to process many input files
concurrently:

```
>>> # Preprocess 4 files concurrently, and interleave blocks of 16 records
>>> # from each file.
>>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",
...              "/var/data/file3.txt", "/var/data/file4.txt"]
>>> dataset = tf.data.Dataset.from_tensor_slices(filenames)
>>> def parse_fn(filename):
...   return tf.data.Dataset.range(10)
>>> dataset = dataset.interleave(lambda x:
...     tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
...     cycle_length=4, block_length=16)
```

The `cycle_length` and `block_length` arguments control the order in which
elements are produced. `cycle_length` controls the number of input elements that
are processed concurrently. If you set `cycle_length` to 1, this transformation
will handle one input element at a time, and will produce identical results to
`tf.data.Dataset.flat_map`. In general, this transformation will apply
`map_func` to `cycle_length` input elements, open iterators on the returned
`Dataset` objects, and cycle through them producing `block_length` consecutive
elements from each iterator, and consuming the next input element each time it
reaches the end of an iterator.

#### For example:

```
>>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
>>> # NOTE: New lines indicate "block" boundaries.
>>> dataset = dataset.interleave(
...     lambda x: Dataset.from_tensors(x).repeat(6),
...     cycle_length=2, block_length=4)
>>> list(dataset.as_numpy_iterator())
[1, 1, 1, 1,
 2, 2, 2, 2,
 1, 1,
 2, 2,
 3, 3, 3, 3,
 4, 4, 4, 4,
 3, 3,
 4, 4,
 5, 5, 5, 5,
 5, 5]
```

Note: The order of elements yielded by this transformation is deterministic, as
long as `map_func` is a pure function and `deterministic=True`. If `map_func`
contains any stateful operations, the order in which that state is accessed is
undefined.

Performance can often be improved by setting `num_parallel_calls` so that
`interleave` will use multiple threads to fetch elements. If determinism isn't
required, it can also improve performance to set `deterministic=False`.

```
>>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",
...              "/var/data/file3.txt", "/var/data/file4.txt"]
>>> dataset = tf.data.Dataset.from_tensor_slices(filenames)
>>> dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
...     cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE,
...     deterministic=False)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`map_func`
</td>
<td>
A function mapping a dataset element to a dataset.
</td>
</tr><tr>
<td>
`cycle_length`
</td>
<td>
(Optional.) The number of input elements that will be
processed concurrently. If not set, the tf.data runtime decides what it
should be based on available CPU. If `num_parallel_calls` is set to
`tf.data.AUTOTUNE`, the `cycle_length` argument identifies
the maximum degree of parallelism.
</td>
</tr><tr>
<td>
`block_length`
</td>
<td>
(Optional.) The number of consecutive elements to produce
from each input element before cycling to another input element. If not
set, defaults to 1.
</td>
</tr><tr>
<td>
`num_parallel_calls`
</td>
<td>
(Optional.) If specified, the implementation creates a
threadpool, which is used to fetch inputs from cycle elements
asynchronously and in parallel. The default behavior is to fetch inputs
from cycle elements synchronously with no parallelism. If the value
`tf.data.AUTOTUNE` is used, then the number of parallel
calls is set dynamically based on available CPU.
</td>
</tr><tr>
<td>
`deterministic`
</td>
<td>
(Optional.) When `num_parallel_calls` is specified, if this
boolean is specified (`True` or `False`), it controls the order in which
the transformation produces elements. If set to `False`, the
transformation is allowed to yield elements out of order to trade
determinism for performance. If not specified, the
`tf.data.Options.experimental_deterministic` option
(`True` by default) controls the behavior.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="list_files"><code>list_files</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>list_files(
    file_pattern, shuffle=None, seed=None
)
</code></pre>

A dataset of all files matching one or more glob patterns.

The `file_pattern` argument should be a small number of glob patterns. If your
filenames have already been globbed, use `Dataset.from_tensor_slices(filenames)`
instead, as re-globbing every filename with `list_files` may result in poor
performance with remote storage systems.

Note: The default behavior of this method is to return filenames in a
non-deterministic random shuffled order. Pass a `seed` or `shuffle=False` to get
results in a deterministic order.

#### Example:

If we had the following files on our filesystem:

-   /path/to/dir/a.txt
-   /path/to/dir/b.py
-   /path/to/dir/c.py

If we pass "/path/to/dir/*.py" as the directory, the dataset would produce:

-   /path/to/dir/b.py
-   /path/to/dir/c.py

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_pattern`
</td>
<td>
A string, a list of strings, or a `tf.Tensor` of string type
(scalar or vector), representing the filename glob (i.e. shell wildcard)
pattern(s) that will be matched.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
(Optional.) If `True`, the file names will be shuffled randomly.
Defaults to `True`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
(Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
seed that will be used to create the distribution. See
`tf.random.set_seed` for behavior.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset` of strings corresponding to file names.
</td>
</tr>
</table>

<h3 id="map"><code>map</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map(
    map_func, num_parallel_calls=None, deterministic=None
)
</code></pre>

Maps `map_func` across the elements of this dataset.

This transformation applies `map_func` to each element of this dataset, and
returns a new dataset containing the transformed elements, in the same order as
they appeared in the input. `map_func` can be used to change both the values and
the structure of a dataset's elements. Supported structure constructs are
documented [here](https://www.tensorflow.org/guide/data#dataset_structure).

For example, `map` can be used for adding 1 to each element, or projecting a
subset of element components.

```
>>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
>>> dataset = dataset.map(lambda x: x + 1)
>>> list(dataset.as_numpy_iterator())
[2, 3, 4, 5, 6]
```

The input signature of `map_func` is determined by the structure of each element
in this dataset.

```
>>> dataset = Dataset.range(5)
>>> # `map_func` takes a single argument of type `tf.Tensor` with the same
>>> # shape and dtype.
>>> result = dataset.map(lambda x: x + 1)
```

```
>>> # Each element is a tuple containing two `tf.Tensor` objects.
>>> elements = [(1, "foo"), (2, "bar"), (3, "baz")]
>>> dataset = tf.data.Dataset.from_generator(
...     lambda: elements, (tf.int32, tf.string))
>>> # `map_func` takes two arguments of type `tf.Tensor`. This function
>>> # projects out just the first component.
>>> result = dataset.map(lambda x_int, y_str: x_int)
>>> list(result.as_numpy_iterator())
[1, 2, 3]
```

```
>>> # Each element is a dictionary mapping strings to `tf.Tensor` objects.
>>> elements =  ([{"a": 1, "b": "foo"},
...               {"a": 2, "b": "bar"},
...               {"a": 3, "b": "baz"}])
>>> dataset = tf.data.Dataset.from_generator(
...     lambda: elements, {"a": tf.int32, "b": tf.string})
>>> # `map_func` takes a single argument of type `dict` with the same keys
>>> # as the elements.
>>> result = dataset.map(lambda d: str(d["a"]) + d["b"])
```

The value or values returned by `map_func` determine the structure of each
element in the returned dataset.

```
>>> dataset = tf.data.Dataset.range(3)
>>> # `map_func` returns two `tf.Tensor` objects.
>>> def g(x):
...   return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])
>>> result = dataset.map(g)
>>> result.element_spec
(TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(3,), dtype=tf.string, name=None))
>>> # Python primitives, lists, and NumPy arrays are implicitly converted to
>>> # `tf.Tensor`.
>>> def h(x):
...   return 37.0, ["Foo", "Bar"], np.array([1.0, 2.0], dtype=np.float64)
>>> result = dataset.map(h)
>>> result.element_spec
(TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(2,), dtype=tf.string, name=None), TensorSpec(shape=(2,), dtype=tf.float64, name=None))
>>> # `map_func` can return nested structures.
>>> def i(x):
...   return (37.0, [42, 16]), "foo"
>>> result = dataset.map(i)
>>> result.element_spec
((TensorSpec(shape=(), dtype=tf.float32, name=None),
  TensorSpec(shape=(2,), dtype=tf.int32, name=None)),
 TensorSpec(shape=(), dtype=tf.string, name=None))
```

`map_func` can accept as arguments and return any type of dataset element.

Note that irrespective of the context in which `map_func` is defined (eager vs.
graph), tf.data traces the function and executes it as a graph. To use Python
code inside of the function you have a few options:

1) Rely on AutoGraph to convert Python code into an equivalent graph
computation. The downside of this approach is that AutoGraph can convert some
but not all Python code.

2) Use `tf.py_function`, which allows you to write arbitrary Python code but
will generally result in worse performance than 1). For example:

```
>>> d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
>>> # transform a string tensor to upper case string using a Python function
>>> def upper_case_fn(t: tf.Tensor):
...   return t.numpy().decode('utf-8').upper()
>>> d = d.map(lambda x: tf.py_function(func=upper_case_fn,
...           inp=[x], Tout=tf.string))
>>> list(d.as_numpy_iterator())
[b'HELLO', b'WORLD']
```

3) Use `tf.numpy_function`, which also allows you to write arbitrary Python
code. Note that `tf.py_function` accepts `tf.Tensor` whereas `tf.numpy_function`
accepts numpy arrays and returns only numpy arrays. For example:

```
>>> d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
>>> def upper_case_fn(t: np.ndarray):
...   return t.decode('utf-8').upper()
>>> d = d.map(lambda x: tf.numpy_function(func=upper_case_fn,
...           inp=[x], Tout=tf.string))
>>> list(d.as_numpy_iterator())
[b'HELLO', b'WORLD']
```

Note that the use of `tf.numpy_function` and `tf.py_function` in general
precludes the possibility of executing user-defined transformations in parallel
(because of Python GIL).

Performance can often be improved by setting `num_parallel_calls` so that `map`
will use multiple threads to process elements. If deterministic order isn't
required, it can also improve performance to set `deterministic=False`.

```
>>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
>>> dataset = dataset.map(lambda x: x + 1,
...     num_parallel_calls=tf.data.AUTOTUNE,
...     deterministic=False)
```

The order of elements yielded by this transformation is deterministic if
`deterministic=True`. If `map_func` contains stateful operations and
`num_parallel_calls > 1`, the order in which that state is accessed is
undefined, so the values of output elements may not be deterministic regardless
of the `deterministic` flag value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`map_func`
</td>
<td>
A function mapping a dataset element to another dataset element.
</td>
</tr><tr>
<td>
`num_parallel_calls`
</td>
<td>
(Optional.) A `tf.int64` scalar `tf.Tensor`,
representing the number elements to process asynchronously in parallel.
If not specified, elements will be processed sequentially. If the value
`tf.data.AUTOTUNE` is used, then the number of parallel
calls is set dynamically based on available CPU.
</td>
</tr><tr>
<td>
`deterministic`
</td>
<td>
(Optional.) When `num_parallel_calls` is specified, if this
boolean is specified (`True` or `False`), it controls the order in which
the transformation produces elements. If set to `False`, the
transformation is allowed to yield elements out of order to trade
determinism for performance. If not specified, the
`tf.data.Options.experimental_deterministic` option
(`True` by default) controls the behavior.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="options"><code>options</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>options()
</code></pre>

Returns the options for this dataset and its inputs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.data.Options` object representing the dataset options.
</td>
</tr>

</table>

<h3 id="padded_batch"><code>padded_batch</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>padded_batch(
    batch_size, padded_shapes=None, padding_values=None, drop_remainder=False
)
</code></pre>

Combines consecutive elements of this dataset into padded batches.

This transformation combines multiple consecutive elements of the input dataset
into a single element.

Like `tf.data.Dataset.batch`, the components of the resulting element will have
an additional outer dimension, which will be `batch_size` (or `N % batch_size`
for the last element if `batch_size` does not divide the number of input
elements `N` evenly and `drop_remainder` is `False`). If your program depends on
the batches having the same outer dimension, you should set the `drop_remainder`
argument to `True` to prevent the smaller batch from being produced.

Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
different shapes, and this transformation will pad each component to the
respective shape in `padded_shapes`. The `padded_shapes` argument determines the
resulting shape for each dimension of each component in an output element:

*   If the dimension is a constant, the component will be padded out to that
    length in that dimension.
*   If the dimension is unknown, the component will be padded out to the maximum
    length of all elements in that dimension.

```
>>> A = (tf.data.Dataset
...      .range(1, 5, output_type=tf.int32)
...      .map(lambda x: tf.fill([x], x)))
>>> # Pad to the smallest per-batch size that fits all elements.
>>> B = A.padded_batch(2)
>>> for element in B.as_numpy_iterator():
...   print(element)
[[1 0]
 [2 2]]
[[3 3 3 0]
 [4 4 4 4]]
>>> # Pad to a fixed size.
>>> C = A.padded_batch(2, padded_shapes=5)
>>> for element in C.as_numpy_iterator():
...   print(element)
[[1 0 0 0 0]
 [2 2 0 0 0]]
[[3 3 3 0 0]
 [4 4 4 4 0]]
>>> # Pad with a custom value.
>>> D = A.padded_batch(2, padded_shapes=5, padding_values=-1)
>>> for element in D.as_numpy_iterator():
...   print(element)
[[ 1 -1 -1 -1 -1]
 [ 2  2 -1 -1 -1]]
[[ 3  3  3 -1 -1]
 [ 4  4  4  4 -1]]
>>> # Components of nested elements can be padded independently.
>>> elements = [([1, 2, 3], [10]),
...             ([4, 5], [11, 12])]
>>> dataset = tf.data.Dataset.from_generator(
...     lambda: iter(elements), (tf.int32, tf.int32))
>>> # Pad the first component of the tuple to length 4, and the second
>>> # component to the smallest size that fits.
>>> dataset = dataset.padded_batch(2,
...     padded_shapes=([4], [None]),
...     padding_values=(-1, 100))
>>> list(dataset.as_numpy_iterator())
[(array([[ 1,  2,  3, -1], [ 4,  5, -1, -1]], dtype=int32),
  array([[ 10, 100], [ 11,  12]], dtype=int32))]
>>> # Pad with a single value and multiple components.
>>> E = tf.data.Dataset.zip((A, A)).padded_batch(2, padding_values=-1)
>>> for element in E.as_numpy_iterator():
...   print(element)
(array([[ 1, -1],
       [ 2,  2]], dtype=int32), array([[ 1, -1],
       [ 2,  2]], dtype=int32))
(array([[ 3,  3,  3, -1],
       [ 4,  4,  4,  4]], dtype=int32), array([[ 3,  3,  3, -1],
       [ 4,  4,  4,  4]], dtype=int32))
```

See also `tf.data.experimental.dense_to_sparse_batch`, which combines elements
that may have different shapes into a `tf.sparse.SparseTensor`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch_size`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the number of
consecutive elements of this dataset to combine in a single batch.
</td>
</tr><tr>
<td>
`padded_shapes`
</td>
<td>
(Optional.) A (nested) structure of `tf.TensorShape` or
`tf.int64` vector tensor-like objects representing the shape to which
the respective component of each input element should be padded prior
to batching. Any unknown dimensions will be padded to the maximum size
of that dimension in each batch. If unset, all dimensions of all
components are padded to the maximum size in the batch. `padded_shapes`
must be set if any component has an unknown rank.
</td>
</tr><tr>
<td>
`padding_values`
</td>
<td>
(Optional.) A (nested) structure of scalar-shaped
`tf.Tensor`, representing the padding values to use for the respective
components. None represents that the (nested) structure should be padded
with default values.  Defaults are `0` for numeric types and the empty
string for string types. The `padding_values` should have the same
(nested) structure as the input dataset. If `padding_values` is a single
element and the input dataset has multiple components, then the same
`padding_values` will be used to pad every component of the dataset.
If `padding_values` is a scalar, then its value will be broadcasted
to match the shape of each component.
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
(Optional.) A `tf.bool` scalar `tf.Tensor`, representing
whether the last batch should be dropped in the case it has fewer than
`batch_size` elements; the default behavior is not to drop the smaller
batch.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If a component has an unknown rank, and  the `padded_shapes`
argument is not set.
</td>
</tr>
</table>

<h3 id="prefetch"><code>prefetch</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>prefetch(
    buffer_size
)
</code></pre>

Creates a `Dataset` that prefetches elements from this dataset.

Most dataset input pipelines should end with a call to `prefetch`. This allows
later elements to be prepared while the current element is being processed. This
often improves latency and throughput, at the cost of using additional memory to
store prefetched elements.

Note: Like other `Dataset` methods, prefetch operates on the elements of the
input dataset. It has no concept of examples vs. batches. `examples.prefetch(2)`
will prefetch two elements (2 examples), while `examples.batch(20).prefetch(2)`
will prefetch 2 elements (2 batches, of 20 examples each).

```
>>> dataset = tf.data.Dataset.range(3)
>>> dataset = dataset.prefetch(2)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`buffer_size`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the maximum
number of elements that will be buffered when prefetching. If the value
`tf.data.AUTOTUNE` is used, then the buffer size is dynamically tuned.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="random"><code>random</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>random(
    seed=None
)
</code></pre>

Creates a `Dataset` of pseudorandom values.

The dataset generates a sequence of uniformly distributed integer values.

```
>>> ds1 = tf.data.Dataset.random(seed=4).take(10)
>>> ds2 = tf.data.Dataset.random(seed=4).take(10)
>>> print(list(ds2.as_numpy_iterator())==list(ds2.as_numpy_iterator()))
True
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`seed`
</td>
<td>
(Optional) If specified, the dataset produces a deterministic
sequence of values.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="range"><code>range</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>range(
    *args, **kwargs
)
</code></pre>

Creates a `Dataset` of a step-separated range of values.

```
>>> list(Dataset.range(5).as_numpy_iterator())
[0, 1, 2, 3, 4]
>>> list(Dataset.range(2, 5).as_numpy_iterator())
[2, 3, 4]
>>> list(Dataset.range(1, 5, 2).as_numpy_iterator())
[1, 3]
>>> list(Dataset.range(1, 5, -2).as_numpy_iterator())
[]
>>> list(Dataset.range(5, 1).as_numpy_iterator())
[]
>>> list(Dataset.range(5, 1, -2).as_numpy_iterator())
[5, 3]
>>> list(Dataset.range(2, 5, output_type=tf.int32).as_numpy_iterator())
[2, 3, 4]
>>> list(Dataset.range(1, 5, 2, output_type=tf.float32).as_numpy_iterator())
[1.0, 3.0]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
follows the same semantics as python's xrange.
len(args) == 1 -> start = 0, stop = args[0], step = 1.
len(args) == 2 -> start = args[0], stop = args[1], step = 1.
len(args) == 3 -> start = args[0], stop = args[1], step = args[2].
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
- output_type: Its expected dtype. (Optional, default: `tf.int64`).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `RangeDataset`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if len(args) == 0.
</td>
</tr>
</table>

<h3 id="reduce"><code>reduce</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reduce(
    initial_state, reduce_func
)
</code></pre>

Reduces the input dataset to a single element.

The transformation calls `reduce_func` successively on every element of the
input dataset until the dataset is exhausted, aggregating information in its
internal state. The `initial_state` argument is used for the initial state and
the final state is returned as the result.

```
>>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1).numpy()
5
>>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y).numpy()
10
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`initial_state`
</td>
<td>
An element representing the initial state of the
transformation.
</td>
</tr><tr>
<td>
`reduce_func`
</td>
<td>
A function that maps `(old_state, input_element)` to
`new_state`. It must take two arguments and return a new element
The structure of `new_state` must match the structure of
`initial_state`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dataset element corresponding to the final state of the transformation.
</td>
</tr>

</table>

<h3 id="repeat"><code>repeat</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>repeat(
    count=None
)
</code></pre>

Repeats this dataset so each original value is seen `count` times.

```
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset = dataset.repeat(3)
>>> list(dataset.as_numpy_iterator())
[1, 2, 3, 1, 2, 3, 1, 2, 3]
```

Note: If this dataset is a function of global state (e.g. a random number
generator), then different repetitions may produce different elements.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`count`
</td>
<td>
(Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
number of times the dataset should be repeated. The default behavior (if
`count` is `None` or `-1`) is for the dataset be repeated indefinitely.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="shard"><code>shard</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>shard(
    num_shards, index
)
</code></pre>

Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

`shard` is deterministic. The Dataset produced by `A.shard(n, i)` will contain
all elements of A whose index mod n = i.

```
>>> A = tf.data.Dataset.range(10)
>>> B = A.shard(num_shards=3, index=0)
>>> list(B.as_numpy_iterator())
[0, 3, 6, 9]
>>> C = A.shard(num_shards=3, index=1)
>>> list(C.as_numpy_iterator())
[1, 4, 7]
>>> D = A.shard(num_shards=3, index=2)
>>> list(D.as_numpy_iterator())
[2, 5, 8]
```

This dataset operator is very useful when running distributed training, as it
allows each worker to read a unique subset.

When reading a single input file, you can shard elements as follows:

```python
d = tf.data.TFRecordDataset(input_file)
d = d.shard(num_workers, worker_index)
d = d.repeat(num_epochs)
d = d.shuffle(shuffle_buffer_size)
d = d.map(parser_fn, num_parallel_calls=num_map_threads)
```

#### Important caveats:

-   Be sure to shard before you use any randomizing operator (such as shuffle).
-   Generally it is best if the shard operator is used early in the dataset
    pipeline. For example, when reading from a set of TFRecord files, shard
    before converting the dataset to input samples. This avoids reading every
    file on every worker. The following is an example of an efficient sharding
    strategy within a complete pipeline:

```python
d = Dataset.list_files(pattern)
d = d.shard(num_workers, worker_index)
d = d.repeat(num_epochs)
d = d.shuffle(shuffle_buffer_size)
d = d.interleave(tf.data.TFRecordDataset,
                 cycle_length=num_readers, block_length=1)
d = d.map(parser_fn, num_parallel_calls=num_map_threads)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`num_shards`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the number of
shards operating in parallel.
</td>
</tr><tr>
<td>
`index`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the worker index.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr> <td> `InvalidArgumentError` </td> <td> if `num_shards` or `index` are
illegal values.

Note: error checking is done on a best-effort basis, and errors aren't
guaranteed to be caught upon dataset creation. (e.g. providing in a
placeholder tensor bypasses the early checking, and will instead result
in an error during a session.run call.)
</td>
</tr>
</table>

<h3 id="shuffle"><code>shuffle</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>shuffle(
    buffer_size, seed=None, reshuffle_each_iteration=None
)
</code></pre>

Randomly shuffles the elements of this dataset.

This dataset fills a buffer with `buffer_size` elements, then randomly samples
elements from this buffer, replacing the selected elements with new elements.
For perfect shuffling, a buffer size greater than or equal to the full size of
the dataset is required.

For instance, if your dataset contains 10,000 elements but `buffer_size` is set
to 1,000, then `shuffle` will initially select a random element from only the
first 1,000 elements in the buffer. Once an element is selected, its space in
the buffer is replaced by the next (i.e. 1,001-st) element, maintaining the
1,000 element buffer.

`reshuffle_each_iteration` controls whether the shuffle order should be
different for each epoch. In TF 1.X, the idiomatic way to create epochs was
through the `repeat` transformation:

```python
dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
dataset = dataset.repeat(2)
# [1, 0, 2, 1, 2, 0]

dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
dataset = dataset.repeat(2)
# [1, 0, 2, 1, 0, 2]
```

In TF 2.0, `tf.data.Dataset` objects are Python iterables which makes it
possible to also create epochs through Python iteration:

```python
dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
list(dataset.as_numpy_iterator())
# [1, 0, 2]
list(dataset.as_numpy_iterator())
# [1, 2, 0]
```

```python
dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
list(dataset.as_numpy_iterator())
# [1, 0, 2]
list(dataset.as_numpy_iterator())
# [1, 0, 2]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`buffer_size`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the number of
elements from this dataset from which the new dataset will sample.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
(Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
seed that will be used to create the distribution. See
`tf.random.set_seed` for behavior.
</td>
</tr><tr>
<td>
`reshuffle_each_iteration`
</td>
<td>
(Optional.) A boolean, which if true indicates
that the dataset should be pseudorandomly reshuffled each time it is
iterated over. (Defaults to `True`.)
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="skip"><code>skip</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>skip(
    count
)
</code></pre>

Creates a `Dataset` that skips `count` elements from this dataset.

```
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.skip(7)
>>> list(dataset.as_numpy_iterator())
[7, 8, 9]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`count`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the number of
elements of this dataset that should be skipped to form the new dataset.
If `count` is greater than the size of this dataset, the new dataset
will contain no elements.  If `count` is -1, skips the entire dataset.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="take"><code>take</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>take(
    count
)
</code></pre>

Creates a `Dataset` with at most `count` elements from this dataset.

```
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.take(3)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`count`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the number of
elements of this dataset that should be taken to form the new dataset.
If `count` is -1, or if `count` is greater than the size of this
dataset, the new dataset will contain all elements of this dataset.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="unbatch"><code>unbatch</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unbatch()
</code></pre>

Splits elements of a dataset into multiple elements.

For example, if elements of the dataset are shaped `[B, a0, a1, ...]`, where `B`
may vary for each input element, then for each element in the dataset, the
unbatched dataset will contain `B` consecutive elements of shape `[a0, a1,
...]`.

```
>>> elements = [ [1, 2, 3], [1, 2], [1, 2, 3, 4] ]
>>> dataset = tf.data.Dataset.from_generator(lambda: elements, tf.int64)
>>> dataset = dataset.unbatch()
>>> list(dataset.as_numpy_iterator())
[1, 2, 3, 1, 2, 1, 2, 3, 4]
```

Note: `unbatch` requires a data copy to slice up the batched tensor into
smaller, unbatched tensors. When optimizing performance, try to avoid
unnecessary usage of `unbatch`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Dataset`.
</td>
</tr>

</table>

<h3 id="window"><code>window</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>window(
    size, shift=None, stride=1, drop_remainder=False
)
</code></pre>

Combines (nests of) input elements into a dataset of (nests of) windows.

A "window" is a finite dataset of flat elements of size `size` (or possibly
fewer if there are not enough input elements to fill the window and
`drop_remainder` evaluates to `False`).

The `shift` argument determines the number of input elements by which the window
moves on each iteration. If windows and elements are both numbered starting at
0, the first element in window `k` will be element `k * shift` of the input
dataset. In particular, the first element of the first window will always be the
first element of the input dataset.

The `stride` argument determines the stride of the input elements, and the
`shift` argument determines the shift of the window.

#### For example:

```
>>> dataset = tf.data.Dataset.range(7).window(2)
>>> for window in dataset:
...   print(list(window.as_numpy_iterator()))
[0, 1]
[2, 3]
[4, 5]
[6]
>>> dataset = tf.data.Dataset.range(7).window(3, 2, 1, True)
>>> for window in dataset:
...   print(list(window.as_numpy_iterator()))
[0, 1, 2]
[2, 3, 4]
[4, 5, 6]
>>> dataset = tf.data.Dataset.range(7).window(3, 1, 2, True)
>>> for window in dataset:
...   print(list(window.as_numpy_iterator()))
[0, 2, 4]
[1, 3, 5]
[2, 4, 6]
```

Note that when the `window` transformation is applied to a dataset of nested
elements, it produces a dataset of nested windows.

```
>>> nested = ([1, 2, 3, 4], [5, 6, 7, 8])
>>> dataset = tf.data.Dataset.from_tensor_slices(nested).window(2)
>>> for window in dataset:
...   def to_numpy(ds):
...     return list(ds.as_numpy_iterator())
...   print(tuple(to_numpy(component) for component in window))
([1, 2], [5, 6])
([3, 4], [7, 8])
```

```
>>> dataset = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3, 4]})
>>> dataset = dataset.window(2)
>>> for window in dataset:
...   def to_numpy(ds):
...     return list(ds.as_numpy_iterator())
...   print({'a': to_numpy(window['a'])})
{'a': [1, 2]}
{'a': [3, 4]}
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`size`
</td>
<td>
A `tf.int64` scalar `tf.Tensor`, representing the number of elements
of the input dataset to combine into a window. Must be positive.
</td>
</tr><tr>
<td>
`shift`
</td>
<td>
(Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
number of input elements by which the window moves in each iteration.
Defaults to `size`. Must be positive.
</td>
</tr><tr>
<td>
`stride`
</td>
<td>
(Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
stride of the input elements in the sliding window. Must be positive.
The default value of 1 means "retain every input element".
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
(Optional.) A `tf.bool` scalar `tf.Tensor`, representing
whether the last windows should be dropped if their size is smaller than
`size`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset` of (nests of) windows -- a finite datasets of flat
elements created from the (nests of) input elements.
</td>
</tr>
</table>

<h3 id="with_options"><code>with_options</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_options(
    options
)
</code></pre>

Returns a new `tf.data.Dataset` with the given options set.

The options are "global" in the sense they apply to the entire dataset. If
options are set multiple times, they are merged as long as different options do
not use different non-default values.

```
>>> ds = tf.data.Dataset.range(5)
>>> ds = ds.interleave(lambda x: tf.data.Dataset.range(5),
...                    cycle_length=3,
...                    num_parallel_calls=3)
>>> options = tf.data.Options()
>>> # This will make the interleave order non-deterministic.
>>> options.experimental_deterministic = False
>>> ds = ds.with_options(options)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
A `tf.data.Options` that identifies the options the use.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset` with the given options.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
when an option is set more than once to a non-default value
</td>
</tr>
</table>

<h3 id="zip"><code>zip</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>zip(
    datasets
)
</code></pre>

Creates a `Dataset` by zipping together the given datasets.

This method has similar semantics to the built-in `zip()` function in Python,
with the main difference being that the `datasets` argument can be a (nested)
structure of `Dataset` objects. The supported nesting mechanisms are documented
[here](https://www.tensorflow.org/guide/data#dataset_structure).

```
>>> # The nested structure of the `datasets` argument determines the
>>> # structure of elements in the resulting dataset.
>>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
>>> b = tf.data.Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
>>> ds = tf.data.Dataset.zip((a, b))
>>> list(ds.as_numpy_iterator())
[(1, 4), (2, 5), (3, 6)]
>>> ds = tf.data.Dataset.zip((b, a))
>>> list(ds.as_numpy_iterator())
[(4, 1), (5, 2), (6, 3)]
>>>
>>> # The `datasets` argument may contain an arbitrary number of datasets.
>>> c = tf.data.Dataset.range(7, 13).batch(2)  # ==> [ [7, 8],
...                                            #       [9, 10],
...                                            #       [11, 12] ]
>>> ds = tf.data.Dataset.zip((a, b, c))
>>> for element in ds.as_numpy_iterator():
...   print(element)
(1, 4, array([7, 8]))
(2, 5, array([ 9, 10]))
(3, 6, array([11, 12]))
>>>
>>> # The number of elements in the resulting dataset is the same as
>>> # the size of the smallest dataset in `datasets`.
>>> d = tf.data.Dataset.range(13, 15)  # ==> [ 13, 14 ]
>>> ds = tf.data.Dataset.zip((a, d))
>>> list(ds.as_numpy_iterator())
[(1, 13), (2, 14)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`datasets`
</td>
<td>
A (nested) structure of datasets.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`Dataset`
</td>
<td>
A `Dataset`.
</td>
</tr>
</table>

<h3 id="__bool__"><code>__bool__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>

<h3 id="__iter__"><code>__iter__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>

Creates an iterator for elements of this dataset.

The returned iterator implements the Python Iterator protocol.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An `tf.data.Iterator` for the elements of this dataset.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If not inside of tf.function and not executing eagerly.
</td>
</tr>
</table>

<h3 id="__len__"><code>__len__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>

Returns the length of the dataset if it is known and finite.

This method requires that you are running in eager mode, and that the length of
the dataset is known and non-infinite. When the length may be unknown or
infinite, or if you are running in graph mode, use `tf.data.Dataset.cardinality`
instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An integer representing the length of the dataset.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If the dataset length is unknown or infinite, or if eager
execution is not enabled.
</td>
</tr>
</table>

<h3 id="__nonzero__"><code>__nonzero__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__nonzero__()
</code></pre>
