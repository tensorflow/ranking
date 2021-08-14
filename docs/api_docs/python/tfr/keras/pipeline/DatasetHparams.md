description: Hyperparameters used in BaseDatasetBuilder.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.DatasetHparams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="dataset_reader"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convert_labels_to_binary"/>
<meta itemprop="property" content="list_size"/>
<meta itemprop="property" content="valid_list_size"/>
</div>

# tfr.keras.pipeline.DatasetHparams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L310-L338">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Hyperparameters used in `BaseDatasetBuilder`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.DatasetHparams(
    train_input_pattern: str,
    valid_input_pattern: str,
    train_batch_size: int,
    valid_batch_size: int,
    list_size: Optional[int] = None,
    valid_list_size: Optional[int] = None,
    dataset_reader: Any = tfr.keras.pipeline.DatasetHparams.dataset_reader,
    convert_labels_to_binary: bool = False
)
</code></pre>

<!-- Placeholder for "Used in" -->

Hyperparameters to be specified to create the dataset_builder.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`train_input_pattern`
</td>
<td>
A glob pattern to specify the paths to the input data
for training.
</td>
</tr><tr>
<td>
`valid_input_pattern`
</td>
<td>
A glob pattern to specify the paths to the input data
for validation.
</td>
</tr><tr>
<td>
`train_batch_size`
</td>
<td>
An integer to specify the batch size of training dataset.
</td>
</tr><tr>
<td>
`valid_batch_size`
</td>
<td>
An integer to specify the batch size of valid dataset.
</td>
</tr><tr>
<td>
`list_size`
</td>
<td>
An integer to specify the list size. When None, data will be
padded to the longest list in each batch.
</td>
</tr><tr>
<td>
`valid_list_size`
</td>
<td>
An integer to specify the list size in valid dataset. When
not specified, valid dataset uses the same list size as `list_size`.
</td>
</tr><tr>
<td>
`dataset_reader`
</td>
<td>
A function or class that can be called with a `filenames`
tensor and (optional) `reader_args` and returns a `Dataset`. Defaults to
`tf.data.TFRecordDataset`.
</td>
</tr><tr>
<td>
`convert_labels_to_binary`
</td>
<td>
A boolean to indicate whether to use binary label.
</td>
</tr>
</table>

## Child Classes
[`class dataset_reader`](../../../tfr/keras/pipeline/DatasetHparams/dataset_reader.md)

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
convert_labels_to_binary<a id="convert_labels_to_binary"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
list_size<a id="list_size"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
valid_list_size<a id="valid_list_size"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
