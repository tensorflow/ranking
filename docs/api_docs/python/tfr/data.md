description: Input data parsing for tf-ranking library.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="EIE"/>
<meta itemprop="property" content="ELWC"/>
<meta itemprop="property" content="SEQ"/>
</div>

# Module: tfr.data

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Input data parsing for tf-ranking library.

Supports the following data formats: - tf.train.SequenceExample -
tf.train.Example in tf.train.Example.

## Functions

[`build_ranking_dataset(...)`](../tfr/data/build_ranking_dataset.md): Builds a
ranking tf.dataset with a standard data format.

[`build_ranking_dataset_with_parsing_fn(...)`](../tfr/data/build_ranking_dataset_with_parsing_fn.md):
Builds a ranking tf.dataset using the provided `parsing_fn`.

[`build_ranking_serving_input_receiver_fn(...)`](../tfr/data/build_ranking_serving_input_receiver_fn.md):
Returns a serving input receiver fn for a standard data format.

[`build_ranking_serving_input_receiver_fn_with_parsing_fn(...)`](../tfr/data/build_ranking_serving_input_receiver_fn_with_parsing_fn.md):
Returns a receiver function with the provided `parsing_fn`.

[`build_sequence_example_serving_input_receiver_fn(...)`](../tfr/data/build_sequence_example_serving_input_receiver_fn.md):
Creates a serving_input_receiver_fn for `SequenceExample` inputs.

[`build_tf_example_serving_input_receiver_fn(...)`](../tfr/data/build_tf_example_serving_input_receiver_fn.md):
Builds a serving input fn for `tensorflow.training.Example`.

[`make_parsing_fn(...)`](../tfr/data/make_parsing_fn.md): Returns a parsing fn
for a standard data format.

[`parse_from_example_in_example(...)`](../tfr/data/parse_from_example_in_example.md):
Parses an ExampleInExample batch to a feature map.

[`parse_from_example_list(...)`](../tfr/data/parse_from_example_list.md): Parses
an `ExampleListWithContext` batch to a feature map.

[`parse_from_sequence_example(...)`](../tfr/data/parse_from_sequence_example.md):
Parses SequenceExample to feature maps.

[`parse_from_tf_example(...)`](../tfr/data/parse_from_tf_example.md): Parse
function to convert `tf.train.Example` to feature maps.

[`read_batched_sequence_example_dataset(...)`](../tfr/data/read_batched_sequence_example_dataset.md):
Returns a `Dataset` of features from `SequenceExample`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
EIE<a id="EIE"></a>
</td>
<td>
`'example_in_example'`
</td>
</tr><tr>
<td>
ELWC<a id="ELWC"></a>
</td>
<td>
`'example_list_with_context'`
</td>
</tr><tr>
<td>
SEQ<a id="SEQ"></a>
</td>
<td>
`'sequence_example'`
</td>
</tr>
</table>
