<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="EIE"/>
<meta itemprop="property" content="SEQ"/>
</div>

# Module: tfr.data

Input data parsing for tf-ranking library.

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

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

[`libsvm_generator(...)`](../tfr/data/libsvm_generator.md): Parses a
LibSVM-formatted input file and aggregates data points by qid.

[`make_parsing_fn(...)`](../tfr/data/make_parsing_fn.md): Returns a parsing fn
for a standard data format.

[`parse_from_example_in_example(...)`](../tfr/data/parse_from_example_in_example.md):
Parses an ExampleInExample batch to a feature map.

[`parse_from_sequence_example(...)`](../tfr/data/parse_from_sequence_example.md):
Parses SequenceExample to feature maps.

[`read_batched_sequence_example_dataset(...)`](../tfr/data/read_batched_sequence_example_dataset.md):
Returns a `Dataset` of features from `SequenceExample`.

## Other Members

*   `EIE = 'example_in_example'` <a id="EIE"></a>
*   `SEQ = 'sequence_example'` <a id="SEQ"></a>
