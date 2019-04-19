<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="ranking.data" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: ranking.data



Defined in [`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

Input data parsing for ranking library.

Supports data stored in SequenceExample proto format.

SequenceExample (`tf.SequenceExample`) is defined in:
tensorflow/core/example/example.proto<!-- Placeholder for "Used in" -->



## Functions

[`build_sequence_example_serving_input_receiver_fn(...)`](../ranking/data/build_sequence_example_serving_input_receiver_fn.md): Creates a serving_input_receiver_fn for `SequenceExample` inputs.

[`libsvm_generator(...)`](../ranking/data/libsvm_generator.md): Parses a LibSVM-formatted input file and aggregates data points by qid.

[`parse_from_sequence_example(...)`](../ranking/data/parse_from_sequence_example.md): Parses SequenceExample to feature maps.

[`read_batched_sequence_example_dataset(...)`](../ranking/data/read_batched_sequence_example_dataset.md): Returns a `Dataset` of features from `SequenceExample`.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

