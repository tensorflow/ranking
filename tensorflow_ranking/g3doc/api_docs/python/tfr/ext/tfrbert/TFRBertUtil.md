<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.ext.tfrbert.TFRBertUtil" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convert_to_elwc"/>
<meta itemprop="property" content="create_optimizer"/>
<meta itemprop="property" content="get_warm_start_settings"/>
</div>

# tfr.ext.tfrbert.TFRBertUtil

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/tfrbert.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Class that defines a set of utility functions for Bert.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.ext.tfrbert.TFRBertUtil(
    bert_config_file, bert_init_ckpt, bert_max_seq_length, bert_vocab_file=None,
    do_lower_case=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`bert_config_file`
</td>
<td>
(string) path to Bert configuration file.
</td>
</tr><tr>
<td>
`bert_init_ckpt`
</td>
<td>
(string)  path to pretrained Bert checkpoint.
</td>
</tr><tr>
<td>
`bert_max_seq_length`
</td>
<td>
(int) maximum input sequence length (#words) after
WordPiece tokenization. Sequences longer than this will be truncated,
and shorter than this will be padded.
bert_vocab_file (optional): (string) path to Bert vocabulary file.
do_lower_case (optional): (bool) whether to lower case the input text.
This should be aligned with the `vocab_file`.
</td>
</tr>
</table>

## Methods

<h3 id="convert_to_elwc"><code>convert_to_elwc</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/tfrbert.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert_to_elwc(
    context, examples, labels, label_name
)
</code></pre>

Converts a <context, example list> pair to an ELWC example.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context`
</td>
<td>
(str) raw text for a context (aka. query).
</td>
</tr><tr>
<td>
`examples`
</td>
<td>
(list) raw texts for a list of examples (aka. documents).
</td>
</tr><tr>
<td>
`labels`
</td>
<td>
(list) a list of labels (int) for the `examples`.
</td>
</tr><tr>
<td>
`label_name`
</td>
<td>
(str) name of the label in the ELWC example.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tensorflow.serving.ExampleListWithContext example containing the
`input_ids`, `input_masks`, `segment_ids` and `label_id` fields.
</td>
</tr>

</table>

<h3 id="create_optimizer"><code>create_optimizer</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/tfrbert.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_optimizer(
    init_lr, train_steps, warmup_steps, optimizer_type='adamw'
)
</code></pre>

Creates an optimizer for TFR-BERT.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`init_lr`
</td>
<td>
(float) the init learning rate.
</td>
</tr><tr>
<td>
`train_steps`
</td>
<td>
(int) the number of train steps.
</td>
</tr><tr>
<td>
`warmup_steps`
</td>
<td>
(int) if global_step < num_warmup_steps, the learning rate
will be `global_step / num_warmup_steps * init_lr`. See more details in
the `tensorflow_models.official.nlp.optimization.py` file.
</td>
</tr><tr>
<td>
`optimizer_type`
</td>
<td>
(string) Optimizer type, can either be `adamw` or `lamb`.
Default to be the `adamw` (AdamWeightDecay). See more details in the
`tensorflow_models.official.nlp.optimization.py` file.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The optimizer training op.
</td>
</tr>

</table>

<h3 id="get_warm_start_settings"><code>get_warm_start_settings</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/tfrbert.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_warm_start_settings(
    exclude
)
</code></pre>

Defines warm-start settings for the TFRBert ranking estimator.

Our TFRBert ranking models will warm-start from a pre-trained Bert model. Here,
we define the warm-start setting by excluding non-Bert parameters.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`exclude`
</td>
<td>
(string) Variable to exclude from the warm-start settings.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
(`tf.estimator.WarmStartSettings`) the warm-start setting for the TFRBert
ranking estimator.
</td>
</tr>

</table>
