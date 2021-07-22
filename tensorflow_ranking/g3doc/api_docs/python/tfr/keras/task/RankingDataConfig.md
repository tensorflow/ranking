description: Data set config.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.task.RankingDataConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="as_dict"/>
<meta itemprop="property" content="from_args"/>
<meta itemprop="property" content="from_json"/>
<meta itemprop="property" content="from_yaml"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="lock"/>
<meta itemprop="property" content="override"/>
<meta itemprop="property" content="replace"/>
<meta itemprop="property" content="validate"/>
<meta itemprop="property" content="IMMUTABLE_TYPES"/>
<meta itemprop="property" content="RESERVED_ATTR"/>
<meta itemprop="property" content="SEQUENCE_TYPES"/>
<meta itemprop="property" content="block_length"/>
<meta itemprop="property" content="cache"/>
<meta itemprop="property" content="convert_labels_to_binary"/>
<meta itemprop="property" content="cycle_length"/>
<meta itemprop="property" content="data_format"/>
<meta itemprop="property" content="dataset_fn"/>
<meta itemprop="property" content="default_params"/>
<meta itemprop="property" content="deterministic"/>
<meta itemprop="property" content="drop_remainder"/>
<meta itemprop="property" content="enable_tf_data_service"/>
<meta itemprop="property" content="global_batch_size"/>
<meta itemprop="property" content="input_path"/>
<meta itemprop="property" content="is_training"/>
<meta itemprop="property" content="list_size"/>
<meta itemprop="property" content="mask_feature_name"/>
<meta itemprop="property" content="restrictions"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="sharding"/>
<meta itemprop="property" content="shuffle_buffer_size"/>
<meta itemprop="property" content="shuffle_examples"/>
<meta itemprop="property" content="tf_data_service_address"/>
<meta itemprop="property" content="tf_data_service_job_name"/>
<meta itemprop="property" content="tfds_as_supervised"/>
<meta itemprop="property" content="tfds_data_dir"/>
<meta itemprop="property" content="tfds_name"/>
<meta itemprop="property" content="tfds_skip_decoding_feature"/>
<meta itemprop="property" content="tfds_split"/>
</div>

# tfr.keras.task.RankingDataConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/task.py#L49-L59">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Data set config.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.task.RankingDataConfig(
    default_params=None,
    restrictions=None,
    input_path=&#x27;&#x27;,
    tfds_name=&#x27;&#x27;,
    tfds_split=&#x27;&#x27;,
    global_batch_size=0,
    is_training: bool = True,
    drop_remainder: bool = True,
    shuffle_buffer_size=100,
    cache=False,
    cycle_length=None,
    block_length=1,
    deterministic=None,
    sharding=True,
    enable_tf_data_service=False,
    tf_data_service_address=None,
    tf_data_service_job_name=None,
    tfds_data_dir=&#x27;&#x27;,
    tfds_as_supervised=False,
    tfds_skip_decoding_feature=&#x27;&#x27;,
    seed: Optional[int] = None,
    data_format: str = &#x27;example_list_with_context&#x27;,
    dataset_fn: str = &#x27;tfrecord&#x27;,
    list_size: Optional[int] = None,
    shuffle_examples: bool = False,
    convert_labels_to_binary: bool = False,
    mask_feature_name: Optional[str] = &#x27;example_list_mask&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`default_params`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`restrictions`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`input_path`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tfds_name`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tfds_split`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`global_batch_size`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`is_training`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`shuffle_buffer_size`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`cache`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`cycle_length`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`block_length`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`deterministic`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`sharding`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`enable_tf_data_service`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tf_data_service_address`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tf_data_service_job_name`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tfds_data_dir`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tfds_as_supervised`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tfds_skip_decoding_feature`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`dataset_fn`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`list_size`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`shuffle_examples`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`convert_labels_to_binary`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`mask_feature_name`
</td>
<td>
Dataclass field
</td>
</tr>
</table>

## Methods

<h3 id="as_dict"><code>as_dict</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_dict()
</code></pre>

Returns a dict representation of params_dict.ParamsDict.

For the nested params_dict.ParamsDict, a nested dict will be returned.

<h3 id="from_args"><code>from_args</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_args(
    *args, **kwargs
)
</code></pre>

Builds a config from the given list of arguments.

<h3 id="from_json"><code>from_json</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_json(
    file_path: str
)
</code></pre>

Wrapper for `from_yaml`.

<h3 id="from_yaml"><code>from_yaml</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_yaml(
    file_path: str
)
</code></pre>

<h3 id="get"><code>get</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    key, value=None
)
</code></pre>

Accesses through built-in dictionary get method.

<h3 id="lock"><code>lock</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lock()
</code></pre>

Makes the ParamsDict immutable.

<h3 id="override"><code>override</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>override(
    override_params, is_strict=True
)
</code></pre>

Override the ParamsDict with a set of given params.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`override_params`
</td>
<td>
a dict or a ParamsDict specifying the parameters to be
overridden.
</td>
</tr><tr>
<td>
`is_strict`
</td>
<td>
a boolean specifying whether override is strict or not. If
True, keys in `override_params` must be present in the ParamsDict. If
False, keys in `override_params` can be different from what is currently
defined in the ParamsDict. In this case, the ParamsDict will be extended
to include the new keys.
</td>
</tr>
</table>

<h3 id="replace"><code>replace</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace(
    **kwargs
)
</code></pre>

Overrides/returns a unlocked copy with the current config unchanged.

<h3 id="validate"><code>validate</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate()
</code></pre>

Validate the parameters consistency based on the restrictions.

This method validates the internal consistency using the pre-defined list of
restrictions. A restriction is defined as a string which specfiies a binary
operation. The supported binary operations are {'==', '!=', '<', '<=', '>',
'>='}. Note that the meaning of these operators are consistent with the
underlying Python immplementation. Users should make sure the define
restrictions on their type make sense.

For example, for a ParamsDict like the following `a: a1: 1 a2: 2 b: bb: bb1: 10
bb2: 20 ccc: a1: 1 a3: 3` one can define two restrictions like this ['a.a1 ==
b.ccc.a1', 'a.a2 <= b.bb.bb2']

#### What it enforces are:

-   a.a1 = 1 == b.ccc.a1 = 1
-   a.a2 = 2 <= b.bb.bb2 = 20

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`KeyError`
</td>
<td>
if any of the following happens
(1) any of parameters in any of restrictions is not defined in
ParamsDict,
(2) any inconsistency violating the restriction is found.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
if the restriction defined in the string is not supported.
</td>
</tr>
</table>

<h3 id="__contains__"><code>__contains__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    key
)
</code></pre>

Implements the membership test operator.

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
IMMUTABLE_TYPES<a id="IMMUTABLE_TYPES"></a>
</td>
<td>
`(<class 'str'>,
 <class 'int'>,
 <class 'float'>,
 <class 'bool'>,
 <class 'NoneType'>)`
</td>
</tr><tr>
<td>
RESERVED_ATTR<a id="RESERVED_ATTR"></a>
</td>
<td>
`['_locked', '_restrictions']`
</td>
</tr><tr>
<td>
SEQUENCE_TYPES<a id="SEQUENCE_TYPES"></a>
</td>
<td>
`(<class 'list'>, <class 'tuple'>)`
</td>
</tr><tr>
<td>
block_length<a id="block_length"></a>
</td>
<td>
`1`
</td>
</tr><tr>
<td>
cache<a id="cache"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
convert_labels_to_binary<a id="convert_labels_to_binary"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
cycle_length<a id="cycle_length"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
data_format<a id="data_format"></a>
</td>
<td>
`'example_list_with_context'`
</td>
</tr><tr>
<td>
dataset_fn<a id="dataset_fn"></a>
</td>
<td>
`'tfrecord'`
</td>
</tr><tr>
<td>
default_params<a id="default_params"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
deterministic<a id="deterministic"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
drop_remainder<a id="drop_remainder"></a>
</td>
<td>
`True`
</td>
</tr><tr>
<td>
enable_tf_data_service<a id="enable_tf_data_service"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
global_batch_size<a id="global_batch_size"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
input_path<a id="input_path"></a>
</td>
<td>
`''`
</td>
</tr><tr>
<td>
is_training<a id="is_training"></a>
</td>
<td>
`True`
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
mask_feature_name<a id="mask_feature_name"></a>
</td>
<td>
`'example_list_mask'`
</td>
</tr><tr>
<td>
restrictions<a id="restrictions"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
seed<a id="seed"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
sharding<a id="sharding"></a>
</td>
<td>
`True`
</td>
</tr><tr>
<td>
shuffle_buffer_size<a id="shuffle_buffer_size"></a>
</td>
<td>
`100`
</td>
</tr><tr>
<td>
shuffle_examples<a id="shuffle_examples"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
tf_data_service_address<a id="tf_data_service_address"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
tf_data_service_job_name<a id="tf_data_service_job_name"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
tfds_as_supervised<a id="tfds_as_supervised"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
tfds_data_dir<a id="tfds_data_dir"></a>
</td>
<td>
`''`
</td>
</tr><tr>
<td>
tfds_name<a id="tfds_name"></a>
</td>
<td>
`''`
</td>
</tr><tr>
<td>
tfds_skip_decoding_feature<a id="tfds_skip_decoding_feature"></a>
</td>
<td>
`''`
</td>
</tr><tr>
<td>
tfds_split<a id="tfds_split"></a>
</td>
<td>
`''`
</td>
</tr>
</table>
