description: A TFR-BERT model configuration.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.extension.premade.TFRBertModelConfig" />
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
<meta itemprop="property" content="default_params"/>
<meta itemprop="property" content="dropout_rate"/>
<meta itemprop="property" content="encoder"/>
<meta itemprop="property" content="restrictions"/>
</div>

# tfr.extension.premade.TFRBertModelConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L86-L90">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A TFR-BERT model configuration.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfr.extension.premade.tfrbert_task.TFRBertModelConfig`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.extension.premade.TFRBertModelConfig(
    default_params: dataclasses.InitVar[Optional[Mapping[str, Any]]] = None,
    restrictions: dataclasses.InitVar[Optional[List[str]]] = None,
    dropout_rate: float = 0.1,
    encoder: encoders.EncoderConfig = <a href="../../../tfr/extension/premade/TFRBertModelConfig.md#encoder"><code>tfr.extension.premade.TFRBertModelConfig.encoder</code></a>
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `BUILDER`<a id="BUILDER"></a> </td> <td>

</td>
</tr><tr>
<td>
`default_params`<a id="default_params"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`restrictions`<a id="restrictions"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`encoder`<a id="encoder"></a>
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

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">What it enforces are</th></tr>
<tr class="alt">
<td colspan="2">
- a.a1 = 1 == b.ccc.a1 = 1
- a.a2 = 2 <= b.bb.bb2 = 20
</td>
</tr>

</table>

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
default_params<a id="default_params"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
dropout_rate<a id="dropout_rate"></a>
</td>
<td>
`0.1`
</td>
</tr><tr>
<td>
encoder<a id="encoder"></a>
</td>
<td>
Instance of `official.nlp.configs.encoders.EncoderConfig`
</td>
</tr><tr>
<td>
restrictions<a id="restrictions"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
