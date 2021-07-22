description: The central part of internal API.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.task.FeatureSpec" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.task.FeatureSpec

<!-- Insert buttons and diff -->

This symbol is a **type alias**.

The central part of internal API.

#### Source:

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>FeatureSpec = <class 'dict'>[
    str,
    Union[tensorflow.python.ops.parsing_config.FixedLenFeature, tensorflow.python.ops.parsing_config.VarLenFeature, tensorflow.python.ops.parsing_config.RaggedFeature]
]
</code></pre>

<!-- Placeholder for "Used in" -->

This represents a generic version of type 'origin' with type arguments 'params'.
There are two kind of these aliases: user defined and special. The special ones
are wrappers around builtin collections and ABCs in collections.abc. These must
have 'name' always set. If 'inst' is False, then the alias can't be
instantiated, this is used by e.g. typing.List and typing.Dict.
