description: Provides a RankingPipeline for running a TF-Ranking model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.extension.pipeline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tfr.extension.pipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/pipeline.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Provides a `RankingPipeline` for running a TF-Ranking model.

This class contains the boilerplate required to run a TF-Ranking model, which
reduces a few replicated setups (e.g., input function, serving input function,
model export strategies) for running TF-Ranking models. Advanced users can also
derive from this class and further tailor for their needs.

## Classes

[`class RankingPipeline`](../../tfr/extension/pipeline/RankingPipeline.md):
Class to set up the input, train and eval processes for a TF Ranking model.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
absolute_import<a id="absolute_import"></a>
</td>
<td>
Instance of `__future__._Feature`
</td>
</tr><tr>
<td>
division<a id="division"></a>
</td>
<td>
Instance of `__future__._Feature`
</td>
</tr><tr>
<td>
print_function<a id="print_function"></a>
</td>
<td>
Instance of `__future__._Feature`
</td>
</tr>
</table>
