<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.ext.pipeline" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.ext.pipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

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

[`class RankingPipeline`](../../tfr/ext/pipeline/RankingPipeline.md): Class to
set up the input, train and eval processes for a TF Ranking model.
