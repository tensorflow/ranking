<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.segment_sorted_ranks" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.segment_sorted_ranks

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Returns an int `Tensor` as the ranks after sorting scores per segment.

```python
tfr.utils.segment_sorted_ranks(
    scores, segments, shuffle_ties=True, seed=None
)
```

<!-- Placeholder for "Used in" -->

The returned ranks are 1-based. For example: scores = [1.0, 3.5, 2.1] segments =
[0, 0, 1] returned ranks = [2, 1, 1] The first 2 scores belong to the same
segment and the first score 1.0 is at rank 2 and second score 3.5 is in rank 1.
The last score is in another segment and its rank is 1 and there is no other
scores in this segment.

#### Args:

*   <b>`scores`</b>: A 1-D `Tensor` representing the scores to be sorted.
*   <b>`segments`</b>: A 1-D `Tensor` representing the segments that each score
    belongs to. This should be the same shape as the scores.
*   <b>`shuffle_ties`</b>: See `sort_by_scores`.
*   <b>`seed`</b>: See `sort_by_scores`.

#### Returns:

A 1-D int `Tensor`s as the ranks (1-based).
