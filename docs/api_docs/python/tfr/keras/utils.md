description: Utils for tfr.keras.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.keras.utils

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Utils for tfr.keras.

## Functions

[`identity(...)`](../../tfr/keras/utils/identity.md): Identity function that
returns the input label.

[`inverse(...)`](../../tfr/keras/utils/inverse.md): Computes the inverse of
input rank.

[`is_greater_equal_1(...)`](../../tfr/keras/utils/is_greater_equal_1.md):
Computes whether label is greater or equal to 1.

[`log2_inverse(...)`](../../tfr/keras/utils/log2_inverse.md): Computes
`1./log2(1+x)` element-wise for each label.

[`pow_minus_1(...)`](../../tfr/keras/utils/pow_minus_1.md): Computes `2**x - 1`
element-wise for each label.

[`symmetric_log1p(...)`](../../tfr/keras/utils/symmetric_log1p.md): Computes
`sign(x) * log(1 + sign(x))`.

## Type Aliases

[`GainFunction`](../../tfr/keras/utils/GainFunction.md)

[`PositiveFunction`](../../tfr/keras/utils/GainFunction.md)

[`RankDiscountFunction`](../../tfr/keras/utils/GainFunction.md)

[`TensorLike`](../../tfr/keras/model/TensorLike.md)
