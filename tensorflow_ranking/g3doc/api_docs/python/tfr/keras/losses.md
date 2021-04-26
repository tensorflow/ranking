description: Keras losses in TF-Ranking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.keras.losses

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Keras losses in TF-Ranking.

## Classes

[`class ApproxMRRLoss`](../../tfr/keras/losses/ApproxMRRLoss.md): For
approximate MRR loss.

[`class ApproxNDCGLoss`](../../tfr/keras/losses/ApproxNDCGLoss.md): For
approximate NDCG loss.

[`class ClickEMLoss`](../../tfr/keras/losses/ClickEMLoss.md): For click EM loss.

[`class DCGLambdaWeight`](../../tfr/keras/losses/DCGLambdaWeight.md): Keras
serializable class for DCG.

[`class GumbelApproxNDCGLoss`](../../tfr/keras/losses/GumbelApproxNDCGLoss.md):
For Gumbel approximate NDCG loss.

[`class ListMLELambdaWeight`](../../tfr/keras/losses/ListMLELambdaWeight.md):
LambdaWeight for ListMLE cost function.

[`class ListMLELoss`](../../tfr/keras/losses/ListMLELoss.md): ListMLE loss.

[`class MeanSquaredLoss`](../../tfr/keras/losses/MeanSquaredLoss.md): Mean
squared loss.

[`class NDCGLambdaWeight`](../../tfr/keras/losses/NDCGLambdaWeight.md): Keras
serializable class for NDCG.

[`class PairwiseHingeLoss`](../../tfr/keras/losses/PairwiseHingeLoss.md):
Pairwise hinge loss.

[`class PairwiseLogisticLoss`](../../tfr/keras/losses/PairwiseLogisticLoss.md):
Pairwise logistic loss.

[`class PairwiseSoftZeroOneLoss`](../../tfr/keras/losses/PairwiseSoftZeroOneLoss.md):
Pairwise soft zero one loss.

[`class PrecisionLambdaWeight`](../../tfr/keras/losses/PrecisionLambdaWeight.md):
Keras serializable class for Precision.

[`class RankingLossKey`](../../tfr/keras/losses/RankingLossKey.md): Ranking loss
key strings.

[`class SigmoidCrossEntropyLoss`](../../tfr/keras/losses/SigmoidCrossEntropyLoss.md):
For sigmoid cross-entropy loss.

[`class SoftmaxLoss`](../../tfr/keras/losses/SoftmaxLoss.md): Softmax
cross-entropy loss.

[`class UniqueSoftmaxLoss`](../../tfr/keras/losses/UniqueSoftmaxLoss.md): For
unique softmax cross entropy loss.

## Functions

[`get(...)`](../../tfr/keras/losses/get.md): Factory method to get a ranking
loss class.
