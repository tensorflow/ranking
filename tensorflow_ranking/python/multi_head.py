# Copyright 2019 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines `Multi-Head`s of TF ranking models.

Given Muiti-Head config, combines heads
"""
import tensorflow_ranking as tfr
from tensorflow_ranking.python import losses as ranking_losses
import tensorflow as tf


def make_ndcg_metric_fn(score_key,
                        topn=None,
                        name=None):
    def _normalized_discounted_cumulative_gain_fn(labels, predictions, features):
        """Returns normalized discounted cumulative gain as the metric."""

        return tfr.metrics.normalized_discounted_cumulative_gain(
            labels,
            features[score_key],
            weights=None,
            topn=topn,
            name=name)

    return _normalized_discounted_cumulative_gain_fn


def get_eval_metric_fns(head_name):
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    metric_fns.update({
        "%s/metric/ndcg@%d" % (head_name, topn): tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 3, 5, 10]
    })
    return metric_fns


def create_head(head_config, logits_dim=1,
                use_unbiased=False):
    lambda_weight_func = None
    if head_config['use_lambda']:
        lambda_weight_func = ranking_losses.create_ndcg_lambda_weight()

    weights_feature_name = None
    if use_unbiased:
        weights_feature_name = 'label_weight'

    return tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(head_config['loss'],
                                        lambda_weight=lambda_weight_func,
                                        weights_feature_name=weights_feature_name),
        eval_metric_fns=get_eval_metric_fns(head_config['name']),
        train_op_fn=None,
        name=head_config['name'],
        logits_dim=logits_dim)


def create_multi_task_heads(mt_config, logits_dim=1,
                            use_unbiased=False):
    head_list = []
    head_weight = []
    for head_config in mt_config:
        head = create_head(head_config, logits_dim, use_unbiased)
        head_list.append(head)
        head_weight.append(head_config['weight'])

    ranking_head = tf.contrib.estimator.multi_head(head_list, head_weights=head_weight)
    return ranking_head
