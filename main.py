import logging
import sys
import shutil

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
import numpy as np

import data_ops
import experiment_logging


def run_experiment(
    train,
    dev,
    random,
    model_fn,
    dataset_fn,
    optimizer,
    train_batch_size,
    dev_batch_size,
    small_dev_batch_size,
    max_context_len,
    max_answer_len,
    max_question_len,
    small_eval_every_steps,
    large_eval_every_steps,
    dropout,
    logdir,
    **unused
):

    train_data, dev_data, misc = dataset_fn()
    assert len(train_data) > 0 and len(dev_data) > 0
    word2id = misc['word2id']
    embedding_matrix = misc['embedding_matrix']

    all_training_data = np.array(
        sum([dataset for dataset in train_data.values()], [])
    )
    # all_dev_data = np.array( for small dev batches
    #     sum([dataset for dataset in dev_data.values()], [])
    # )

    # Graph inputs
    context_t = tf.placeholder(tf.int32, [None, max_context_len], name='context_t')
    context_t_length = tf.placeholder(tf.int32, [None], name='context_t_length')

    question_t = tf.placeholder(tf.int32, [None, max_question_len], name='question_t')
    question_t_length = tf.placeholder(tf.int32, [None], name='question_t_length')

    span2position = data_ops.make_span2position(
        seq_size=max_context_len,
        max_len=max_answer_len
    )

    label_t = tf.placeholder(tf.float32, [None, len(span2position)], name='label_t')

    position2span = {v: k for k, v in span2position.items()}
    id2word = {v: k for k, v in word2id.items()}

    # Model outputs
    logits, spans = model_fn(
        context_t,
        context_t_length,
        question_t,
        question_t_length,
        span2position,
        embedding_matrix
    )

    # Build a mask which masks out-of-bound spans
    span_mask = tf.cast(tf.reduce_any(tf.not_equal(spans, 0), axis=-1), tf.float32)
    prediction_probs = tf.sigmoid(logits) * span_mask

    # Loss
    divergence = tf.nn.weighted_cross_entropy_with_logits(
        targets=label_t,
        logits=logits,
        pos_weight=50,
        name='multilabel_weighted_loss'
    )

    loss = tf.reduce_mean(divergence * span_mask)

    # Optimizer
    global_step_t = tf.train.create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step_t)

    # Session
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=300,
        save_summaries_steps=small_eval_every_steps
    )

    # Summaries
    summary_writer = SummaryWriterCache.get(logdir)
    metrics_logger = experiment_logging.TensorboardLogger(writer=summary_writer)
    shutil.copyfile(config_path, logdir + '/config.py')  # save config in logdir

    # Fetch entire dev set (no need to do this inside the eval loop repeatedly)
    dev_feed_dicts = {  # One feed dict for each dataset
        dataset_name: {
            context_t: np.asarray([x['context'] for x in dataset]),
            context_t_length: np.asarray([x['context_len'] for x in dataset]),
            question_t: np.asarray([x['question'] for x in dataset]),
            question_t_length: np.asarray([x['question_len'] for x in dataset]),
            label_t: np.asarray([x['label'] for x in dataset]),
        } for dataset_name, dataset in dev_data.items()
    }

    # Train-Eval loop
    epoch_indices = np.random.permutation(np.arange(len(all_training_data)))
    while True:
        train_indices = epoch_indices[:train_batch_size]
        if len(epoch_indices) < train_batch_size:
            epoch_indices = np.random.permutation(np.arange(len(all_training_data)))

        train_batch = all_training_data[train_indices]
        train_feed_dict = {
            context_t: np.asarray([x['context'] for x in train_batch]),
            context_t_length: np.asarray([x['context_len'] for x in train_batch]),
            question_t: np.asarray([x['question'] for x in train_batch]),
            question_t_length: np.asarray([x['question_len'] for x in train_batch]),
            label_t: np.asarray([x['label'] for x in train_batch]),
            'out_dropout:0': dropout,
        }
        current_step, train_loss, _ = sess.run(
            [
                global_step_t,
                loss,
                train_op
            ],
            feed_dict=train_feed_dict
        )

        basic_metrics = {
            'f1_score': f1_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
        }

        if large_eval_every_steps is not None and current_step % large_eval_every_steps == 0:
            metrics_logger.log_scalar('train/loss', train_loss.mean(), current_step)
            logging.info('<small eval>:dev')

            outputs_for_each_dataset = {
                dataset_name: sess.run(
                    {
                        'logits': logits,
                        'spans': spans,
                        'prediction_probs': prediction_probs,
                        'label_t': label_t,
                        'loss': loss
                    },
                    feed_dict=dataset_feed_dict
                ) for dataset_name, dataset_feed_dict in dev_feed_dicts.items()
            }

            # build a combined dataset
            output_names = outputs_for_each_dataset[list(outputs_for_each_dataset.keys())[0]].keys()  # HACK
            all_dev_outputs = {
                output_name: np.concatentate([
                    outputs_for_each_dataset[dataset_name][output_name] for dataset_name in outputs_for_each_dataset
                ]) for output_name in output_names
            }
            outputs_for_each_dataset['combined'] = all_dev_outputs

            for dataset_name, dev_model_outputs in outputs_for_each_dataset.items():

                dev_probs = dev_model_outputs['prediction_probs']
                dev_labels = dev_model_outputs['label_t']

                predicted_labels = (dev_probs > 0.5).astype(int)

                for metric_name, metric_fn in basic_metrics.items():
                    score = metric_fn(
                        y_true=np.ndarray.flatten(dev_labels),
                        y_pred=np.ndarray.flatten(predicted_labels),
                        average=None
                    )

                    for i, val in enumerate(score):
                        metrics_logger.log_scalar(
                            f'dev_large/{metric_name}/{dataset_name}/label_{i}',
                            val,
                            current_step
                        )

                acc = accuracy_score(
                    y_true=np.ndarray.flatten(dev_labels),
                    y_pred=np.ndarray.flatten(predicted_labels),
                )

                metrics_logger.log_scalar(
                    f'dev_large/{dataset_name}/accuracy',
                    acc,
                    current_step
                )

                if dataset_name == 'combined':  # only want per-dataset examples
                    continue

                context_dev = np.asarray([x['context'] for x in dev_feed_dicts[dataset_name]])
                question_dev = np.asarray([x['context'] for x in dev_feed_dicts[dataset_name]])
                to_pick_correct = experiment_logging.select_n_classified(
                    ground_truth=dev_labels,
                    predicted=predicted_labels,
                    correct=True,
                    n=2
                )

                to_pick_wrong = experiment_logging.select_n_classified(
                    ground_truth=dev_labels,
                    predicted=predicted_labels,
                    correct=False,
                    n=2
                )

                if to_pick_correct is not None:
                    experiment_logging.print_spans(
                        to_pick=to_pick_correct,
                        predicted_labels=predicted_labels,
                        context_ids=context_dev,
                        question_ids=question_dev,
                        position2span=position2span,
                        span_color='\x1b[6;30;42m',
                        id2word=id2word,
                    )
                if to_pick_wrong is not None:
                    experiment_logging.print_spans(
                        to_pick=to_pick_wrong,
                        predicted_labels=predicted_labels,
                        context_ids=context_dev,
                        question_ids=question_dev,
                        position2span=position2span,
                        span_color='\x1b[0;37;41m',
                        id2word=id2word,
                    )


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Read in user-provided config.py and run name.
    args = sys.argv[1:]
    assert len(args) == 2, 'usage: <path to config> <run_name>'
    config_path, run_name = args

    # Import the user-provided config.py as a module
    config = data_ops.import_module(config_path).config

    # Logdir naming convention
    logdir = f'model_logs/{run_name}'

    run_experiment(logdir=logdir, **config)
