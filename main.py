import time
import logging
import sys
import shutil

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
import numpy as np
from tqdm import tqdm
from collections import Counter

import data_ops
import experiment_logging


def run_experiment(
    random,
    model_fn,
    dataset_fn,
    optimizer,
    train_batch_size,
    dev_batch_size,
    max_context_len,
    max_answer_len,
    max_question_len,
    eval_every_steps,
    dropout,
    logdir,
    **unused
):
    basic_metrics = {
        'f1_score': f1_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
    }
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

    span_mask_t = tf.placeholder(tf.int32, [None, len(span2position)], name='span_mask_t')

    label_t = tf.placeholder(tf.float32, [None, len(span2position)], name='label_t')

    position2span = {v: k for k, v in span2position.items()}
    id2word = {v: k for k, v in word2id.items()}

    # Model outputs
    logits_t, *the_rest = model_fn(
        context_t,
        context_t_length,
        question_t,
        question_t_length,
        span2position,
        embedding_matrix,
        span_mask_t,
    )

    # Build a mask which masks out-of-bound spans
    span_mask = tf.cast(span_mask_t, tf.float32)

    # Mask the logits of spans which shouldn't be considered
    logits_t *= span_mask

    logit_min = tf.reduce_min(logits_t, axis=1, keepdims=True)
    logits_t -= logit_min
    logits_t *= span_mask

    # Find the indexes of the predicted spans
    y_preds = tf.argmax(logits_t, axis=1)

    # For numerical stability reasons subtract the max
    logit_max = tf.reduce_max(logits_t, axis=1, keepdims=True)
    logits_t -= logit_max
    logits_t *= span_mask

    # Negative log likelihood (i.e. multiclass cross-entropy) loss
    exp_logits_t = tf.exp(logits_t) * span_mask
    log_sum_exp_logits_t = tf.log(tf.reduce_sum(exp_logits_t, axis=1) + 1e-7)

    gather_mask = tf.one_hot(y_preds, depth=logits_t.get_shape()[-1], dtype=tf.bool, on_value=True, off_value=False)
    y_logits = tf.boolean_mask(logits_t, gather_mask)

    xents = log_sum_exp_logits_t - y_logits

    loss_t = tf.reduce_mean(xents)
    tf.summary.scalar('mean_train_loss', loss_t)

    prediction_probs_t = exp_logits_t / tf.expand_dims(tf.reduce_sum(exp_logits_t, axis=1), 1)

    # Optimizer
    global_step_t = tf.train.create_global_step()
    train_op = optimizer.minimize(loss_t, global_step=global_step_t)

    # Session
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=60000,
        save_summaries_steps=50
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
            span_mask_t: np.asarray([x['span_mask'] for x in dataset])
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
            span_mask_t: np.asarray([x['span_mask'] for x in train_batch]),
            # 'out_dropout:0': dropout,
        }
        current_step, train_loss, _xents, _logits_t, _exp_logits_t, _log_sum_exp_logits_t,  *_the_rest = sess.run(
            [global_step_t, loss_t, xents, logits_t, exp_logits_t, log_sum_exp_logits_t] + the_rest + [train_op],
            feed_dict=train_feed_dict
        )

        if eval_every_steps is not None and current_step % eval_every_steps == 0:
            beginning_of_eval_time = time.time()
            logging.info('<large eval>:dev')

            # batch eval each dataset
            outputs_for_each_dataset = {}
            for dataset_name, dataset_feed_dict in dev_feed_dicts.items():
                logging.info(f'Computing dev outputs for {dataset_name}')
                batched_feed_dicts = [
                    {
                        placeholder: eval_data[i: i+dev_batch_size]
                        for placeholder, eval_data in dataset_feed_dict.items()
                    }
                    for i in range(0, len(dev_data[dataset_name]), dev_batch_size)
                ]
                dataset_model_output = None
                batched_model_outputs = [
                    sess.run(
                        {
                            'prediction_probs_t': prediction_probs_t,
                            'label_t': label_t,
                            'loss_per_example_t': xents
                        },
                        feed_dict=batch_feed_dict
                    ) for batch_feed_dict in tqdm(batched_feed_dicts)
                ]

                dataset_model_output = {
                    tensor_name: np.concatenate([output[tensor_name] for output in batched_model_outputs])
                    for tensor_name in batched_model_outputs[0].keys()
                }

                outputs_for_each_dataset[dataset_name] = dataset_model_output

            # much nicer, non batched version of evaluating
            # outputs_for_each_dataset = {
            #     dataset_name: sess.run(
            #         {
            #             'prediction_probs_t': prediction_probs_t,
            #             'label_t': label_t,
            #             'loss_per_example_t': loss_per_example_t
            #         },
            #         feed_dict=dataset_feed_dict
            #     ) for dataset_name, dataset_feed_dict in dev_feed_dicts.items()
            # }

            # build a combined dataset
            output_names = outputs_for_each_dataset[list(outputs_for_each_dataset.keys())[0]].keys()  # HACK

            all_dev_outputs = {
                output_name: np.concatenate([
                    outputs_for_each_dataset[dataset_name][output_name] for dataset_name in outputs_for_each_dataset
                ]) for output_name in output_names
            }
            outputs_for_each_dataset['combined'] = all_dev_outputs

            for dataset_name, dev_model_outputs in outputs_for_each_dataset.items():
                metrics_logger.log_scalar(
                    f'loss/{dataset_name}',
                    dev_model_outputs['loss_per_example_t'].mean(),
                    current_step
                )

                dev_probs = dev_model_outputs['prediction_probs_t']
                dev_labels = dev_model_outputs['label_t']

                # predicted_labels = (dev_probs > 0.5).astype(int)
                predicted_labels = (dev_probs.max(axis=1, keepdims=1) == dev_probs).astype(int)

                for metric_name, metric_fn in basic_metrics.items():
                    score = metric_fn(
                        y_true=np.ndarray.flatten(dev_labels),
                        y_pred=np.ndarray.flatten(predicted_labels),
                        average=None
                    )

                    for i, val in enumerate(score):
                        metrics_logger.log_scalar(
                            f'{metric_name}/{dataset_name}/label_{i}',
                            val,
                            current_step
                        )

                acc = accuracy_score(
                    y_true=np.ndarray.flatten(dev_labels),
                    y_pred=np.ndarray.flatten(predicted_labels),
                )

                metrics_logger.log_scalar(
                    f'accuracy/{dataset_name}',
                    acc,
                    current_step
                )

                if dataset_name == 'combined':  # only want per-dataset examples
                    continue

                context_dev = [x['context_raw'] for x in dev_data[dataset_name]]
                question_dev = [x['question_raw'] for x in dev_data[dataset_name]]

                np.all((dev_labels == predicted_labels), axis=1)

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

                prob_dist = np.argmax(dev_probs, axis=1)
                print('DEV predicted span distribution')
                print(Counter(prob_dist))

                # TODO: repeated code, move to methods? + the following code cannot handle cases where some spans are
                # correct and others aren't (it will just show them as being all wrong).

                if to_pick_correct:
                    correct_spans = [
                        [position2span[i] for i, x in enumerate(predicted_labels[p]) if x == 1]
                        for p in to_pick_correct
                    ]
                    correct_contexts = [context_dev[p] for p in to_pick_correct]
                    correct_questions = [question_dev[p] for p in to_pick_correct]

                    for s, c, q in zip(correct_spans, correct_contexts, correct_questions):
                        prompt = ' '.join(q)
                        experiment_logging.print_spans(c, s, prompt)

                if to_pick_wrong:
                    wrong_spans = [
                        [position2span[i] for i, x in enumerate(predicted_labels[p]) if x == 1]
                        for p in to_pick_wrong
                    ]
                    wrong_contexts = [context_dev[p] for p in to_pick_wrong]
                    wrong_questions = [question_dev[p] for p in to_pick_wrong]

                    for s, c, q in zip(wrong_spans, wrong_contexts, wrong_questions):
                        prompt = ' '.join(q)
                        experiment_logging.print_spans(
                            tokens=c,
                            spans=s,
                            prompt=prompt,
                            span_color='\x1b[6;30;41m',
                            prompt_color='\33[1m\33[31m'
                        )

            logging.info(f'evaluation took {time.time() - beginning_of_eval_time:.2f} seconds')


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

