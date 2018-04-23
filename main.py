import json
import data_ops
import logging
import pickle
import sys
import time
import experiment_logging
import shutil

import tensorflow as tf
import numpy as np

from tensorflow.python.training.summary_io import SummaryWriterCache
from collections import defaultdict

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

# Read in user-provided config.py and run name.
args = sys.argv[1:]
assert len(args) == 2, 'usage: <path to config> <run_name>'
config_path, run_name = args

# Import the user-provided config.py as a module
config = data_ops.import_module(config_path).config

# Logdir naming convention
time_str = time.strftime('%d|%m|%Y@%H:%M:%S')
logdir = f'model_logs/{run_name}_{time_str}'

# Load raw data
train_raw = json.load(open('data/train-v1.1.json'))
dev_raw = json.load(open('data/dev-v1.1.json'))

try:
    word2id = json.load(open('data/word2id.json', 'rb'))
    word2id = defaultdict(lambda: 1, word2id)  # use 1 for unk 0 for pad
except FileNotFoundError:
    word2id = data_ops.build_vocab_from_json_searches(
        data=train_raw,
        search_keys=['context', 'question']
    )
    json.dump(word2id, open('data/word2id.json', 'w'))
    embedding_matrix = data_ops.make_glove_embedding_matrix(embedding_size=100, word2id=word2id)
    np.save('data/embedding_matrix.npy', embedding_matrix)


# Prepare data (tokenize + vectorize + truncate)
try:
    train = pickle.load(open('data/train.pkl', 'rb'))
    dev = pickle.load(open('data/dev.pkl', 'rb'))
except FileNotFoundError:
    glove_vectors = data_ops.glove_embeddings(100)
    train = data_ops.make_examples(
        train_raw,
        word2id=word2id,
        tokenizer='nltk',
        name='train',
        max_context_len=config['max_context_len'],
        max_answer_len=config['max_answer_len'],
        max_question_len=config['max_question_len']
    )
    dev = data_ops.make_examples(
        dev_raw,
        word2id=word2id,
        tokenizer='nltk',
        name='dev',
        max_context_len=config['max_context_len'],
        max_answer_len=config['max_answer_len'],
        max_question_len=config['max_question_len']
    )
    logging.info('Saving train and dev...')
    pickle.dump(train, open('data/train.pkl', 'wb'))
    pickle.dump(dev, open('data/dev.pkl', 'wb'))

# Prepare batchers
next_train = data_ops.make_batcher(train, batch_size=config['train_batch_size'])
next_dev = data_ops.make_batcher(dev, batch_size=config['dev_batch_size'])
next_small_dev = data_ops.make_batcher(dev, batch_size=config['small_dev_batch_size'])


# Graph inputs
context_t = tf.placeholder(
    tf.int32,
    [
        None,
        config['max_context_len'],
    ],
    name='context_t'
    )

question_t = tf.placeholder(
    tf.int32,
    [
        None,
        config['max_question_len'],
    ],
    name='question_t'
)

span2position = data_ops.make_span2position(
        seq_size=config['max_context_len'],
        max_len=config['max_answer_len']
    )

label_t = tf.placeholder(
    tf.float32,
    [None, len(span2position)],
    name='label_t'
)

# Model outputs
logits = config['model_fn'](
    context_t,
    question_t,
    span2position,
)

prediction_probs = tf.sigmoid(logits)

# Loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                               labels=label_t,
                                               name='multilabel_loss')
# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=config['lr'])
global_step_t = tf.train.create_global_step()
train_op = optimizer.minimize(loss, global_step=global_step_t)

# Session
sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=60,
        save_summaries_steps=50
        )

# Summaries
summary_writer = SummaryWriterCache.get(logdir)
metrics_logger = experiment_logging.TensorboardLogger(writer=summary_writer)
shutil.copyfile(config_path, logdir + '/config.py')  # save config in logdir

# Fetch entire dev set (no need to do this inside the eval loop repeatedly)
dev_batch = next_dev()

dev_feed_dict = {
        context_t: np.asarray([x['context'] for x in dev_batch]),
        question_t: np.asarray([x['question'] for x in dev_batch]),
        label_t: np.asarray([x['label'] for x in dev_batch]),
    }


# Train-Eval loop
while True:
    train_batch = next_train()
    train_feed_dict = {
        context_t: np.asarray([x['context'] for x in train_batch]),
        question_t: np.asarray([x['question'] for x in train_batch]),
        label_t: np.asarray([x['label'] for x in train_batch]),
    }
    current_step, train_loss, _ = sess.run(
        [
            global_step_t,
            loss,
            train_op
        ],
        feed_dict=train_feed_dict
    )

    metrics_logger.log_scalar('train/loss', train_loss.mean(), current_step)

    print('tick')
    if config['large_eval_every_steps'] and current_step != 0 and current_step % config['large_eval_every_steps'] == 0:
        logging.info('<large eval>:dev')

        dev_probs, dev_labels, dev_loss = sess.run(
            [
                prediction_probs,
                'label_t:0',
                loss
            ],
            feed_dict=dev_feed_dict
        )
    elif config['small_eval_every_steps'] and current_step != 0 and current_step % config['small_eval_every_steps'] == 0:
        logging.info('<small eval>:dev')

        dev_small_batch = next_small_dev()

        dev_small_feed_dict = {
            context_t: np.asarray([x['context'] for x in dev_small_batch]),
            question_t: np.asarray([x['question'] for x in dev_small_batch]),
            label_t: np.asarray([x['label'] for x in dev_small_batch]),
        }

        dev_probs, dev_labels, dev_loss = sess.run(
            [
                prediction_probs,
                'label_t:0',
                loss
            ],
            feed_dict=dev_small_feed_dict
        )

