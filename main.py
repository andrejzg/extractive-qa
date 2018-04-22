import json
import data_ops
import logging
import pickle
import sys
import time
import experiment_logging
import shutil

import tensorflow as tf

from itertools import cycle
from tensorflow.python.training.summary_io import SummaryWriterCache

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

# Prepare data (tokenize + vectorize + truncate)
try:
    train = pickle.load(open('data/train.pkl', 'rb'))
    dev = pickle.load(open('data/dev.pkl', 'rb'))
except FileNotFoundError:
    glove_vectors = data_ops.glove_embeddings(100)
    train = data_ops.make_examples(
        train_raw,
        tokenizer='nltk',
        name='train',
        max_context_len=config['max_context_len'],
        max_answer_len=config['max_answer_len'],
        word2vec=glove_vectors,
        embedding_size=config['embeddings_size']
    )
    dev = data_ops.make_examples(
        dev_raw,
        tokenizer='nltk',
        name='dev',
        max_context_len=config['max_context_len'],
        max_answer_len=config['max_answer_len'],
        word2vec=glove_vectors,
        embedding_size=config['embeddings_size']
    )
    logging.info('Saving train and dev...')
    pickle.dump(train, open('data/train.pkl', 'wb'))
    pickle.dump(dev, open('data/dev.pkl', 'wb'))

# Prepare batchers
train_it = cycle(train)
dev_it = cycle(dev)

next_train_batch = lambda: [next(train_it) for _ in range(config['train_batch_size'])]
next_dev_batch = lambda: [next(train_it) for _ in range(config['train_batch_size'])]

# Graph inputs
context_t = tf.placeholder(
    tf.float32,
    [
        None,
        config['max_context_len'],
        config['embeddings_size']
    ],
    name='contexts_t'
    )

context_t_length = tf.placeholder(
    tf.int32,
    [None],
    name='context_length'
)

question_t = tf.placeholder(
    tf.float32,
    [
        None,
        config['max_question_len'],
        config['embeddings_size']],
    name='questions_t'
)

question_t_length = tf.placeholder(
    tf.int32,
    [None],
    name='question_length'
)


label_t = tf.placeholder(
    tf.int32,
    [None, ],
    name='labels_t'
)


# Model outputs
logits = config['model_fn'](
    context_t,
    context_t_length,
    question_t,
    question_t_length,
    config
)
# prediction_probs = tf.nn.softmax(logits, axis=-1)
# predictions = tf.argmax(logits, axis=-1)
#
# # Loss
# # TODO: MULTILABEL LOSS
# loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
#                                                labels=labels_t,
#                                                name='multilabel_loss')
# # Optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=config['lr'])
# global_step_t = tf.train.create_global_step()
# train_op = optimizer.minimize(loss, global_step=global_step_t)
#
# # Session
# sess = tf.train.MonitoredTrainingSession(
#         checkpoint_dir=logdir,
#         save_checkpoint_secs=60,
#         save_summaries_steps=50
#         )
#
#
# # Summaries
# summary_writer = SummaryWriterCache.get(logdir)
# metrics_logger = experiment_logging.TensorboardLogger(writer=summary_writer)
# shutil.copyfile(config_path, logdir + '/config.py')  # save config in logdir

# Fetch entire dev set (no need to do this inside the eval loop repeatedly)
# image_dev, question_dev, label_dev = next(batcher_dev)
# dev_feed_dict = {
#                 images_t: image_dev,
#                 questions_t: question_dev,
#                 labels_t: label_dev
#             }

import code
code.interact(local=locals())
