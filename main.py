import json
import data_ops
import logging
import pickle
import sys
import time

import tensorflow as tf

from itertools import cycle

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

# Prepare config file
config = {
        'model_fn': 'rah',
        'vocab_size': 'glove',
        'lr': 5e-3,
        'embeddings_size': 100,
        'max_context_len': 300,
        'max_answer_len': 10,
        'eval_every_steps': 200,
        'train_batch_size': 10,
        'dev_batch_size': 10
    }

# Read in user-provided config.py and run name.
args = sys.argv[1:]
assert len(args) == 2, 'usage: <path to config> <run_name>'
config_path, run_name = args

# Import the user-provided config.py as a module
user_config = data_ops.import_module(config_path).config

# Logdir naming convention
time_str = time.strftime('%d|%m|%Y@%H:%M:%S')
logdir = f'model_logs/{run_name}_{time_str}'

# Load raw data
train_raw = json.load(open('data/train-v1.1.json'))
dev_raw = json.load(open('data/dev-v1.1.json'))

# Prepare data (tokenize + vectorize + truncate)
try:
    train = pickle.load(open('data/train.pkl', 'rb'))
    test = pickle.load(open('data/dev.pkl', 'rb'))
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
        max_context_len=config['max_context_lent'],
        max_answer_len=config['max_answer_len'],
        word2vec=glove_vectors,
        embedding_size=config['embeddings_size']
    )
    logging.info('Saving train and dev...')
    pickle.dump(train, open('data/train.pkl', 'wb'))
    pickle.dump(dev, open('data/dev.pkl', 'wb'))


# Prepare batchers
train_it = cycle(train)
dev_it = cycle(test)

next_train_batch = lambda: [next(train_it) for _ in range(config['train_batch_size'])]
next_dev_batch = lambda: [next(train_it) for _ in range(config['train_batch_size'])]

# Graph inputs
contexts_t = tf.placeholder(tf.float32, [None, None, config['embeddings_size']], name='contexts_t')
questions_t = tf.placeholder(tf.float32, [None, None, config['embeddings_size']], name='questions_t')


import code
code.interact(local=locals())
