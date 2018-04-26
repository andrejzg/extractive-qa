import json
import data_ops
import logging
import pickle
import sys
import time
import experiment_logging
import shutil
import random
import itertools

import tensorflow as tf
import numpy as np

from tensorflow.python.training.summary_io import SummaryWriterCache
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

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
squad_train_raw = json.load(open('data/train-v1.1.json'))
squad_dev_raw = json.load(open('data/dev-v1.1.json'))
conll_train_raw = data_ops.parse_conll('data/conll/eng.train')
conll_dev_raw = data_ops.parse_conll('data/conll/eng.testa')

try:
    word2id = json.load(open('data/word2id.json', 'rb'))
    word2id = defaultdict(lambda: 1, word2id)  # use 1 for unk 0 for pad
except FileNotFoundError:
    word2id = data_ops.make_vocab_from_nested_lookups(
        data=squad_train_raw,
        search_keys=['context', 'question'],
        additional_words=[x[0] for ex in conll_train_raw for x in ex]
    )
    json.dump(word2id, open('data/word2id.json', 'w'))
    embeddings = data_ops.glove_embeddings(embedding_size=100)
    embedding_matrix = data_ops.make_glove_embedding_matrix(word2vec=embeddings, word2id=word2id)
    np.save('data/embedding_matrix.npy', embedding_matrix)


# Prepare data (tokenize + vectorize + truncate)
try:
    squad_train = pickle.load(open('data/squad_train.pkl', 'rb'))
    conll_train = pickle.load(open('data/conll_train.pkl', 'rb'))
    squad_dev = pickle.load(open('data/squad_dev.pkl', 'rb'))
    conll_dev = pickle.load(open('data/conll_dev.pkl', 'rb'))
except FileNotFoundError:
    glove_vectors = data_ops.glove_embeddings(100)
    squad_train = data_ops.make_squad_examples(
        squad_train_raw,
        word2id=word2id,
        tokenizer='nltk',
        name='squad train',
        max_context_len=config['max_context_len'],
        max_answer_len=config['max_answer_len'],
        max_question_len=config['max_question_len']
    )
    squad_dev = data_ops.make_squad_examples(
        squad_dev_raw,
        word2id=word2id,
        tokenizer='nltk',
        name='squad dev',
        max_context_len=config['max_context_len'],
        max_answer_len=config['max_answer_len'],
        max_question_len=config['max_question_len']
    )
    logging.info('Saving squad train and dev...')

    pickle.dump(squad_train, open('data/squad_train.pkl', 'wb'))
    pickle.dump(squad_dev, open('data/squad_dev.pkl', 'wb'))

    label2question = {
        'LOC': 'Mark all locations',
        'PER': 'Mark all people',
        'ORG': 'Mark all organisations'
    }

    conll_train = data_ops.make_conll_examples(
        conll_train_raw,
        word2id=word2id,
        label2question=label2question,
        name='conll train',
        max_context_len=config['max_context_len'],
        max_answer_len=config['max_answer_len'],
        max_question_len=config['max_question_len']
    )

    conll_dev = data_ops.make_conll_examples(
        conll_dev_raw,
        word2id=word2id,
        label2question=label2question,
        name='conll dev',
        max_context_len=config['max_context_len'],
        max_answer_len=config['max_answer_len'],
        max_question_len=config['max_question_len']
    )

    logging.info('Saving conll train and dev...')
    pickle.dump(squad_train, open('data/conll_train.pkl', 'wb'))
    pickle.dump(squad_dev, open('data/conll_dev.pkl', 'wb'))


trainsets = {
    'squad': squad_train,
    'conll': conll_train
}

devsets = {
    'squad': squad_dev,
    'conll': conll_dev
}

# Using config pick which datasets to train/eval on
train = list(itertools.chain(*[trainsets[x] for x in config['train']]))
dev = list(itertools.chain(*[devsets[x] for x in config['dev']]))

# Shuffle datasets
random.seed(config['seed'])
random.shuffle(train)
random.shuffle(dev)

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
        save_checkpoint_secs=300,
        save_summaries_steps=config['small_eval_every_steps']
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

        predicted_labels = (dev_probs > 0.5).astype(int)

        f1 = f1_score(
            y_true=dev_labels,
            y_pred=predicted_labels,
            average='weighted'
        )

        precision = precision_score(
            y_true=dev_labels,
            y_pred=predicted_labels,
            average='weighted'
        )

        recall = recall_score(
            y_true=dev_labels,
            y_pred=predicted_labels,
            average='weighted'
        )

        metrics_logger.log_scalar('f1/dev_small', f1, current_step)
        metrics_logger.log_scalar('precision/dev_small', precision, current_step)
        metrics_logger.log_scalar('recall/dev_small', recall, current_step)
