import tensorflow as tf

import models

config = {
        'optimizer': tf.train.AdamOptimizer(),
        'embeddings_size': 100,
        'max_context_len': 300,
        'max_answer_len': 10,
        'max_question_len': 20,
        'large_eval_every_steps': None,
        'small_eval_every_steps': 30,
        'train_batch_size': 16,
        'dev_batch_size': 16,
        'small_dev_batch_size': 16,
        'model_fn': models.rasor_net,
        'seed': 1337,
        'train': ['squad', 'conll'],
        'dev': ['squad']
    }
