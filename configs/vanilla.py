import tensorflow as tf

import models

config = {
        'optimizer': tf.train.AdamOptimizer(learning_rate=1e-3),
        'embeddings_size': 100,
        'max_context_len': 60,
        'max_answer_len': 5,
        'max_question_len': 20,
        'large_eval_every_steps': 200,
        'small_eval_every_steps': 20,
        'train_batch_size': 50,
        'dev_batch_size': 1000,
        'small_dev_batch_size': 50,
        'model_fn': models.rasor_net,
        'seed': 1337,
        'train': ['conll'],
        'dev': ['conll'],
        'dropout': 0.5
    }
