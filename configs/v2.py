import numpy as np
import tensorflow as tf

import models
import datasets

dataset_seed = 1337
train_datasets = (datasets.SquadDataset('train'),)
dev_datasets = (datasets.SquadDataset('development'),)

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
    'dropout': 0.5,
    'random': np.random.RandomState(1337)
}

config['dataset_fn'] = lambda: datasets.find_cached_or_build_dataset(
    train_datasets, dev_datasets, **config
)
