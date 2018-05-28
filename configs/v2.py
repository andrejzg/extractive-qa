import numpy as np
import tensorflow as tf

import models
import datasets

dataset_seed = 1337
train_datasets = {'squad': datasets.SquadDataset('train')}
dev_datasets = {'squad': datasets.SquadDataset('development')}
# train_datasets = {'squad': datasets.SquadDataset('debug_train')}
# dev_datasets = {'squad': datasets.SquadDataset('debug_dev')}
# train_datasets = {'conll': datasets.ConllDataset('train')}
# dev_datasets = {'conll': datasets.ConllDataset('development')}

config = {
    'optimizer': tf.train.AdamOptimizer(learning_rate=5e-4),
    'embeddings_size': 100,
    'max_context_len': 300,
    'max_answer_len': 1,
    'max_question_len': 20,
    'eval_every_steps': 100,
    'train_batch_size': 50,
    'dev_batch_size': 2000,
    'model_fn': models.rasor_net,
    'dropout': 0.8,
    'random': np.random.RandomState(1337)
}

config['dataset_fn'] = lambda: datasets.find_cached_or_build_dataset(
    train_datasets, dev_datasets, **config
)
