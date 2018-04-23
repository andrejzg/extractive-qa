import models

config = {
        'vocab_size': 'glove',
        'lr': 5e-3,
        'embeddings_size': 100,
        'max_context_len': 300,
        'max_answer_len': 10,
        'max_question_len': 20,
        'large_eval_every_steps': None,
        'small_eval_every_steps': 20,
        'train_batch_size': 10,
        'dev_batch_size': None,
        'small_dev_batch_size': 10,
        'model_fn': models.rasor_net
    }
