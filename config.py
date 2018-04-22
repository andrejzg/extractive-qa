import models

config = {
        'vocab_size': 'glove',
        'lr': 5e-3,
        'embeddings_size': 100,
        'max_context_len': 300,
        'max_answer_len': 10,
        'max_question_len': 20,
        'eval_every_steps': 200,
        'train_batch_size': 10,
        'dev_batch_size': 10,
        'model_fn': models.rasor_net
    }
