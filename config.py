import models

config = {
        'model_fn': 'rah',
        'vocab_size': 'glove',
        'lr': 5e-3,
        'embeddings_size': 100,
        'max_context_len': 300,
        'max_answer_len': 10,
        'eval_every_steps': 200,
        'train_batch_size': 10,
        'dev_batch_size': 10,
        'model_fn': models.simple_lstm_attn
    }