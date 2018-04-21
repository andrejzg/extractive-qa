import json
import data_ops
import logging

from collections import defaultdict

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

train_raw = json.load(open('data/train-v1.1.json'))
dev_raw = json.load(open('data/dev-v1.1.json'))

try:
    word2id = json.load(open('data/word2id.json', 'rb'))
    word2id = defaultdict(lambda: 1, word2id)  # use 1 for unk 0 for pad
except FileNotFoundError:
    word2id = data_ops.build_vocab_from_json_searches(data=train_raw,
                                                      search_keys=['context', 'question'])
    json.dump(word2id, open('data/word2id.json', 'w'))


train = data_ops.make_examples(train_raw, tokenizer='nltk', name='train')
dev = data_ops.make_examples(dev_raw, tokenizer='nltk', name='dev')




def make_spans(tokens):
    pass


import code
code.interact(local=locals())