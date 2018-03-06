import json
import spacy


train_raw = json.load(open('data/train-v1.1.json'))
dev_raw = json.load(open('data/dev-v1.1.json'))

for example in train_raw['data']:
    title = example['title']

    for paragraph in example['paragraphs']:
        context = paragraph['context']
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')






import code
code.interact(local=locals())