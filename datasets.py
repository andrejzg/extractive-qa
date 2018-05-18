import logging
import json
from collections import defaultdict
import pickle
import hashlib

from nested_lookup import nested_lookup
import numpy as np

import data_ops


def find_cached_or_build_dataset(
    train_datasets,
    dev_datasets,
    max_context_len,
    max_answer_len,
    max_question_len,
    data_directory='data/cache',
    **unused
):
    """
    Nice shorthand to remove clutter from configs
    """
    hasher = hashlib.sha256()
    hasher.update(bytes(max_context_len))
    hasher.update(bytes(max_question_len))
    hasher.update(bytes(max_answer_len))
    hasher.update(bytes(''.join(sorted(dev_datasets.keys())), encoding='utf8'))
    hasher.update(bytes(''.join(sorted(train_datasets.keys())), encoding='utf8'))
    dataset_hash = hasher.digest().hex()[:10]

    try:
        with open(f'{data_directory}/dataset_{dataset_hash}.p', 'rb') as f:
            logging.info('found cached dataset')
            return pickle.load(f)
    except Exception as e:
        pass

    logging.info('building dataset...')
    data_for_experiment = build_datasets(
        train_datasets,
        dev_datasets,
        max_context_len,
        max_answer_len,
        max_question_len,
    )
    try:
        with open(f'{data_directory}/dataset_{dataset_hash}.p', 'wb') as f:
            logging.info('caching dataset at: ' + f'{data_directory}/dataset_{dataset_hash}.p')
            pickle.dump(data_for_experiment, f)
    except Exception as e:
        logging.error(e)
    return data_for_experiment


def one():
    """
    Used for Pickling defaultdicts as word2id
    """
    return 1


def build_datasets(
    train_datasets,
    dev_datasets,
    max_context_len,
    max_answer_len,
    max_question_len,
):
    """
    # Arguments
    * train_datasets a dict of {name: Dataset, ...}
    """
    # arguments that can be parameterized but aren't at the moment
    tokenizer = 'nltk'

    all_words = ' '.join([dataset.raw_words() for dataset in train_datasets.values()])
    unique_words = {word for word in data_ops.tokenize(all_words, tokenizer)}
    word2id = defaultdict(one)  # by default we use 0 for pad, 1 for unk (unknown words)
    for i, word in enumerate(unique_words, start=2):
        word2id[word] = i

    embeddings = data_ops.glove_embeddings(embedding_size=100)
    random = np.random.RandomState(123)
    embedding_matrix = data_ops.make_glove_embedding_matrix(
        word2vec=embeddings,
        word2id=word2id,
        unk_state=random.rand
    )

    train_data = {
        dataset_name: dataset.build(
            max_context_len,
            max_answer_len,
            max_question_len,
            word2id,
            dataset_type='train'
        ) for dataset_name, dataset in train_datasets.items()
    }
    dev_data = {
        dataset_name: dataset.build(
            max_context_len,
            max_answer_len,
            max_question_len,
            word2id,
            dataset_type='train'
        ) for dataset_name, dataset in dev_datasets.items()
    }

    misc = {
        'word2id': word2id,
        'embedding_matrix': embedding_matrix
    }
    return (train_data, dev_data, misc)


class SquadDataset():
    def __init__(self, dataset_type):
        if dataset_type == 'train':
            self.data_path = 'data/train-v1.1.json'
        elif dataset_type == 'development':
            self.data_path = 'data/dev-v1.1.json'
        else:
            raise RuntimeError(f'{dataset_type} is not a valid type')

        logging.info(f'loading squad {dataset_type}...')
        self.raw_data = json.load(open(self.data_path))
        self.dataset_type = dataset_type

    def raw_words(self):
        search_keys = {'context', 'question'}
        all_text = [
            text
            for key in search_keys
            for text in nested_lookup(key, self.raw_data)
        ]
        all_text = ' '.join(all_text)
        return all_text

    def build(
        self,
        max_context_len,
        max_answer_len,
        max_question_len,
        word2id,
        dataset_type=None
    ):
        return data_ops.make_squad_examples(
            self.raw_data,
            word2id=word2id,
            tokenizer='nltk',
            name=f'squad {dataset_type}',
            max_context_len=max_context_len,
            max_answer_len=max_answer_len,
            max_question_len=max_question_len
        )


class ConllDataset():

    def __init__(self, dataset_type):
        if dataset_type == 'train':
            self.data_path = 'data/conll/eng.train'
        elif dataset_type == 'development':
            self.data_path = 'data/conll/eng.testa'
        else:
            raise RuntimeError(f'{dataset_type} is not a valid type')

        logging.info(f'loading squad {dataset_type}...')

        def parse_conll(filepath):
            """ Parses a CoNLL-style file and turns it into nested list of (word,tag) tuples:
            [
            [(word, tag), (word, tag), (word, tag), (word,tag)],
            [(word, tag), (word, tag), (word, tag)]
            ]
            """
            data = open(filepath).read().split('\n\n')[1:]
            data = [[(x.split()[0], x.split()[-1].split('-')[-1]) for x in line.split('\n') if len(x) >= 2] for line in data if line]

            return data

        self.parsed_data = parse_conll(self.data_path)
        self.dataset_type = dataset_type

    def raw_words(self):
        return ' '.join([x[0] for line in self.parsed_data for x in line])

    def build(
        self,
        max_context_len,
        max_answer_len,
        max_question_len,
        word2id,
        dataset_type=None
    ):
        data, questions = self.data_questions_tuple(self.parsed_data)
        return data_ops.make_conll_examples(
            data,
            questions,
            word2id,
            name=None,
            max_context_len=max_context_len,
            max_answer_len=max_answer_len,
            max_question_len=max_question_len
        )

    def data_questions_tuple(self, original_data):
        np.random.seed(12345)
        possible_labels = ['PER', 'ORG', 'LOC']
        templates_dict = {
            1: ['Where are all the {}?', 'Mark all the {}', 'Highlight {}'],
            2: ['Where are all the {} and {}?', 'Mark any {} or {}', 'Highlight {} and {}'],

        }
        question_map = {}
        for label in possible_labels:
            question_map[label] = []
            other_labels = [l for l in possible_labels if l != label]
            for n, templates in templates_dict.items():
                fill_labels = [label] + [np.random.choice(other_labels) for _ in range(n - 1)]
                np.random.shuffle(fill_labels)
                question_map[label].extend([t.format(*fill_labels) for t in templates])

        questions = defaultdict(dict)
        data = []
        for r in range(100):
            for i, line in enumerate(original_data):
                labels = set([x[1] for x in line if x[1] in possible_labels])
                data.append(line)
                if len(labels) == 0:
                    questions[i + r * len(original_data)] = {}
                for label in labels:
                    questions[i + r * len(original_data)][label] = np.random.choice(question_map[label])
            # tmp condition
            if r == 2:
                break

        return data, questions
