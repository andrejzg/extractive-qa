from collections import defaultdict

import pytest
import numpy as np

import datasets


def dataset_test(
    dataset,
    max_context_len,
    max_answer_len,
    max_question_len,
):
    all_text = dataset.raw_words()
    assert type(all_text) == str

    max_context_len = 60
    max_answer_len = 3
    max_question_len = 10

    train_data = dataset.build(
        max_context_len=max_context_len,
        max_answer_len=max_answer_len,
        max_question_len=max_question_len,
        word2id=defaultdict(lambda: 1),
        dataset_type=None
    )
    contexts = [x['context'] for x in train_data]
    for context in contexts:
        assert len(context) == max_context_len
    questions = [x['question'] for x in train_data]
    for question in questions:
        assert len(question) == max_question_len
    answers = [x['answer'] for x in train_data]
    for answer in answers:
        assert len(answer) == max_answer_len


@pytest.mark.long
def test_squad_dataset():
    dataset_test(
        datasets.SquadDataset('train'),
        max_context_len=60,
        max_answer_len=3,
        max_question_len=10,
    )


@pytest.mark.long
def test_reproducible_dataset():
    train_datasets = {'squad': datasets.SquadDataset('train')}
    dev_datasets = {'squad': datasets.SquadDataset('development')}

    config = {
        'optimizer': None,
        'embeddings_size': 100,
        'max_context_len': 60,
        'max_answer_len': 5,
        'max_question_len': 20,
        'large_eval_every_steps': 200,
        'small_eval_every_steps': 20,
        'train_batch_size': 50,
        'dev_batch_size': 1000,
        'small_dev_batch_size': 50,
        'model_fn': None,
        'dropout': None,
        'random': None
    }

    def dataset_fn():
        return datasets.find_cached_or_build_dataset(
            train_datasets, dev_datasets,
            data_directory='/dev/null',
            **config,
        )

    train_data1, dev_data1, misc1 = dataset_fn()

    train_data2, dev_data2, misc2 = dataset_fn()

    for key in misc1:
        x, y = misc1[key], misc2[key]
        if isinstance(y, np.ndarray):
            np.testing.assert_array_equal(x, y)
        else:
            assert x == y

    for a, b in zip(train_data1['squad'], train_data2['squad']):
        for key in a:
            x, y = a[key], b[key]
            if isinstance(y, np.ndarray):
                np.testing.assert_array_equal(x, y)
            else:
                assert x == y
