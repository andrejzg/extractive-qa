import logging
import os
import importlib.util
from itertools import islice
from subprocess import call
import json
from itertools import groupby
from collections import defaultdict

import numpy as np
import spacy
import nltk
from spacy.lang.en import English
from nltk.parse.corenlp import CoreNLPParser
from tqdm import tqdm

import evaluate

logging.basicConfig(level=logging.INFO)

BASEDIR = os.path.join(os.path.dirname(__file__), '.')

nlp = spacy.load('en')
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)
stanford_tokenizer = CoreNLPParser()


def index_by_starting_character(text, tokens):
    """
    Given a string of text and a list of its tokens produce a dict where the keys are the character start
    positions of the tokens and the values are [token, index] where index denotes the position of the token
    in the text.
    """
    seen = ''
    current_token_id = 0
    token_map = dict()
    for char_id, char in enumerate(text):
        if char != u' ':
            seen += char
            context_token = tokens[current_token_id]
            if seen == context_token:
                syn_start = char_id - len(seen) + 1
                token_map[syn_start] = [seen, current_token_id]
                seen = ''
                current_token_id += 1
    return token_map


def tokenize(text, tokenizer):
    """
    Tokenize text into a list of tokens using one of the following tokenizers:
        - nltk
        - spacy
        - stanford
    """
    if tokenizer == 'nltk':
        tokens = ['"' if token in {'``', '\'\''} else token for token in nltk.word_tokenize(text)]
    elif tokenizer == 'spacy':
        tokens = [word.text for word in spacy_tokenizer(text)]
    elif tokenizer == 'stanford':
        tokens = list(stanford_tokenizer.tokenize(text))
    else:
        raise ValueError('tokenizer param must be one of the following: nltk|spacy|stanford')
    return tokens


def fix_double_quotes(text):
    """
    Given some text standardize all its double quotes by replacing them with the standard double quote symbol.
    """
    ans = text.replace("''", '" ')
    ans = ans.replace("``", '" ')
    return ans


def fix_whitespace(text):
    """
    Remove all non-single spaces from text.
    """
    return ' '.join([x for x in [x.strip() for x in text.split(' ')] if x != ''])


def make_squad_examples(data,
                        tokenizer,
                        word2id,
                        name=None,
                        max_context_len=300,
                        max_answer_len=10,
                        max_question_len=20,
                        ):
    """
    Given a SQuAD dataset, builds a list of example dicts (see implementation).
    """
    examples = []
    total = 0
    total_passed = 0
    skipped = defaultdict(lambda: 0)
    span2position = make_span2position(
        seq_size=max_context_len,
        max_len=max_answer_len
    )
    position2span = {v: k for k, v in span2position.items()}

    for line in tqdm(data['data'], desc=name):
        title = line['title']

        for paragraph in line['paragraphs']:
            # Extract context
            context = paragraph['context']
            context = fix_double_quotes(context)
            context_tokens = tokenize(context, tokenizer=tokenizer)

            if max_context_len and len(context_tokens) > max_context_len:
                skipped['context too long'] += len(paragraph['qas'])
                total += len(paragraph['qas'])
                continue

            answer_map = index_by_starting_character(context, context_tokens)

            for qa in paragraph['qas']:
                # Extract question
                question = qa['question']
                question = fix_double_quotes(question)
                question_tokens = tokenize(question, tokenizer=tokenizer)

                # Extract answer
                answer = qa['answers'][0]['text']
                answer = fix_double_quotes(answer)
                answer_tokens = tokenize(answer, tokenizer=tokenizer)

                if max_answer_len and len(answer_tokens) > max_answer_len:
                    skipped['answer too long'] += 1
                    total += 1
                    continue

                answer_start = qa['answers'][0]['answer_start']
                answer_end = answer_start + len(answer)

                # Find answer span
                try:
                    last_word_answer = len(answer_tokens[-1])  # add one to get the first char

                    _, span_start = answer_map[answer_start]  # start token index
                    _, span_end = answer_map[answer_end - last_word_answer]

                    extracted_answer = context_tokens[span_start:span_end + 1]
                    extracted_answer = ' '.join(extracted_answer)
                    extracted_answer = evaluate.normalize_answer(extracted_answer)

                    actual_clean = evaluate.normalize_answer(answer)

                    assert extracted_answer == actual_clean, f'{extracted_answer} != {actual_clean}'

                    span_positions = [span2position[(span_start, span_end)]]

                    s, e = position2span[span_positions[0]]
                    assert ' '.join(context_tokens[s:e+1]) == ' '.join(answer_tokens), \
                        'Extracted span does not match answer'

                    correct_spans = np.asarray(list({
                                                        k: v for
                                                        k, v in
                                                        span2position.items()
                                                        if np.all(np.asarray(k) < len(context_tokens))
                                                    }.values()))
                    span_mask = np.zeros(len(span2position))
                    span_mask[correct_spans] = 1

                    example = {
                        'title': title,
                        'context_raw': context_tokens,
                        'question_raw': question_tokens,
                        'answer_raw': answer_tokens,
                        'context': pad_seq([word2id[w] for w in context_tokens], maxlen=max_context_len),
                        'question': pad_seq([word2id[w] for w in question_tokens], maxlen=max_question_len),
                        'answer': pad_seq([word2id[w] for w in answer_tokens], maxlen=max_answer_len),
                        'context_len': len_or_maxlen(context_tokens, max_context_len),
                        'question_len': len_or_maxlen(question_tokens, max_question_len),
                        'answer_len': len_or_maxlen(answer_tokens, max_answer_len),
                        'starts': [span_start],
                        'ends': [span_end],
                        'span_positions': span_positions,
                        'span_mask': span_mask,
                        'label': np.asarray([1 if x in span_positions else 0 for x in span2position.values()])
                    }

                    total_passed += 1
                    total += 1
                    
                    examples.append(example)

                except (AssertionError, KeyError) as e:
                    skipped['error finding span'] += 1
                    total += 1
                    continue

                examples.append(example)

    total_skipped = sum(skipped.values())
    ratio_skipped = total_skipped/total if total != 0 else 0
    logging.info(f'max_context_len: {max_context_len}')
    logging.info(f'max_answer_len: {max_answer_len}')
    logging.info(f'skipped {skipped}/{total}\t({ratio_skipped})')
    print(json.dumps(skipped, indent=4))
    print(f'ratio skipped: {ratio_skipped}')
    print(f'{total_passed} examples PASSED')
    return examples


def len_or_maxlen(seq, maxlen):
    return len(seq) if len(seq) <= maxlen else maxlen


def make_conll_examples(
    data,
    questions,
    word2id,
    name=None,
    max_context_len=300,
    max_answer_len=10,
    max_question_len=20
):
    """
    Given a CoNLL dataset, builds a list of example dicts (see implementation).
    """

    span2position = make_span2position(
        seq_size=max_context_len,
        max_len=max_answer_len
    )

    examples = []

    for i, line in tqdm(enumerate(data), desc=name):
        context_tokens = [x[0] for x in line]
        labels = [x[1] for x in line]

        for label, question in questions[i].items():
                if max_context_len and len(context_tokens) > max_context_len:
                    # context_tokens = context_tokens[:max_context_len]
                    continue
                question_tokens = question.split()
                indicators = [1 if x == label else 0 for x in labels]

                span_starts = []
                span_ends = []

                for k, g in groupby(enumerate(indicators), lambda ix: ix[1]):
                    if k == 1:
                        res = list(g)
                        if max_answer_len and len(res) > max_answer_len:
                            continue
                        span_starts.append(res[0][0])
                        span_ends.append(res[-1][0])

                span_positions = [span2position[(s, e)] for s, e in zip(span_starts, span_ends)]

                answer_tokens = [token for s, e in zip(span_starts, span_ends) for token in context_tokens[s:e+1]]

                example = {
                    'title': '',
                    'context_raw': context_tokens,
                    'question_raw': question_tokens,
                    'answer_raw': answer_tokens,
                    'context': pad_seq([word2id[w] for w in context_tokens], maxlen=max_context_len),
                    'question': pad_seq([word2id[w] for w in question_tokens], maxlen=max_question_len),
                    'answer': pad_seq([word2id[w] for answer in answer_tokens for w in answer], maxlen=max_answer_len),
                    'context_len': len_or_maxlen(context_tokens, max_context_len),
                    'question_len': len_or_maxlen(question_tokens, max_question_len),
                    'answer_len': len_or_maxlen(answer_tokens, max_answer_len),
                    'starts': span_starts,
                    'ends': span_ends,
                    'span_positions': span_positions,
                    'label': np.asarray([1 if x in span_positions else 0 for x in span2position.values()])
                }

                examples.append(example)
    return examples


def make_spans(seq, max_len=10):
    """
    Given a sequence creates a list of spans up to and including size max_len where every span is an indexed windows
    (start_idx, end_idx, [items]). See make_indexed_windows for more information.
    """
    spans = []
    for span_len in range(1, max_len+1):
        spans.extend(list(make_indexed_windows(seq, n=span_len)))
    # now sort the spans by start position + end position (if start is the same)
    return sorted(spans, key=lambda x: (x[0], x[1]))


def make_indexed_windows(seq, n=2):
    """
    Return a sliding window of n items from a sequence of items. Every window is a tuple (star_idx, end_idx, [items])
    where start_idx and end_idx are the start and end item indexes of the window items and [items] are the items
    themselves.
    """
    it = iter(enumerate(seq))
    result = tuple(islice(it, n))
    if len(result) == n:
        yield (result[0][0], result[-1][0], [x[1] for x in result])
    for elem in it:
        result = result[1:] + (elem,)
        yield (result[0][0], result[-1][0], [x[1] for x in result])


def make_span2position(seq_size, max_len):
    """
    Create a dictionary (start_idx, end_idx) -> span position.
    """
    seq = [0] * seq_size  # getting a bit hacky here...
    spans = make_spans(seq, max_len)
    span2position = {}
    for i, span in enumerate(spans):
        span2position[(span[0], span[1])] = i
    return span2position


def glove_embeddings(embedding_size, emb_path=None, script_path=None):
    """
    Prepare a word2vec dictionary {word -> vector} where the pre-trained vector embeddings are GloVe embeddings.
    If the user does not have glove vectors in the project's /data directory then download them into it.
    """
    emb_path = emb_path if emb_path else '{}/data/glove/glove.6B.{}d.txt'.format(BASEDIR, embedding_size)

    try:
        f = open(emb_path, 'r')
    except IOError:
        call(script_path if script_path else '{}/download_glove.sh'.format(BASEDIR), shell=True)
        f = open(emb_path, 'r')

    rows = f.read().split('\n')[:-1]

    def _parse_embedding_row(embedding_row):
        word, string_embedding = embedding_row.split(' ', 1)
        return word, np.fromstring(string_embedding, sep=' ')

    return dict([_parse_embedding_row(row) for row in tqdm(rows, desc='Parsing glove file.')])


def make_glove_embedding_matrix(word2id, word2vec, unk=0, pad=1, unk_state=np.zeros):
    """
    Takes (1) a word2id dictionary and (2) a word2vec dictionary and creates a numpy embedding matrix
    |vocab size| x |embedding size| with matched id -> vector matrix rows. The reason why we need this function
    is because we tend to build the word2id dict ourselves from tokenized text data whereas word2vec usually
    comes from some outside source.

    IMPORTANT:
    - assumes word2id reserves values for pad and unknown words
    - by default words in (1) but not in (2) are set to unk_state (e.g. np.zeros or np.random.rand)
    - embeddings of words not found in word2id are THROWN AWAY in this function (a workaround it to make sure they're
      included in word2id)
    """
    embedding_size = len(next(iter(word2vec.values())))  # get first value from dict to find

    embedding_matrix = np.zeros((len(word2id) + 2, embedding_size))  # +2 for unk & pad
    zero_vec = np.zeros(embedding_size)
    embedding_matrix[pad] = zero_vec  # pad
    embedding_matrix[unk] = zero_vec  # unk

    for word, id in word2id.items():
        if word in word2vec:
            vec = word2vec[word]
        elif word.lower() in word2vec:
            vec = word2vec[word.lower()]
        else:
            vec = unk_state(embedding_size)

        assert embedding_size == len(vec)
        embedding_matrix[id] = vec

    return embedding_matrix


def vectorize_tokens(tokens, word2vec):
    """
    Given a sequence of tokens, vectorize them using a word2vec dict.
    """
    embedding_size = len(next(iter(word2vec.values())))  # get first value from dict to find

    token_vectors = []
    for token in tokens:
        if token in word2vec:
            token_vectors.append(word2vec[token])
        else:
            if token.lower() in word2vec:
                token_vectors.append(word2vec[token.lower()])
            else:
                token_vectors.append(np.zeros(embedding_size))
    return np.asarray(token_vectors)


def import_module(path):
    """
    Primarily used for importing config files.
    """
    spec = importlib.util.spec_from_file_location('', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pad_seq(seq, maxlen, reverse=False):
    """ Pad or shorten list of items to a specified maxlen """
    res = seq
    if len(seq) > maxlen:
        if reverse:
            del res[:(len(seq) - maxlen)]
        else:
            del res[maxlen:]
    elif len(seq) < maxlen:
        if reverse:
            res = [0] * (maxlen - len(seq)) + res
        else:
            res.extend([0] * (maxlen - len(seq)))
    return res
