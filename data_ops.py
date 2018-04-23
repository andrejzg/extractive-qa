# import spacy
import nltk
import evaluate
import logging
import os
import importlib.util

import numpy as np

from collections import defaultdict
# from spacy.lang.en import English
from nested_lookup import nested_lookup
from nltk.parse.corenlp import CoreNLPParser
from tqdm import tqdm
from itertools import islice
from subprocess import call
from itertools import cycle
from itertools import groupby

logging.basicConfig(level=logging.INFO)

BASEDIR = os.path.join(os.path.dirname(__file__), '.')

# nlp = spacy.load('en')
# spacy_tokenizer = English().Defaults.create_tokenizer(nlp)
stanford_tokenizer = CoreNLPParser()


def fix_apostrophe(text, start, end):
    ans = text[:start] + text[start:end].replace("'", '') + text[end:]
    return ans


def token_idx_map(text, tokens):
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


def build_vocab_from_json_searches(data, search_keys, additional_words=None):
    all_text = [text for key in search_keys for text in nested_lookup(key, data)]
    all_text = ' '.join(all_text + additional_words)

    words = set()

    for word in tokenize(all_text, 'nltk'):
        words.add(word)

    word2id = defaultdict(lambda: 1)  # use 0 for pad, 1 for unk
    for i, word in enumerate(words, start=2):
        word2id[word] = i

    return word2id


def tokenize(text, tokenizer):
    if tokenizer == 'nltk':
        tokens = nltk.word_tokenize(text)
    # elif tokenizer == 'spacy':
    #     tokens = [word.text for word in spacy_tokenizer(text)]
    elif tokenizer == 'stanford':
        tokens = list(stanford_tokenizer.tokenize(text))
    else:
        raise ValueError('Tokenizer must be one of the following: nltk|spacy|stanford')
    return tokens


def fix_quotes(text):
    ans = text.replace("''", '" ')
    ans = ans.replace("``", '" ')
    return ans


def fix_whitespace(text):
    return ' '.join([x for x in [x.strip() for x in text.split(' ')] if x != ''])


def make_squad_examples(data,
                        tokenizer,
                        word2id,
                        name=None,
                        max_context_len=300,
                        max_answer_len=10,
                        max_question_len=20,
                        ):
    examples = []
    total = 0
    skipped = 0
    span2position = make_span2position(
        seq_size=max_context_len,
        max_len=max_answer_len
    )

    for line in tqdm(data['data'], desc=name):
        title = line['title']

        for paragraph in line['paragraphs']:
            # Extract context
            context = paragraph['context']
            context = fix_quotes(context)
            context_tokens = tokenize(context, tokenizer=tokenizer)

            if max_context_len and len(context_tokens) > max_context_len:
                skipped += len(paragraph['qas'])
                continue

            answer_map = token_idx_map(context, context_tokens)

            for qa in paragraph['qas']:
                # Extract question
                question = qa['question']
                question = fix_quotes(question)
                question_tokens = tokenize(question, tokenizer=tokenizer)

                # Extract answer
                answer = qa['answers'][0]['text']
                answer = fix_quotes(answer)
                answer_tokens = tokenize(answer, tokenizer=tokenizer)

                if max_answer_len and len(answer_tokens) > max_answer_len:
                    skipped += 1
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

                    total += 1
                    assert extracted_answer == actual_clean, f'{extracted_answer} != {actual_clean}'

                    span_positions = [span2position[(span_start, span_end)]]

                    example = {
                        'title': title,
                        'context_raw': context_tokens,
                        'question_raw': question_tokens,
                        'answer_raw': answer_tokens,
                        'context': pad_seq([word2id[w] for w in context_tokens], maxlen=max_context_len),
                        'question': pad_seq([word2id[w] for w in question_tokens], maxlen=max_question_len),
                        'answer': pad_seq([word2id[w] for w in answer_tokens], maxlen=max_answer_len),
                        'starts': [span_start],
                        'ends': [span_end],
                        'span_positions': span_positions,
                        'label': np.asarray([1 if x in span_positions else 0 for x in span2position.values()])
                    }

                    examples.append(example)

                except (AssertionError, KeyError) as e:
                    skipped += 1
                    continue
    ratio_skipped = skipped/total
    logging.info(f'max_context_len: {max_context_len}')
    logging.info(f'max_answer_len: {max_answer_len}')
    logging.info(f'skipped {skipped}/{total}\t({ratio_skipped})')
    print(skipped)
    print(ratio_skipped)
    return examples


def make_conll_examples(data,
                        word2id,
                        label2question,
                        name=None,
                        max_context_len=300,
                        max_answer_len=10,
                        max_question_len=20):

    span2position = make_span2position(
        seq_size=max_context_len,
        max_len=max_answer_len
    )

    examples = []

    for line in tqdm(data, desc=name):
        words = [x[0] for x in line]
        labels = [x[1] for x in line]

        for label in label2question.keys():
            if label in labels:
                context = words
                if max_context_len and len(context) > max_context_len:
                    continue

                question = label2question[label]
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

                span_positions = [span2position[(s, e)] for s,e in zip(span_starts, span_ends)]

                answers = [context[s:e+1] for s, e in zip(span_starts, span_ends)]

                example = {
                    'title': '',
                    'context_raw': context,
                    'question_raw': question,
                    'answer_raw': answers,
                    'context': pad_seq([word2id[w] for w in context], maxlen=max_context_len),
                    'question': pad_seq([word2id[w] for w in question], maxlen=max_question_len),
                    'answer': pad_seq([word2id[w] for answer in answers for w in answer], maxlen=max_answer_len),
                    'starts': span_starts,
                    'ends': span_ends,
                    'span_positions': span_positions,
                    'label': np.asarray([1 if x in span_positions else 0 for x in span2position.values()])
                }

                examples.append(example)

    return examples


def window(seq, n=2):
    """Return a sliding window (of width n) over data from the iterable"""
    it = iter(enumerate(seq))
    result = tuple(islice(it, n))
    if len(result) == n:
        yield (result[0][0], result[-1][0], [x[1] for x in result])
    for elem in it:
        result = result[1:] + (elem,)
        yield (result[0][0], result[-1][0], [x[1] for x in result])


def make_spans(seq, max_len=10):
    spans = []
    for span_len in range(1, max_len+1):
        spans.extend(list(window(seq, n=span_len)))
    # now sort the spans
    return sorted(spans, key=lambda x: (x[0], x[1]))


def glove_embeddings(embedding_size, emb_path=None, script_path=None):
    emb_path = emb_path if emb_path else 'data/glove/glove.6B.{0}d.txt'.format(embedding_size)

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


def make_glove_embedding_matrix(word2id, embedding_size, emb_path=None, script_path=None):
    glove_embs = glove_embeddings(
        embedding_size,
        emb_path,
        script_path
    )

    embedding_matrix = np.zeros((len(word2id) + 2, embedding_size))  # +1 for
    unk = np.zeros(embedding_size)
    embedding_matrix[0] = unk  # pad
    embedding_matrix[1] = unk  # unk

    for word, id in word2id.items():
        if word in glove_embs:
            vec = glove_embs[word]
        elif word.lower() in glove_embs:
            vec = glove_embs[word.lower()]
        else:
            vec = unk

        embedding_matrix[id] = vec

    return embedding_matrix


def vectorize_tokens(tokens, word2vec, embedding_size):
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
    spec = importlib.util.spec_from_file_location('', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_span2position(seq_size, max_len):
    seq = [0] * seq_size
    spans = make_spans(seq, max_len)
    span2position = {}
    for i, span in enumerate(spans):
        span2position[(span[0], span[1])] = i
    return span2position


def pad_seq(seq, maxlen, reverse=False):
    """ Pad or shorten a list of items """
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


def make_batcher(seq, batch_size):
    seq_iterator = cycle(seq)

    def batcher():
        if batch_size:
            batch = [next(seq_iterator) for _ in range(batch_size)]
        else:
            return seq
        return batch

    return batcher


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
