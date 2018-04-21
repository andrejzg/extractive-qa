import spacy
import nltk
import evaluate
import logging

from collections import defaultdict
from spacy.lang.en import English
from nested_lookup import nested_lookup
from nltk.parse.corenlp import CoreNLPParser
from tqdm import tqdm
from itertools import islice

logging.basicConfig(level=logging.INFO)

nlp = spacy.load('en')
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)
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


def build_vocab_from_json_searches(data, search_keys):
    all_text = [text for key in search_keys for text in nested_lookup(key, data)]
    all_text = ' '.join(all_text)

    tokenizer = English().Defaults.create_tokenizer(nlp)
    words = set()

    for word in tokenizer(all_text):
        words.add(word.text)

    word2id = defaultdict(lambda: 1)  # use 0 for pad, 1 for unk
    for i, word in enumerate(words, start=2):
        word2id[word] = i

    return word2id


def tokenize(text, tokenizer):
    if tokenizer == 'nltk':
        tokens = nltk.word_tokenize(text)
    elif tokenizer == 'spacy':
        tokens = [word.text for word in spacy_tokenizer(text)]
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


def make_examples(data, tokenizer, name=None):
    examples = []
    total = 0
    skipped = 0
    for example in tqdm(data['data'], desc=name):
        title = example['title']

        for paragraph in example['paragraphs']:
            # Extract context
            context = paragraph['context']
            context = fix_quotes(context)
            context_tokens = tokenize(context, tokenizer=tokenizer)

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

                    example = (title, context_tokens, question_tokens, answer_tokens)
                    examples.append(example)

                except (AssertionError, KeyError) as e:
                    skipped += 1
                    continue
    ratio_skipped = skipped/total
    logging.info(f'skipped {skipped}/{total}\t({ratio_skipped})')
    print(skipped)
    print(ratio_skipped)
    return examples


def window(seq, n=2):
    """Return a sliding window (of width n) over data from the iterable"""
    it = iter(enumerate(seq))
    result = tuple(islice(it, n))
    if len(result) == n:
        yield (result[0][0], result[-1][0], [x  [1] for x in result])
    for elem in it:
        result = result[1:] + (elem,)
        yield (result[0][0], result[-1][0], [x[1] for x in result])


def make_spans(seq, max_len=10):
    spans = []
    for span_len in range(1, max_len+1):
        spans.extend(list(window(seq, n=span_len)))
    # now sort the spans
    return sorted(spans, key=lambda x: (x[0], x[1]))
