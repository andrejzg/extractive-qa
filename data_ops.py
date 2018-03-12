import spacy
import nltk

from collections import defaultdict
from spacy.lang.en import English
from nested_lookup import nested_lookup
from nltk.parse.corenlp import CoreNLPParser

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
    current_token_idx = 0
    token_map = dict()
    for char_idx, char in enumerate(text):
        if char != u' ':
            seen += char
            context_token = tokens[current_token_idx]
            if seen == context_token:
                syn_start = char_idx - len(seen) + 1
                token_map[syn_start] = [seen, current_token_idx]
                seen = ''
                current_token_idx += 1
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

