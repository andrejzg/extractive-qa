import spacy

from collections import defaultdict
from spacy.lang.en import English
from nested_lookup import nested_lookup

nlp = spacy.load('en')


def token_idx_map(context, context_tokens):
    """
    Add description
    """
    seen = ''
    current_token_idx = 0
    token_map = dict()
    for char_idx, char in enumerate(context):
        if char != u' ':
            seen += char
            context_token = context_tokens[current_token_idx]
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
