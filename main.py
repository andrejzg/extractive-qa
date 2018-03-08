import json
import spacy
import data_ops
import evaluate

from tqdm import tqdm
from collections import defaultdict
from spacy.lang.en import English

nlp = spacy.load('en')
tokenizer = English().Defaults.create_tokenizer(nlp)

train_raw = json.load(open('data/train-v1.1.json'))
dev_raw = json.load(open('data/dev-v1.1.json'))

try:
    word2id = json.load(open('data/word2id.json', 'rb'))
    word2id = defaultdict(lambda: 1, word2id)  # use 1 for unk 0 for pad
except FileNotFoundError:
    word2id = data_ops.build_vocab_from_json_searches(data=train_raw,
                                                      search_keys=['context', 'question'])
    json.dump(word2id, open('data/word2id.json', 'w'))

skipped = 0

for example in tqdm(train_raw['data']):
    title = example['title']

    for paragraph in example['paragraphs']:

        context = paragraph['context']
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')

        context_tokens = [x.text for x in nlp(context)]
        context_numeric = [word2id[x] for x in context_tokens]

        answer_map = data_ops.token_idx_map(context, context_tokens)

        for qa in paragraph['qas']:
            # Extract question
            question = qa['question']
            question = question.replace("''", '" ')
            question = question.replace("``", '" ')

            question_tokens = [word.text for word in tokenizer(question)]
            question_numeric = [word2id[word] for word in question_tokens]

            # Extract answer
            answer = qa['answers'][0]['text']
            answer = answer.replace("''", '" ')
            answer = answer.replace("``", '" ')

            answer_tokens = [word.text for word in tokenizer(answer)]
            answer_numeric = [word2id[word] for word in answer_tokens]

            answer_start = qa['answers'][0]['answer_start']
            answer_end = answer_start + len(answer)

            try:
                last_word_answer = len(answer_tokens[-1])  # add one to get the first char

                _, start_answer_word_index = answer_map[answer_start]
                _, end_answer_word_index = answer_map[answer_end - last_word_answer]

                extracted_answer = context_tokens[start_answer_word_index:end_answer_word_index + 1]
                extracted_answer = ' '.join(extracted_answer)

                extracted_clean = evaluate.normalize_answer(extracted_answer)
                actual_clean = evaluate.normalize_answer(answer)

                # assert extracted_clean == actual_clean, f'{extracted_clean} =/= {actual_clean}'
                if actual_clean != extracted_clean:
                    import code
                    code.interact(local=locals())
            except:
                skipped += 1
                continue

print(f'skipped: {skipped}')










import code
code.interact(local=locals())