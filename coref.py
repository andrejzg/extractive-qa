import jsonlines

def flatten(l):
  return [item for sublist in l for item in sublist]

train_lines = jsonlines.open('data/train.english.jsonlines')
train = [x for x in train_lines]

for example in train:
    sentences = flatten(example['sentences'])
    clusters = example['clusters']

    clusters = [sorted(c, key=lambda x: x[0]) for c in clusters]

    for cluster in clusters:
        context = sentences[:]
        first_ref = ' '.join(sentences[cluster[0][0]:cluster[0][1]+1])
        for i, ref_range in enumerate(cluster):
            extracted_ref = ' '.join(sentences[ref_range[0]:ref_range[1]+1])
            print(f'{extracted_ref}\t\t\t{ref_range}')
            context.insert(ref_range[0]+(2*i), '\x1b[6;30;42m')
            context.insert(ref_range[1]+2+(2*i), '\x1b[0m')
        print(f'\033[1m\033[91mQ: Find all mentions of {first_ref} \033[0m')
        print(' '.join(context))
        print('\n\n')

        import code
        code.interact(local=locals())



