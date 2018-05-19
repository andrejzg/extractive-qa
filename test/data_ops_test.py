import data_ops


def test_index_by_starting_character():
    # example_text = 'hello how are you?'
    example_text = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'  # NOQA
    tokens = data_ops.tokenize(example_text, 'nltk')

    token_lookup = data_ops.index_by_starting_character(example_text, tokens)
    for char_index, (token, token_index) in token_lookup.items():
        assert token[0] == example_text[char_index]
        assert tokens[token_index] == token
        
