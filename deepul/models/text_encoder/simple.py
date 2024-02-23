class CharEncoder:
    def __init__(self, text_data):
        all_chars = []
        for text in text_data:
            all_chars.extend(text)
        all_chars = list(set(all_chars))

        self.encoder = dict(zip(all_chars, range(len(all_chars))))
        self.encoder['<bos>'] = len(all_chars)
        self.encoder['<eos>'] = len(all_chars)+1

        self.decoder = dict(zip(range(len(all_chars)), all_chars))
        self.decoder[len(all_chars)] = '<bos>'
        self.decoder[len(all_chars)+1] = '<eos>'

    def encode(self, text):
        return (
            [self.encoder['<bos>']] + 
            [self.encoder[char] for char in text] + 
            [self.encoder['<eos>']]
        )

    def decode(self, inds):
        out = []
        for ind in inds:
            char = self.decoder[ind]
            if char == '<bos>':
                out = []
            elif char == '<eos>':
                break
            else:
                out.append(char)
        return "".join(out)

    def __len__(self):
        return len(self.encoder)


class WordEncoder:
    def __init__(self, text_data):
        all_words = []
        for text in text_data:
            all_words.extend(text.split(" "))
        all_words = list(set(all_words))

        self.encoder = dict(zip(all_words, range(len(all_words))))
        self.encoder['<bos>'] = len(all_words)
        self.encoder['<eos>'] = len(all_words)+1

        self.decoder = dict(zip(range(len(all_words)), all_words))
        self.decoder[len(all_words)] = '<bos>'
        self.decoder[len(all_words)+1] = '<eos>'

    def encode(self, text):
        return (
            [self.encoder['<bos>']] + 
            [self.encoder[word] for word in text.split(" ")] + 
            [self.encoder['<eos>']]
        )

    def decode(self, inds):
        out = []
        for ind in inds:
            word = self.decoder[ind]
            if word == '<bos>':
                out = []
            elif word == '<eos>':
                break
            else:
                out.append(word)
        return " ".join(out)

    def __len__(self):
        return len(self.encoder)
