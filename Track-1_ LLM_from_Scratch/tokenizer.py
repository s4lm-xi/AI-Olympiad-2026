from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


class MiniTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def save(self, path="tokenizer.json"):
        self._tok.save(path)

    @classmethod
    def load(cls, path="tokenizer.json"):
        return cls(Tokenizer.from_file(path))


def create_tokenizer(file="dataset.txt", vocab_size=5000, special_tokens=None, save_path="tokenizer.json"):
    special_tokens = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )
    tok.train([file], trainer=trainer)
    tok.save(save_path)

    return MiniTokenizer(tok)


def load_tokenizer(path="tokenizer.json"):
    return MiniTokenizer.load(path)


def encode(tokenizer, text):
    return tokenizer.encode(text)


def decode(tokenizer, ids):
    return tokenizer.decode(ids)
