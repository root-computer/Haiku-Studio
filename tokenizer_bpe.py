from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    @classmethod
    def train_from_iterator(
        cls,
        iterator,
        vocab_size=50000,
        min_freq=2,
        special_tokens=("<unk>", "<bos>", "<eos>")
    ):
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=True,
            use_regex=True
        )

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=list(special_tokens),
            limit_alphabet=268,
            show_progress=True,
        )

        tokenizer.train_from_iterator(iterator, trainer=trainer)
        tokenizer.decoder = ByteLevelDecoder()

        return cls(tokenizer)

    def encode(self, text: str, add_special=True):
        ids = self.tokenizer.encode(text).ids
        if add_special and self.bos_id is not None and self.eos_id is not None:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        if self.bos_id is not None:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id)]
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        tok = Tokenizer.from_file(path)
        return cls(tok)