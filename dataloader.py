import torch
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator
import string
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class BaseDataset(Dataset):
    def __init__(self, file_path: str, column: str, add_sos_eos: bool=False):
        self.dataframe = pd.read_csv(file_path)
        self.sentences = self.dataframe[column].head(100)
        self.vocabs = build_vocab_from_iterator(
            self.token_genarator(self.sentences)
        )
        self.add_sos_eos = add_sos_eos
        if self.add_sos_eos == True:
            extra_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
            for token in extra_tokens:
                self.vocabs.append_token(token)
        else:
            extra_tokens = ["<PAD>", "<UNK>"]
            for token in extra_tokens:
                self.vocabs.append_token(token)

    def token_genarator(self, sentences: str):
        for text in sentences:
            clean_text = "".join(
                [word for word in text
                 if word not in string.punctuation]
            )
            tokens = word_tokenize(clean_text)
            yield tokens

    def text_to_sequences(self, sentences: str) -> list[int]:
        sequence = [
            self.vocabs[token] if token in self.vocabs
            else self.vocabs["<UNK>"]
            for token in word_tokenize(sentences)
        ]
        if self.add_sos_eos == True:
            sequence = [self.vocabs["<SOS>"]] + sequence + [self.vocabs["<EOS>"]]
        
        return sequence

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index) -> torch.Tensor:
        item = self.sentences[index]
        sequence = self.text_to_sequences(item)
        return torch.tensor(sequence)


class CombineDataset(Dataset):
    def __init__(self, data_path: str, source_column: str, target_column: str):
        self.source = BaseDataset(
            file_path=data_path,
            column=source_column,
            add_sos_eos=True
        )
        self.target = BaseDataset(
            file_path=data_path,
            column=target_column,
            add_sos_eos=True
        )

    @staticmethod
    def collate_fn(batch) -> torch.Tensor:
        # en, bn = zip(batch)
        source = [item[0] for item in batch]
        target = [item[1] for item in batch]
        source_padded = pad_sequence(source, padding_value=0, batch_first=False)
        target_padded = pad_sequence(target, padding_value=0, batch_first=False)
        return source_padded, target_padded

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index) -> torch.Tensor:
        source_item = self.source[index]
        target_item = self.target[index]
        return source_item, target_item


if __name__ == "__main__":
    dataset = CombineDataset(
        data_path="Sample Data/english to bengali.csv",
        source_column="english_caption",
        target_column="bengali_caption"
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=32,
        collate_fn=dataset.collate_fn
    )

    english, bangla = next(iter(dataloader))
    print(english.shape, bangla.shape)
    print(f"Source Vocabs: {len(dataset.target.vocabs)}, Target Vocabs: {len(dataset.target.vocabs)}")
