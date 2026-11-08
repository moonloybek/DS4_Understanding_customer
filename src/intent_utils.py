import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

FALLBACK = 'Fallback Intent'


def simple_tokenize(text):
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9]", str(text).lower())


class Vocab:
    def __init__(self, texts):
        counts = Counter(t for s in texts for t in simple_tokenize(s))
        self.itos = ['<pad>', '<unk>'] + sorted(counts)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text, max_len=32):
        ids = [self.stoi.get(t, 1) for t in simple_tokenize(text)][:max_len]
        if not ids:
            ids = [1]
        return ids + [0] * (max_len - len(ids))

    def __len__(self):
        return len(self.itos)


class RNNNet(nn.Module):
    # embedding -> (bi)RNN or (bi)LSTM -> mean pooling over real tokens -> linear
    def __init__(self, vocab_size, n_classes, cell='lstm', emb=64, hid=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        rnn_cls = nn.LSTM if cell == 'lstm' else nn.RNN
        self.rnn = rnn_cls(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid * 2, n_classes)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(-1).float()[:, :out.size(1)]
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(pooled))


class IntentClassifier:
    # picklable wrapper: predict(texts) -> list of intent names, with Fallback Intent below `threshold`
    def __init__(self, kind, model, classes, vocab=None, tokenizer=None, threshold=0.0, max_len=32):
        self.kind, self.model, self.classes = kind, model, list(classes)
        self.vocab, self.tokenizer = vocab, tokenizer
        self.threshold, self.max_len = threshold, max_len

    @torch.no_grad()
    def predict_proba(self, texts):
        texts = [str(t) for t in texts]
        self.model.cpu().eval()
        if self.kind == 'bert':
            enc = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors='pt')
            logits = self.model(input_ids=enc['input_ids'],
                                attention_mask=enc['attention_mask']).logits
        else:
            x = torch.tensor([self.vocab.encode(t, self.max_len) for t in texts])
            logits = self.model(x)
        return torch.softmax(logits, dim=-1).numpy()

    def predict(self, texts, use_fallback=True):
        p = self.predict_proba(texts)
        idx, conf = p.argmax(1), p.max(1)
        return [self.classes[i] if (conf_i >= self.threshold or not use_fallback) else FALLBACK
                for i, conf_i in zip(idx, conf)]
