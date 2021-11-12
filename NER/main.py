import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformer import NER
from transformer import fit
from dataset import Corpus, ToPaddedTensor
from utils import train_val_test_split

PADDING_IDX = 0

print('Configuring device...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device set as {device}')
num_epochs = 1
batch_size = 128
learning_rate = 0.001

print('Loading dataset...')
data = Corpus('data/corpus.pkl')
transform = ToPaddedTensor(data.max_sent_len + 2, PADDING_IDX)
data.transform = transform
print('Splitting dataset...')
train_sampler, dev_sampler, test_sampler = train_val_test_split(
    data, val_split=0.1, test_split=0.1,
    random_seed=43
)

train_loader = DataLoader(
    data,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=1,
)

dev_loader = DataLoader(
    data,
    batch_size=batch_size,
    sampler=dev_sampler
)

test_loader = DataLoader(
    data,
    batch_size=batch_size,
    sampler=test_sampler
)

input_size = len(transform.idx2token)
hidden_size = 64
num_clasess = len(transform.idx2tag)
padtok_idx = PADDING_IDX
embedding_size = 296
dropout = 0.1

model = NER(
    embedding_size=embedding_size,
    vocab_size=input_size,
    tag_size=num_clasess,
    pad_idx=padtok_idx,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dropout=0.1,
    max_len=223,
    forward_expansion=4,
    device=device
).to(device)

loss_func = nn.CrossEntropyLoss(ignore_index=padtok_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss, val_loss = fit(model, optimizer, loss_func, train_loader,
                           dev_loader, num_epochs, device)
