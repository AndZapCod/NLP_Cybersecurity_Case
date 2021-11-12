import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformer import NER
from dataset import Corpus, ToPaddedTensor
from utils import train_val_test_split
from utils import train

PADDING_IDX = 0

print('Configuring device...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device set as {device}')
num_epoch = 20
batch_size = 2048
learning_rate = 0.001

print('Loading dataset...')
data = Corpus('data/corpus.pkl')
transform = ToPaddedTensor(data.max_sent_len, PADDING_IDX)
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
    num_workers=4,
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
embedding_size = 512
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
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=padtok_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    print(f'Epoch {epoch}')

    for i, (x, y) in enumerate(train_loader):
        model.train()
        x = x.to(device)
        y = x.to(device)

        output = model(x, y[:-1])
        output = output.reshape(-1, output.shape[2])
        y = y[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output,y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
