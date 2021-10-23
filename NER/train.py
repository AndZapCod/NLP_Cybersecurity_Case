import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from bilstmcrf import BiLSTMCRF
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
embedding_size = 200
dropout = 0.1

model = BiLSTMCRF(
    input_size=input_size,
    hidden_size=hidden_size,
    num_clasess=num_clasess,
    padtok_idx=padtok_idx,
    embedding_size=embedding_size,
    dropout=dropout
).to(device)

criterion = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('Training...')
train(
    model,
    device,
    optimizer=optimizer,
    loss=criterion,
    train_loader=train_loader,
    dev_loader=dev_loader,
    epochs=num_epoch,
    mask=padtok_idx
)
print('Finished training.')
