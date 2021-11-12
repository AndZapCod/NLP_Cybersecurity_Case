import torch
import torch.nn as nn
import numpy as np


class NER(nn.Module):
    def __init__(
            self,
            embedding_size,
            vocab_size,
            tag_size,
            pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device
    ):
        super(NER, self).__init__()

        self.device = device
        self.pad_idx = pad_idx

        # Word Embeddings
        self.word_embedding = nn.Embedding(
            vocab_size, embedding_size, max_norm=1
        )
        self.position_embedding = PositionalEncoding(
            embedding_size,
            dropout,
            max_len
        )

        # Tags Embeddings
        self.tag_embedding = nn.Embedding(
            tag_size, embedding_size, max_norm=1
        )
        self.tag_position_embedding = PositionalEncoding(
            embedding_size,
            dropout,
            max_len - 1
        )

        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout
        )

        self.out = nn.Linear(embedding_size, tag_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, tags):
        # Sequence size:(batch_size, max_length)
        # Tags size: (batch_size, max_length)
        sequence = sequence.to(self.device)
        tags = tags.to(self.device)

        sequence = self.word_embedding(sequence)
        sequence = self.position_embedding(sequence)

        tags = self.tag_embedding(tags)
        tags = self.tag_position_embedding(tags)

        tags_mask = self.sequence_mask(tags.size(1)).to(self.device)
        sequence_padding_mask = self.padding_mask(sequence).to(self.device)
        tags_padding_mask = self.padding_mask(tags).to(self.device)

        # size: (max_size, batch_size, vocab_size)
        sequence = sequence.permute(1, 0, 2)
        # size: (max_size, batch_size, tag_size)
        tags = tags.permute(1, 0, 2)

        transformer_out = self.transformer(
            sequence,
            tags,
            tgt_mask=tags_mask,
            src_key_padding_mask=sequence_padding_mask,
            tgt_key_padding_mask=tags_padding_mask
        )
        out = self.dropout(self.out(transformer_out))
        return out

    @staticmethod
    def sequence_mask(size):
        mask = torch.tril(torch.ones(size, size) * float('-inf'))
        return mask

    def padding_mask(self, batch):
        batch = batch[:, :, 0].squeeze()
        return batch == self.pad_idx


class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-np.log(10000.0) / dim)
        )
        positional_encoding = torch.zeros(max_len, dim)
        # Even position
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # Odd position
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, embedding):
        pos_encoding = self.positional_encoding
        out = embedding + pos_encoding
        return self.dropout(out)


def train(model, optimizer, loss_func, dataloader, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        pred = model(x, y_input)

        pred = pred.permute(1, 2, 0)
        loss = loss_func(pred, y_expected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validate(model, loss_func, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0], batch[1]

            x = x.to(device)
            y = y.to(device)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            pred = model(x, y_input)

            pred = pred.permute(1, 2, 0)
            loss = loss_func(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(
        model,
        optimizer,
        loss_func,
        train_dataloader,
        dev_dataloader,
        epochs,
        device
):
    train_loss_history, validation_loss_history = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train(
            model,
            optimizer,
            loss_func,
            train_dataloader,
            device
        )

        if epoch % 2 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': train_loss
                },
                'checkpoint.pth'
            )

        validation_loss = validate(
            model,
            loss_func,
            dev_dataloader,
            device
        )

        train_loss_history.append(train_loss)
        validation_loss_history.append(validation_loss)

        print(f'Training loss: {train_loss:.4f}')
        print(f'Validation loss: {validation_loss:.4f}')
        print('\n')

    return train_loss_history, validation_loss_history
