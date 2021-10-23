import numpy as np
import torch

from torch.utils.data import SubsetRandomSampler


def train_val_test_split(dataset, val_split, test_split, suffle=True,
                         random_seed=None):
    dataset_size = len(dataset)
    idxs = list(range(dataset_size))
    t_split = int(np.floor(dataset_size * test_split))
    v_split = int(np.floor(dataset_size * val_split))
    if suffle:
        if random_seed:
            np.random.seed(random_seed)
        np.random.shuffle(idxs)

    train_idxs = idxs[t_split + v_split:]
    dev_idxs = idxs[t_split: t_split + v_split]
    test_idxs = idxs[:t_split]

    train_sampler = SubsetRandomSampler(train_idxs)
    dev_sampler = SubsetRandomSampler(dev_idxs)
    test_sampler = SubsetRandomSampler(test_idxs)

    return train_sampler, dev_sampler, test_sampler


def accuracy(model, device, x, y, mask=None):
    model.eval()
    model.to(device)
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)

        pred_prob = model(x)
        pred = pred_prob.argmax(dim=2).view(-1)

        if mask is not None:
            mask_idxs = (y != mask).nonzero()
            correct = pred[mask_idxs].eq(y[mask_idxs])
            return (correct.sum() / y[mask_idxs].shape[0]).item()
        else:
            correct = pred.eq(y)
            return (correct.sum() / y.shape[0]).item()


def train(model, device, optimizer, loss, train_loader, dev_loader,
          epochs=100, mask=None, save_model=True):
    model = model.to(device)
    for epoch in range(epochs):
        train_acc = 0
        for i, (x, y) in enumerate(train_loader):
            model.train()
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            output = output.view(-1, output.shape[-1])
            y = y.view(-1)
            cost = loss(output, y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            train_acc += accuracy(model, device, x, y, mask=mask)

        train_acc = train_acc / len(train_loader)

        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': cost
            },
                'model.pt')

        val_acc = 0
        for i, (x, y) in enumerate(dev_loader):
            y = y.view(-1)
            val_acc += accuracy(model, device, x, y, mask=mask)
        val_acc = val_acc / len(dev_loader)

        print(
            f'Epoch: {epoch}\t',
            f'Train accuracy: {train_acc}\t',
            f'Validation accuracy: {val_acc}'
        )
