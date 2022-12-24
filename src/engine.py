import torch
import torch.nn as nn


def loss_fn(outputs, targets):
    # we use binary cross-entropy with logits which first
    # applies sigmoid and then calculates the loss
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    # put the model in the training mode
    model.train()

    # loop over all batches
    for data in data_loader:
        # extract ids, token type ids and mask
        # from current batch
        # also etract targets
        ids = data["ids"]
        token_type_ids = data["token_type_ids"]
        mask = data["mask"]
        targets = data["targets"]

        # move everything to a specified device
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # zero-grad the optimizer
        optimizer.zero_grad()

        # pass through the model
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )

        # calculate loss
        loss = loss_fn(outputs, targets)

        # backward step the loss
        loss.backward()

        # step optimizer
        optimizer.step()

        # step scheduler
        scheduler.step()
