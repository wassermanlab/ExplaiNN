# =============================================================================
# IMPORTS
# =============================================================================
import copy
import os
import torch
from torch.nn import CosineSimilarity

# =============================================================================
# FUNCTIONS
# =============================================================================

def train_explainn(train_loader, test_loader, model, device, criterion,
                   optimizer, num_epochs=100, weights_folder="./",
                   name_ind=None, verbose=False, trim_weights=False,
                   checkpoint=1, patience=0):
    """
    Function to train the ExplaiNN model

    :param train_loader: pytorch DataLoader, train data
    :param test_loader: pytorch DataLoader, validation data
    :param model: ExplaiNN model
    :param device: current available device ("cuda:0" or "cpu")
    :param criterion: objective (loss) function to use (e.g. MSELoss)
    :param optimizer: pytorch Optimizer (e.g. SGD)
    :param num_epochs: int, number of epochs to train the model
    :param weights_folder: string, folder where to save checkpoints
    :param name_ind: string, suffix name of the checkpoints
    :param verbose: boolean, if False, does not print the progress
    :param trim_weights: boolean, if True, makes output layer weights non-negative
    :param checkpoint: int, how often to save checkpoints (e.g. 1 means that the model will be saved after each epoch;
                       0 that only the best model will be saved)
    :param patience: int, number of epochs to wait before stopping training if validation loss does not improve;
                     if 0, this parameter is ignored
    :return: tuple:
                    trained ExplaiNN model,
                    list, train losses,
                    list, test losses
    """

    train_error = []
    test_error = []

    best_model_wts = copy.deepcopy(model.state_dict())
    # if save_optimizer:
    #     best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_loss_valid = float('inf')
    best_epoch = 1

    for epoch in range(1, num_epochs+1):

        running_loss = 0.0

        model.train()
        for seqs, labels in train_loader:
            x = seqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, labels)

            # backward and optimize
            loss.backward()

            optimizer.step()

            # to clip the weights (constrain them to be non-negative)
            if trim_weights:
                model.final.weight.data.clamp_(0)

            running_loss += loss.item()

        # save training loss to file
        epoch_loss = running_loss / len(train_loader)
        train_error.append(epoch_loss)

        # calculate test (validation) loss for epoch
        test_loss = 0.0

        with torch.no_grad():  # we don't train and don't save gradients here
            model.eval()  # we set forward module to change dropout and batch normalization techniques
            for seqs, labels in test_loader:
                x = seqs.to(device)
                y = labels.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()

        test_loss = test_loss / len(test_loader)
        test_error.append(test_loss)

        if verbose:
            print('Epoch [{}], Current Train Loss: {:.5f}, Current Val Loss: {:.5f}'
                  .format(epoch, epoch_loss, test_loss))

        if test_loss < best_loss_valid:
            best_loss_valid = test_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            # if save_optimizer:
            #     best_optimizer_wts = copy.deepcopy(optimizer.state_dict())

        if checkpoint:
            if epoch % checkpoint == 0:
                model.load_state_dict(best_model_wts)
                if name_ind:
                    f = f"model_epoch_{epoch}_{name_ind}.pth"
                else:
                    f = f"model_epoch_{epoch}.pth"
                torch.save(best_model_wts, os.path.join(weights_folder, f))
                # if save_optimizer:
                #     optimizer.load_state_dict(best_optimizer_wts)
                #     if name_ind:
                #         f = f"optimizer_epoch_{epoch}_{name_ind}.pth"
                #     else:
                #         f = f"optimizer_epoch_{epoch}.pth"
                #     torch.save(best_optimizer_wts, os.path.join(weights_folder, f))

        if patience:
            if epoch >= best_epoch + patience:  # at last, we lost our patience!
                if verbose:
                    print('Early stopping, Current Epoch {}, Best Epoch: {}, Patience: {}'
                        .format(epoch, best_epoch, patience))
                break

    model.load_state_dict(best_model_wts)
    if name_ind:
        f = f"model_epoch_best_{best_epoch}_{name_ind}.pth"
    else:
        f = f"model_epoch_best_{best_epoch}.pth"
    torch.save(best_model_wts, os.path.join(weights_folder, f))
    # if save_optimizer:
    #     optimizer.load_state_dict(best_optimizer_wts)
    #     if name_ind:
    #         f = f"optimizer_epoch_best_{best_epoch}_{name_ind}.pth"
    #     else:
    #         f = f"optimizer_epoch_best_{best_epoch}.pth"
    #     torch.save(best_optimizer_wts, os.path.join(weights_folder, f))

    return model, train_error, test_error
