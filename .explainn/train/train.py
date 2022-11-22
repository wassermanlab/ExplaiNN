# =============================================================================
# IMPORTS
# =============================================================================
import torch
import copy

# =============================================================================
# FUNCTIONS
# =============================================================================

def train_explainn(train_loader, test_loader, model, device, criterion, optimizer, num_epochs,
                          weights_folder, name_ind, verbose, trim_weights, checkpoint=1):
    """
    Function to train the ExplaiNN model

    :param train_loader: pytorch DataLoader, train data
    :param test_loader: pytorch DataLoader, validation data
    :param model: ExplaiNN model
    :param device: current available device ('cuda:0' or 'cpu')
    :param criterion: objective (loss) function to use (e.g. MSELoss)
    :param optimizer: pytorch Optimizer (e.g. SGD)
    :param num_epochs: int, number of epochs to train the model
    :param weights_folder: string, folder where to save checkpoints
    :param name_ind: string, suffix name of the checkpoints
    :param verbose: boolean, if False, does not print the progress
    :param trim_weights: boolean, if True, makes output layer weights non negative
    :param checkpoint: int, how often to save checkpoints. For example, 1 means that every model is saved
    :return: tuple:
                    trained ExplaiNN model,
                    list, train losses,
                    list, test losses
    """

    train_error = []
    test_error = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_valid = float('inf')
    best_epoch = 1

    for epoch in range(1,num_epochs+1):

        running_loss = 0.0

        model.train()
        for seqs, labels in train_loader:
            x = seqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, labels)

            # Backward and optimize
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

        if epoch % checkpoint == 0:
            model.load_state_dict(best_model_wts)
            torch.save(best_model_wts, weights_folder + "/"+"model_epoch_"+str(epoch)+"_"+
                          name_ind+".pth")

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, weights_folder + "/" + "model_epoch_best_" + str(best_epoch) + "_" +
               name_ind + ".pth")  # weights_folder, name_ind

    return model, train_error, test_error