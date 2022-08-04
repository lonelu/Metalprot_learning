"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for model training and hyperparameter optimization.
"""

#imports
import os
import json
import numpy as np
import torch
from Metalprot_learning.train import datasets, models

def load_data(features_file: str, path2output: str, partitions: tuple, batch_size: int, seed: int, encodings: bool):
    """
    Loads data for model training.
    :param encodings: boolean that determines whether or not sequence encodings are included during model training.
    """
    train_set, test_set, val_set = datasets.split_data(features_file, path2output, partitions, seed)
    train_dataloader = torch.utils.data.DataLoader(datasets.ImageSet(train_set, encodings), batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(datasets.ImageSet(test_set, encodings), batch_size=batch_size, shuffle=False)
    validation_dataloader = torch.utils.data.DataLoader(datasets.ImageSet(val_set, encodings), batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, validation_dataloader

def configure_model(config: dict):
    return models.AlphafoldNet(config)

def train_loop(model, train_dataloader, loss_fn, optimizer, device):
    """
    Runs a single epoch of model training.
    """
    model.train() #set model to train mode
    running_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        
        #make prediction
        prediction = model.forward(X) 
        loss = loss_fn(prediction, y)

        #backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    running_loss /= len(train_dataloader)
    return running_loss

def validation_loop(model, test_dataloader, loss_fn, device):
    """
    Computes a forward pass of the testing dataset through the network and the resultant test loss.
    """
    model.eval() #set model to evaluation mode

    vloss = 0
    with torch.no_grad():
        for X,y in test_dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            vloss += loss_fn(prediction,y).item()
    vloss /= len(test_dataloader)
    return vloss

def train_model(path2output: str, config: dict, features_file: str):
    """
    Main function for running model training.
    :param config: dictionary defining model hyperparameters for a given training run.
    """

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    #instantiate model
    model = configure_model(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(f'CUDA available? {torch.cuda.is_available()}')
    print(f'Model on GPU? {next(model.parameters()).is_cuda}')

    #instantiate dataloader objects for train and test sets
    train_loader, test_loader, val_loader = load_data(features_file, path2output, (0.8,0.1,0.1), config['batch_size'], config['seed'], config['encodings'])

    #define optimizer and loss function
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad],lr=config['lr'])    
    criterion = torch.nn.L1Loss()

    train_loss = np.array([])
    test_loss = np.array([])
    model = model.float()
    validation_loss = np.array([])
    for epoch in range(0, config['epochs']):
        _train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        _test_loss = validation_loop(model, test_loader, criterion, device)
        _validation_loss = validation_loop(model, val_loader, criterion, device)

        train_loss = np.append(train_loss, _train_loss)
        test_loss = np.append(test_loss, _test_loss)
        validation_loss = np.append(validation_loss, _validation_loss)

        print(f'Train Loss for Epoch {epoch}: {_train_loss}')
        print(f'Test Loss for Epoch {epoch}: {_test_loss}')
        print(f'Val Loss for Epoch {epoch}: {_validation_loss}')
        print('')

    np.save(os.path.join(path2output, 'train_loss.npy'), train_loss)
    np.save(os.path.join(path2output, 'test_loss.npy'), test_loss)
    np.save(os.path.join(path2output, 'validation_loss.npy'), validation_loss)

    torch.save(model.state_dict(), os.path.join(path2output, 'model.pth'))
    with open(os.path.join(path2output, 'config.json'), 'w') as f:
        json.dump(config, f)