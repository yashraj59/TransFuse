import torch
import torch.utils.data
import torch.nn as nn
from torch._C import device
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss, confusion_matrix
)

import math
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from model import (MultiOmicsLayer, TransferModuleBase, 
                   TransferModuleSNP2Layer, TransferModuleGene2Layer, 
                   TransferModuleProtein2Layer, TransFuse)

from typing import ( Callable, Iterator, Iterable, Optional, 
                    Tuple, List, Dict, Any)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(x: torch.Tensor, y: torch.Tensor) -> tuple:
    """
    Preprocess input and target tensors by converting their data types and 
    moving them to the appropriate device.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Target tensor.

    Returns:
        tuple: Processed input and target tensors.
    """
    return x.float().to(device), y.int().reshape(-1, 1).to(device)
    

class WrappedDataLoader:
    """
    WrappedDataLoader is a utility class that wraps around a DataLoader to 
    apply a transformation function to each batch of data.

    Args:
        dl (Iterable): The data loader to be wrapped.
        func (Callable): The function to be applied to each batch of data.
    """
    
    def __init__(self, dl: Iterable, func: Callable) -> None:
        self.dl = dl
        self.func = func

    def __len__(self) -> int:
        """
        Returns the number of batches in the data loader.
        
        Returns:
            int: Number of batches.
        """
        return len(self.dl)

    def __iter__(self) -> Iterator:
        """
        Returns an iterator over the batches of the data loader,
        with the transformation function applied to each batch.
        
        Returns:
            Iterator: Iterator over transformed batches.
        """
        for batch in iter(self.dl):
            yield self.func(*batch)
            
def loss_batch_pretrain(model: nn.Module, loss_fn: Callable, L1REG: float, 
                        xb: torch.Tensor, yb: torch.Tensor, ythresh: float, 
                        opt: Optional[torch.optim.Optimizer] = None) \
                        -> Tuple[float, float]:
    """
    Compute the loss and accuracy for a batch of data.

    Args:
        model (nn.Module): The neural network model.
        loss_fn (Callable): The loss function.
        L1REG (float): L1 regularization coefficient.
        xb (torch.Tensor): Input batch tensor.
        yb (torch.Tensor): Target batch tensor.
        ythresh (float): Threshold for converting probabilities to class labels
        opt (torch.optim.Optimizer, optional): Optimizer for updating model
                                               parameters.

    Returns:
        float: The computed loss for the batch.
        float: The accuracy for the batch.
    """
    # Forward pass
    yhat = model(xb)
    
    # Compute the loss
    loss = loss_fn(yhat, yb.float())
    loss += L1REG * sum(torch.sum(torch.abs(param))
                    for param in model.SingleOmic.parameters())

    # Perform backpropagation and optimization if an optimizer is provided
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Detach yhat from the computation graph, move to CPU,
    # and convert to a NumPy array.
    # detach(): Removes tensor from the computation graph
    #           (no gradient computation).
    # cpu(): Transfers the tensor from GPU to CPU (if necessary).
    # numpy(): Converts the tensor to a NumPy array for CPU-based processing
    #          (e.g., accuracy calculation).
    yhat_class = np.where(yhat.detach().cpu().numpy() < ythresh, 0, 1)
    accuracy = accuracy_score(yb.detach().cpu().numpy(), yhat_class)

    # Return the scalar value of the `loss` tensor.
    # .item(): Converts a single-element tensor to a Python scalar (float/int)
    # and detaches it from the computation graph, reducing memory usage
    # and computational overhead.
    return loss.item(), accuracy


def fit_pretrain(epochs: int, model: nn.Module, loss_fn: Callable, 
                 L1REG: float, ythresh: float, 
                 opt: torch.optim.Optimizer, train_dl: DataLoader, 
                 val_dl: DataLoader) \
                 -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train and validate the model.

    Args:
        epochs (int): Number of epochs to train.
        model (nn.Module): The neural network model.
        loss_fn (Callable): The loss function.
        L1REG (float): L1 regularization coefficient.
        ythresh (float): Threshold for converting probabilities to class labels
        opt (torch.optim.Optimizer): Optimizer for updating model parameters.
        train_dl (DataLoader): DataLoader for the training set.
        val_dl (DataLoader): DataLoader for the validation set.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
        - Training losses over epochs.
        - Training accuracies over epochs.
        - Validation losses over epochs.
        - Validation accuracies over epochs.
    """
    train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_results = [
            loss_batch_pretrain(model, loss_fn, L1REG, xb, yb, ythresh, opt)
            for xb, yb in train_dl]
        # zip(*train_results): Unpacks the list of tuples, transposing the data
        # so that we get separate tuples for all losses and all accuracies.
        losses, accuracies = zip(*train_results)
        train_loss.append(np.mean(losses))
        train_accuracy.append(np.mean(accuracies))

        model.eval()
        with torch.no_grad():
            val_results = [
                loss_batch_pretrain(model, loss_fn, L1REG, xb, yb, ythresh)
                for xb, yb in val_dl]
            losses, accuracies = zip(*val_results)
        val_loss.append(np.mean(losses))
        val_accuracy.append(np.mean(accuracies))
        
    return train_loss, train_accuracy, val_loss, val_accuracy


def loss_batch(model: nn.Module, loss_fn: Callable, L1REG: float, 
               xb: torch.Tensor, yb: torch.Tensor, ythresh: float, 
               opt: Optional[torch.optim.Optimizer] = None) \
               -> Tuple[float, float]:
    """
    Compute the loss and accuracy for a batch of data, including L1 
    regularization.

    Args:
        model (nn.Module): The neural network model.
        loss_fn (Callable): The loss function.
        L1REG (float): L1 regularization coefficient.
        xb (torch.Tensor): Input batch tensor.
        yb (torch.Tensor): Target batch tensor.
        ythresh (float): Threshold for converting probabilities to class labels
        opt (torch.optim.Optimizer, optional): Optimizer for updating model
                                               parameters.

    Returns:
        float: The computed loss for the batch.
        float: The accuracy for the batch.
    """
    
    yhat = model(xb)
    loss = loss_fn(yhat, yb.float())
    # Apply L1 regularization to specific model parameters
    for param_snp in model.LayerSnp2gen.parameters():
          loss += L1REG * torch.sum(torch.abs(param_snp))
    for param_gen in model.LayerGen2pro.parameters():
          loss += L1REG * torch.sum(torch.abs(param_gen))
    for param_pro in model.LayerPro2pro.parameters():
          loss += L1REG * torch.sum(torch.abs(param_pro))

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Detach yhat from the computation graph, move to CPU,
    # and convert to a NumPy array.
    # detach(): Removes tensor from the computation graph
    #           (no gradient computation).
    # cpu(): Transfers the tensor from GPU to CPU (if necessary).
    # numpy(): Converts the tensor to a NumPy array for CPU-based processing
    #          (e.g., accuracy calculation).
    yhat_class = np.where(yhat.detach().cpu().numpy()<ythresh, 0, 1)
    accuracy = accuracy_score(yb.detach().cpu().numpy(), yhat_class)

    # Return the scalar value of the `loss` tensor.
    # .item(): Converts a single-element tensor to a Python scalar (float/int)
    # and detaches it from the computation graph, reducing memory usage
    # and computational overhead.
    return loss.item(), accuracy


def fit(epochs: int, model: nn.Module, loss_fn: Callable, 
        L1REG: float, ythresh: float, 
        opt: torch.optim.Optimizer, train_dl: DataLoader, 
        val_dl: DataLoader) \
        -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train and validate the model.

    Args:
        epochs (int): Number of epochs to train.
        model (nn.Module): The neural network model.
        loss_fn (Callable): The loss function.
        L1REG (float): L1 regularization coefficient.
        ythresh (float): Threshold for converting probabilities to class labels
        opt (torch.optim.Optimizer): Optimizer for updating model parameters.
        train_dl (DataLoader): DataLoader for the training set.
        val_dl (DataLoader): DataLoader for the validation set.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
        - Training losses over epochs.
        - Training accuracies over epochs.
        - Validation losses over epochs.
        - Validation accuracies over epochs.
    """
    
    train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_results =[loss_batch(model, loss_fn, L1REG, xb, yb, ythresh, opt)
                          for xb, yb in train_dl]
        # zip(*train_results): Unpacks the list of tuples, transposing the data
        # so that we get separate tuples for all losses and all accuracies.
        losses, accuracies = zip(*train_results)
        train_loss.append(np.mean(losses))
        train_accuracy.append(np.mean(accuracies))

        model.eval()
        with torch.no_grad():
            val_results = [loss_batch(model, loss_fn, L1REG, xb, yb, ythresh)
                           for xb, yb in val_dl]
            losses, accuracies = zip(*val_results)
        val_loss.append(np.mean(losses))
        val_accuracy.append(np.mean(accuracies))

    return train_loss, train_accuracy, val_loss, val_accuracy


def reset_weights(m: nn.Module) -> None:
    """
    Reset model weights to avoid weight leakage.

    Args:
        m (nn.Module): The neural network model or layer.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # Reset trainable parameters of the layer
            layer.reset_parameters()
            
def pretraining(train_X: pd.DataFrame, train_y: pd.DataFrame, 
                valid_X: pd.DataFrame, valid_y: pd.DataFrame,
                model_class: Callable, adj_matrix: torch.Tensor,
                in_features: int, out_features: int,
                H1: int, H2: Optional[int], D_out: int, dropout_rate: float,
                LR: float, L1REG: float, L2REG: float, 
                betas_range: Tuple[float, float], ythresh: float, 
                BS: int, epochs: int, device: torch.device) \
                -> Tuple[nn.Module, dict, float, float]:
    """
    Pretrain a model on training data and evaluate it on validation data.

    Parameters:
        train_X (pd.DataFrame): Training input data.
        train_y (pd.DataFrame): Training target data.
        valid_X (pd.DataFrame): Validation input data.
        valid_y (pd.DataFrame): Validation target data.
        model_class (Callable): Class of the model to be instantiated.
        adj_matrix (torch.Tensor): Adjacency matrix.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        H1 (int): Size of the first hidden layer.
        H2 (Optional[int]): Size of the second hidden layer (optional).
        D_out (int): Size of the output layer.
        dropout_rate (float): Dropout rate.
        LR (float): Learning rate.
        L1REG (float): L1 regularization coefficient.
        L2REG (float): L2 regularization coefficient.
        betas_range (Tuple[float, float]): Betas for the Adam optimizer.
        ythresh (float): Threshold for converting probabilities to class labels
        BS (int): Batch size.
        epochs (int): Number of epochs.
        device (torch.device): Device to run the model on.

    Returns:
        nn.Module: The trained model.
        dict: State dictionary of the pretrained model weights.
        float: Loss on the validation set.
        float: Accuracy on the validation set.
    """

    train_X = np.array(train_X)
    valid_X = np.array(valid_X)
    train_y = np.array(train_y)
    valid_y = np.array(valid_y)

    # Convert to PyTorch tensors
    train_X, train_y, valid_X_t, valid_y_t = map(torch.tensor,
                                  (train_X, train_y, valid_X, valid_y))

    # Create DataLoaders
    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(valid_X_t, valid_y_t)
    
    train_dl = DataLoader(dataset=train_ds, batch_size=BS)
    val_dl = DataLoader(dataset=val_ds)

    # Wrap DataLoaders
    train_dl = WrappedDataLoader(train_dl, preprocess)
    val_dl = WrappedDataLoader(val_dl, preprocess)

    # Initialize model and optimizer
    model = model_class(adj_matrix,
                        in_features, out_features,
                        H1, H2,
                        D_out, dropout_rate).to(device)
    model.apply(reset_weights)

    # Using Binary Cross-Entropy Loss
    # BCELoss provides stable gradients during training,
    # especially when combined with sigmoid activation in the output layer.
    loss_fn = nn.BCELoss()

    opt = torch.optim.Adam(model.parameters(),
                           lr=LR,
                           weight_decay=L2REG,
                           betas=betas_range,
                           amsgrad=True)

    # Train the model
    train_loss, train_accuracy, val_loss, val_accuracy = fit_pretrain(
        epochs, model, loss_fn, L1REG, ythresh, opt, train_dl, val_dl)

    # save the pretrained weight for transfer learning
    pretrain_weights = model.SingleOmic.state_dict()

    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=(8,4))
    ax[0].plot(train_loss)
    ax[0].plot(val_loss)
    ax[0].set_ylabel('Loss')
    ax[0].set_title("Pretrain Loss")
    ax[1].plot(train_accuracy)
    ax[1].plot(val_accuracy)
    ax[1].legend(labels=['Train','Validation'])
    ax[1].set_ylabel('Classification Accuracy')
    ax[1].set_title('Training Accuracy')
    plt.tight_layout()  # This will adjust the spacing
    plt.show()
    plt.close(fig)

    # Evaluate on validation dataset
    with torch.no_grad():
        x_tensor_val = torch.from_numpy(valid_X).float().to(device)
        model.eval()
        yhat_val = model(x_tensor_val)
        y_hat_val_class = np.where(yhat_val.cpu().numpy() < ythresh, 0, 1)

        # Convert to the correct shape for sklearn metrics
        valid_y_reshaped = valid_y.reshape(-1, 1)

        # Calculate metrics
        accuracy = accuracy_score(valid_y_reshaped, y_hat_val_class)
        f1 = f1_score(valid_y_reshaped, y_hat_val_class)
        precision = precision_score(valid_y_reshaped, y_hat_val_class)
        recall = recall_score(valid_y_reshaped, y_hat_val_class)
        auc = roc_auc_score(valid_y_reshaped, y_hat_val_class)
        loss = log_loss(valid_y_reshaped, y_hat_val_class)
        cm = confusion_matrix(valid_y_reshaped, y_hat_val_class)

        # Specificity, Sensitivity (Recall), Precision,
        # and Negative Predictive Value (NPV)
        try:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
            npv = tn / (tn + fn)
        except:
            print("tn+tp or tn+fn Invalid value encountered")

    # Print metrics
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Log Loss: {loss:.4f}")
    print(f"Validation Confusion Matrix: \n{cm}")
    print(f"Validation Specificity: {specificity:.4f}")
    print(f"Validation Negative Predictive Value: {npv:.4f}")

    return model, pretrain_weights, loss, accuracy



def train_baseline_model(fold: int, adj_matrices: Tuple[torch.Tensor, 
                            torch.Tensor, torch.Tensor, torch.Tensor], 
                         dimensions: Tuple[int, int, int, int, 
                            Optional[int], int], dropout_rate: float,
                         LR: float, L1REG: float, L2REG: float, 
                         betas_range: Tuple[float, float],
                         ythresh: float, BS: int, epochs: int, 
                         device: torch.device, n_seed: int,
                         pretrained_weights: Tuple[Dict[str, torch.Tensor], 
                                                   Dict[str, torch.Tensor], 
                                                   Dict[str, torch.Tensor]], 
                         transfer_flag: bool, finetune_flag: bool,
                         folder_value: str) \
                         -> Tuple[Dict[str, float], nn.Module]:
    """
    Train and evaluate the model for a given fold.

    Args:
        fold (int): The current fold number in cross-validation.
        adj_matrices (tuple): Tuple of adjacency matrices.
        dimensions (tuple): Tuple containing dimensions
                            (SNP_in, Gen_in, Pro_in, H1, H2, D_out).
        dropout_rate (float): Dropout rate for the model.
        LR (float): Learning rate for the optimizer.
        L1REG (float): L1 regularization coefficient.
        L2REG (float): L2 regularization coefficient (weight decay).
        betas_range (tuple): Betas for Adam optimizer.
        ythresh (float): Threshold for classification.
        BS (int): Batch size.
        epochs (int): Number of training epochs.
        device: Device to train the model on (e.g., 'cuda' or 'cpu').
        n_seed (int): Random seed for reproducibility.
        pretrained_weights (tuple): Tuple of pretrained weights.
        transfer_flag (bool): Whether to use pretrained weights.
        finetune_flag (bool): Whether to fine-tune pretrained weights.

    Returns:
        tuple: Tuple containing various evaluation metrics and trained model.
    """
    
    # Unpack dimensions and adjacency matrices
    SNP_in, Gen_in, Pro_in, H1, H2, D_out = dimensions
    adj_snp_gen, adj_gen_pro, adj_genpro_pro, adj_pro_pro = adj_matrices

    # Load data for the current fold
    X_train_all = pd.read_csv(
        f'./{folder_value}/xtrain_{fold}.csv', header=None)
    X_test = pd.read_csv(
        f'./{folder_value}/xtest_{fold}.csv', header=None)
    y_train_all = pd.read_csv(
        f'./{folder_value}/ytrain_{fold}.csv', header=None)
    y_test = pd.read_csv(
        f'./{folder_value}/ytest_{fold}.csv', header=None)

    # Convert data to numpy arrays
    X_train_all = np.array(X_train_all)
    X_test      = np.array(X_test)
    y_train_all = np.array(y_train_all)
    y_test      = np.array(y_test)

    # Define the size of the validation set. For example, 0.1 for 10%.
    validation_size = 0.1

    # Split the data into training and validation sets
    X_train, X_val = train_test_split(X_train_all,
                                      test_size=validation_size,
                                      random_state=n_seed)

    y_train, y_val = train_test_split(y_train_all,
                                      test_size=validation_size,
                                      random_state=n_seed)

    # Convert data to PyTorch tensors
    X_train, y_train, X_val_t, y_val_t, X_test_t, y_test_t = map(torch.tensor,
                            (X_train, y_train, X_val, y_val, X_test, y_test))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    # Create DataLoaders
    train_dl = DataLoader(dataset=train_ds, batch_size=BS)
    val_dl = DataLoader(dataset=val_ds)
    test_dl = DataLoader(dataset=test_ds)

    train_dl = WrappedDataLoader(train_dl, preprocess)
    val_dl = WrappedDataLoader(val_dl, preprocess)
    test_dl = WrappedDataLoader(test_dl, preprocess)

    # Initialize and train the model
    model_transfuse = TransFuse(*adj_matrices, 
                                *dimensions, 
                                dropout_rate).to(device)
    model_transfuse.apply(reset_weights)


    if transfer_flag is True:
        # unpack the pretrained weights
        snp_weights, gene_weights, protein_weights = pretrained_weights
        # ------------------------------------------------------------ #
        #              Transfer the pretrained weights                 #
        # ------------------------------------------------------------ #
        model_transfuse.LayerSnp2gen.load_state_dict(snp_weights)
        model_transfuse.LayerGen2pro.load_state_dict(gene_weights)
        model_transfuse.LayerPro2pro.load_state_dict(protein_weights)
        if finetune_flag is False:
            # ------------------------------------------------------------ #
            #            Step 1: Freeze the pretrained weights             #
            # ------------------------------------------------------------ #
            for param in model_transfuse.LayerSnp2gen.parameters():
                param.requires_grad = False
            for param in model_transfuse.LayerGen2pro.parameters():
                param.requires_grad = False
            for param in model_transfuse.LayerPro2pro.parameters():
                param.requires_grad = False
        else:
            # ------------------------------------------------------------ #
            #            Step 2: Finetune the pretrained weights           #
            # ------------------------------------------------------------ #
            for param in model_transfuse.LayerSnp2gen.parameters():
                param.requires_grad = True
            for param in model_transfuse.LayerGen2pro.parameters():
                param.requires_grad = True
            for param in model_transfuse.LayerPro2pro.parameters():
                param.requires_grad = True

    loss_fn = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model_transfuse.parameters(),
                                 lr=LR, weight_decay=L2REG,
                                 betas=betas_range, amsgrad=True)

    train_loss, train_accuracy, val_loss, val_accuracy = fit(epochs,
                                 model_transfuse, loss_fn, L1REG, ythresh,
                                 optimizer, train_dl, val_dl)

    # Plot training and validation loss and accuracy
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].plot(train_loss)
    ax[0].plot(val_loss)
    ax[0].set_ylabel(f'Training Loss Fold {fold}')
    ax[0].set_title(f"Train Loss, P_{H1}_{H2}, L1:{L1REG:.4f}, "
            f"L2:{L2REG:.4f}, LR:{LR:.4f}, BS:{BS}, epochs:{epochs}" )
    ax[1].plot(train_accuracy)
    ax[1].plot(val_accuracy)
    ax[1].legend(labels=['Train', 'Validation'])
    ax[1].set_ylabel('Classification Accuracy')
    ax[1].set_title('Training Accuracy')
    plt.tight_layout()  # This will adjust the spacing
    plt.show()
    plt.close(fig)

    # Evaluate the model on the validation set
    val_metrics = evaluate_model(
                      model_transfuse, X_val_t, y_val_t, ythresh, device)

    return val_metrics,  model_transfuse


def test_baseline_model(fold: int, test_model: torch.nn.Module, 
                        ythresh: float, device: torch.device,
                        folder_value: str) -> dict:
    """
    Test the model on the given fold.

    Args:
        fold (int): The current fold number.
        test_model (torch.nn.Module): The trained model to be tested.
        ythresh (float): Threshold for classification.
        device (torch.device): Device to test the model on ('cuda' or 'cpu').

    Returns:
        dict: Dictionary containing various evaluation metrics.
    """

    # Load data for the current fold
    X_test_df = pd.read_csv(
        f'./{folder_value}/xtest_{fold}.csv', header=None)
    y_test_df = pd.read_csv(
        f'./{folder_value}/ytest_{fold}.csv', header=None)

    # Convert data to numpy arrays
    X_test = np.array(X_test_df)
    y_test = np.array(y_test_df)

    # Convert data to PyTorch tensors
    X_test_tensor, y_test_tensor = map(torch.tensor, (X_test, y_test))

    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    test_dl = DataLoader(dataset=test_ds)

    test_dl = WrappedDataLoader(test_dl, preprocess)

    # Initialize and train the model
    model_transfuse = test_model

    # Evaluate the model on the testing set
    test_metrics = evaluate_model(model_transfuse, X_test_tensor, 
                                  y_test_tensor, ythresh, device)

    return test_metrics



def evaluate_model(model, X, y, threshold, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model (torch.nn.Module): The trained model.
        X (torch.Tensor): Input features tensor.
        y (torch.Tensor): Target labels tensor.
        threshold (float): Threshold for classification.
        device: Device on which the model is loaded.

    Returns:
        dict: Dictionary containing various evaluation metrics.
    """
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        predictions = model(X.float().to(device))
        predictions = predictions.cpu().numpy()
        y_true = y.cpu().numpy()

    y_hat_class = np.where(predictions < threshold, 0, 1)

    # Calculate evaluation metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_hat_class),
        'f1_score': f1_score(y_true, y_hat_class),
        'recall': recall_score(y_true, y_hat_class),
        'precision': precision_score(y_true, y_hat_class),
        'specificity': specificity(y_true, y_hat_class),
        'auc_score': roc_auc_score(y_true, y_hat_class)
    }

    return metrics


def specificity(y_true, y_pred):
    """
    Calculate the specificity.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.

    Returns:
        float: Specificity score.
    """
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()

    if tn + fp > 0:
        return tn / (tn + fp)
    else:
        return np.nan  # Return np.nan if tn + fp is zero