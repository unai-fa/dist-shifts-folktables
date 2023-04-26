import pandas as pd
import torch
import torch.nn as nn

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.losses import LabelRelaxationLoss, LabelSmoothingLossCanonical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import TensorDataset, DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss, alpha: float = 0.1, hidden_layer_sizes: tuple = (128,), activation: str = "relu",
                 l2_penalty: float = 1e-4, learning_rate: float = 1e-3, momentum: float = 0.0, epochs: int = 300,
                 batch_size: int = 256, provide_alphas: bool = False, n_classes: int = None,
                 validation_split: float = 0.0, instance_weight: float = None, patience: int = 10, tol: float = 1e-3,
                 seed: int = 42):
        super().__init__()

        self.alpha = alpha

        self.loss = loss

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        if self.n_classes is not None:
            self.classes_ = np.arange(int(self.n_classes))

        self._internal_model = None

        self.provide_alphas = provide_alphas

        self._history = None
        self.validation_split = validation_split

        self.model = None

        self.instance_weight = instance_weight

        self.patience = patience
        self.tol = tol

        self.seed = seed

    def _one_hot_encoding(self, targets):
        if self.n_classes is None:
            enc = OneHotEncoder(sparse=False)
            return enc.fit_transform(targets.reshape(-1, 1))
        else:
            one_hot_encoded = np.zeros((len(targets), self.n_classes))
            one_hot_encoded[np.arange(len(targets)), targets] = 1
            return one_hot_encoded

    def fit(self, X, y, val_X=None, val_y=None):
        """
        Fits the label relaxation model. The targets y are one-hot encoded in case a simple list is provided.
        """
        input_dim = X.shape[1]

        if self.provide_alphas:
            alphas = np.squeeze(y[:, -1] * self.alpha)
            targets = np.squeeze(y[:, :-1])
        elif self.instance_weight is not None:
            weights = np.squeeze(y[:, -1])
            alphas = np.where(weights == 0, np.ones_like(weights), np.ones_like(weights) * self.instance_weight)
            targets = np.squeeze(y[:, :-1])
        else:
            alphas = None
            targets = y

        # targets = y if not self.provide_alphas else y[0]
        if len(targets.shape) < 2:
            targets = self._one_hot_encoding(targets.astype(int))

        if self.n_classes is None:
            self.n_classes = targets.shape[1]
        if self.classes_ is None:
            self.classes_ = np.arange(int(self.n_classes))

        if alphas is not None:
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(targets), torch.Tensor(alphas))
        else:
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(targets))

        if self.validation_split > 0:
            train_size = int((1 - self.validation_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                                       torch.Generator().manual_seed(self.seed))
            trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        elif val_X is not None and val_y is not None:
            trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            val_y = self._one_hot_encoding(val_y.values.astype(int))
            valloader = DataLoader(TensorDataset(torch.Tensor(val_X.values), torch.Tensor(val_y)),
                                   batch_size=self.batch_size,
                                   shuffle=False)
        else:
            trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            valloader = None

        # Create model
        model_layers = []
        # if len(self.hidden_layer_sizes) > 0:
        in_features = input_dim
        for hl_size in self.hidden_layer_sizes:
            model_layers.append(nn.Linear(in_features, hl_size))
            model_layers.append(nn.ReLU())
            # model_layers.append(nn.Dropout(p=0.5))
            in_features = hl_size

        model_layers.append(nn.Linear(in_features=in_features, out_features=self.n_classes))
        model_layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*model_layers)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty,
        #                             momentum=0.0, nesterov=False)

        # LR schedule
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        self._history = {'val_loss': [], 'val_acc': [], 'train_loss': [], 'train_acc': []}

        best_loss = np.Inf
        wait = 0

        for epoch in range(self.epochs):
            losses = AverageMeter()
            train_acc = AverageMeter()

            self.model.train()

            for batch_idx, elems in enumerate(trainloader):
                if len(elems) == 3:
                    inputs, targets, alphas = elems
                else:
                    inputs, targets = elems
                    alphas = None

                outputs = self.model(inputs)

                loss = self.loss(outputs, targets, alphas)

                losses.update(loss.item(), inputs.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Measure accuracy
                preds = torch.argmax(outputs, dim=-1)
                arg_max_targets = torch.argmax(targets, dim=-1)
                acc = torch.mean((preds == arg_max_targets).float()).item()
                train_acc.update(acc, inputs.size(0))

            scheduler.step()

            self._history['train_loss'].append(losses.avg)
            self._history['train_acc'].append(train_acc.avg)

            if losses.avg < (best_loss + self.tol):
                best_loss = losses.avg
                wait = 0
            else:
                wait += 1

            if wait >= self.patience:
                print("Stopping early at epoch %d with loss: %.4f" % (epoch + 1, losses.avg))
                break

            # Validation scoring
            if valloader is not None:
                self.model.eval()

                val_losses = AverageMeter()
                val_acc = AverageMeter()

                for batch_idx, elems in enumerate(valloader):
                    if len(elems) == 3:
                        inputs, targets, alphas = elems
                    else:
                        inputs, targets = elems
                        alphas = None

                    outputs = self.model(inputs)

                    loss = self.loss(outputs, targets, alphas)
                    val_losses.update(loss.item(), inputs.size(0))

                    # Measure accuracy
                    preds = torch.argmax(outputs, dim=-1)
                    arg_max_targets = torch.argmax(targets, dim=-1)
                    acc = torch.mean((preds == arg_max_targets).float()).item()
                    val_acc.update(acc, inputs.size(0))

                self._history['val_loss'].append(val_losses.avg)
                self._history['val_acc'].append(val_acc.avg)

                # print('[epoch: %d] Loss: %.4f, Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f' % (
                #     epoch + 1, losses.avg, train_acc.avg, val_losses.avg, val_acc.avg))
            # else:
            #     print('[epoch: %d] Loss: %.4f, Acc: %.4f' % (epoch + 1, losses.avg, train_acc.avg))

    def predict(self, X):
        assert self.model is not None, "Model needs to be fit before prediction."

        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X = X.values
            X_tensor = torch.Tensor(X)

            return torch.argmax(self.model(X_tensor), dim=-1).detach().numpy()

    def predict_proba(self, X):
        # As last layer is a softmax layer
        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X = X.values
            X_tensor = torch.Tensor(X)

            return self.model(X_tensor).detach().numpy()

    def history(self):
        return self._history

    def __str__(self):
        # Return class name and all attributes
        return self.__class__.__name__ + ": " + str(self.__dict__)


class TorchLabelRelaxationNNClassifier(TorchNNClassifier):
    def __init__(self, alpha: float = 0.1, hidden_layer_sizes: tuple = (128,), activation: str = "relu",
                 l2_penalty: float = 1e-4, learning_rate: float = 1e-3, momentum: float = 0.0, epochs: int = 300,
                 batch_size: int = 256, provide_alphas: bool = False, n_classes: int = None,
                 validation_split: float = 0.0, instance_weight: float = None) -> object:
        if provide_alphas:
            loss = LabelRelaxationLoss(0.0)
        else:
            loss = LabelRelaxationLoss(alpha)

        super().__init__(loss, alpha, hidden_layer_sizes, activation, l2_penalty, learning_rate, momentum, epochs,
                         batch_size, provide_alphas, n_classes, validation_split, instance_weight)


class TorchLabelSmoothingNNClassifier(TorchNNClassifier):
    def __init__(self, alpha: float = 0.1, hidden_layer_sizes: tuple = (128,), activation: str = "relu",
                 l2_penalty: float = 1e-4, learning_rate: float = 1e-3, momentum: float = 0.0, epochs: int = 300,
                 batch_size: int = 256, provide_alphas: bool = False, n_classes: int = None,
                 validation_split: float = 0.0, instance_weight: float = None):
        loss = LabelSmoothingLossCanonical(alpha)

        super().__init__(loss, alpha, hidden_layer_sizes, activation, l2_penalty, learning_rate, momentum, epochs,
                         batch_size, provide_alphas, n_classes, validation_split, instance_weight)


class TorchCrossEntropyNNClassifier(TorchNNClassifier):
    def __init__(self, alpha: float = 0.1, hidden_layer_sizes: tuple = (128,), activation: str = "relu",
                 l2_penalty: float = 1e-4, learning_rate: float = 1e-3, momentum: float = 0.0, epochs: int = 300,
                 batch_size: int = 256, provide_alphas: bool = False, n_classes: int = None,
                 validation_split: float = 0.0, instance_weight: float = None):
        self.internal_loss = nn.CrossEntropyLoss(reduction="none")
        loss = self.loss

        super().__init__(loss, alpha, hidden_layer_sizes, activation, l2_penalty, learning_rate, momentum, epochs,
                         batch_size, provide_alphas, n_classes, validation_split, instance_weight)

    def loss(self, pred, target, alpha):
        losses = self.internal_loss(pred, target)
        if alpha is not None:
            return (alpha * losses).mean()
        else:
            return losses.mean()
