# Note: Code adapted from https://github.com/bartbussmann/NAVAR , MIT license
import torch.nn as nn
import torch
import numpy as np
import os



class NAVAR(nn.Module):
    def __init__(self, num_nodes, num_hidden, maxlags, hidden_layers=1, dropout=0):
        """
        Neural Additive Vector AutoRegression (NAVAR) model
        Args:
            num_nodes: int
                The number of time series (N)
            num_hidden: int
                Number of hidden units per layer
            maxlags: int
                Maximum number of time lags considered (K)
            hidden_layers: int
                Number of hidden layers
            dropout:
                Dropout probability of units in hidden layers
        """
        super(NAVAR, self).__init__()
        self.num_nodes = num_nodes
        self.num_hidden = num_hidden
        self.first_hidden_layer = nn.Conv1d(num_nodes, num_hidden * num_nodes, kernel_size=maxlags, groups=num_nodes)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_layer_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for k in range(hidden_layers - 1):
            self.hidden_layer_list.append(
                nn.Conv1d(num_nodes, num_hidden * num_nodes, kernel_size=num_hidden, groups=num_nodes))
            self.dropout_list.append(nn.Dropout(p=dropout))
        self.contributions = nn.Conv1d(num_nodes, num_nodes * num_nodes, kernel_size=num_hidden, groups=num_nodes)
        self.biases = nn.Parameter(torch.ones(1, num_nodes) * 0.0001)
        self.causal_matrix = None
        pass

    def forward(self, x):
        hidden = self.first_hidden_layer(x).clamp(min=0).transpose(-1, -2).reshape([-1, self.num_nodes, self.num_hidden])
        hidden = self.dropout(hidden)
        for i in range(len(self.hidden_layer_list)):
            hidden = self.hidden_layer_list[i](hidden).clamp(min=0).view([-1, self.num_nodes, self.num_hidden])
            hidden = self.dropout_list[i](hidden)
        contributions = self.contributions(hidden)
        contributions = contributions.view([-1, self.num_nodes, self.num_nodes, 1])
        predictions = torch.sum(contributions, dim=1).squeeze() + self.biases
        contributions = contributions.view([-1, self.num_nodes * self.num_nodes, 1])
        return predictions, contributions

    def GC(self,):
        return self.causal_matrix 

    
    def fit(self, save_path, X_train, Y_train, criterion, optimizer, X_val=None, Y_val=None, val_proportion=0.0, epochs=200, 
            batch_size=300, split_timeseries=False, check_every=1000, lambda1=0):
        X_train = np.swapaxes(X_train,2,1) # need to change X from being formatted as (batch, time, nodes) to (batch, nodes, time)
        X_train = torch.from_numpy(X_train)
        if X_val is not None:
            X_val = np.swapaxes(X_val,2,1)
            X_val = torch.from_numpy(X_val)

        if torch.cuda.is_available():
            X_train = X_train.cuda()
            if X_val is not None:
                X_val = X_val.cuda()

        num_training_samples = X_train.size()[0]
        total_loss = 0
        loss_val = 0

        # start of training loop
        batch_counter = 0
        for t in range(1, epochs +1):
            #obtain batches
            batch_indeces_list = []
            if batch_size < num_training_samples:
                batch_perm = np.random.choice(num_training_samples, size=num_training_samples, replace=False)
                for i in range(int(num_training_samples/batch_size) + 1):
                    start = i*batch_size
                    batch_i = batch_perm[start:start+batch_size]
                    if len(batch_i) > 0:
                        batch_indeces_list.append(batch_perm[start:start+batch_size])
            else:
                batch_indeces_list = [np.arange(num_training_samples)]

            for batch_indeces in batch_indeces_list:
                batch_counter += 1
                X_batch = X_train[batch_indeces][:,:,:-1]
                Y_batch = X_train[batch_indeces][:,:,-1]
                # forward pass to calculate predictions and contributions
                predictions, contributions = self.forward(X_batch)
                # calculate the loss
                if not split_timeseries:
                    loss_pred = criterion(predictions, Y_batch)
                else:
                    loss_pred = criterion(predictions[:,:,-1], Y_batch[:,:,-1])
                loss_l1 = (lambda1/self.num_nodes) * torch.mean(torch.sum(torch.abs(contributions), dim=1))
                loss = loss_pred + loss_l1
                total_loss += loss
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # every 'check_every' epochs we calculate and print the validation loss
            if t % check_every == 0:
                self.eval()
                if val_proportion > 0.0:
                    val_pred, val_contributions = self.forward(X_val[:,:,:-1])
                    loss_val = criterion(val_pred, X_val[:,:,-1])#Y_val)
                self.train()
                print(f'iteration {t}. Loss: {total_loss/batch_counter}  Val loss: {loss_val}')
                total_loss = 0
                batch_counter = 0

        # use the trained model to calculate the causal scores
        self.eval()
        y_pred, contributions = self.forward(X_train[:,:,:-1])
        self.causal_matrix = torch.std(contributions, dim=0).view(self.num_nodes, self.num_nodes).detach().cpu().numpy()
        model_save_path = os.path.join(save_path, "final_best_model.bin")
        torch.save(self, model_save_path)
        return loss_val


    
class NAVARLSTM(nn.Module):
    def __init__(self, num_nodes, num_hidden, maxlags, hidden_layers=1, dropout=0):
        """
        Neural Additive Vector AutoRegression (NAVAR) model
        Args:
            num_nodes: int
                The number of time series (N)
            num_hidden: int
                Number of hidden units per layer
            maxlags: int
                Maximum number of time lags considered (K)
            hidden_layers: int
                Number of hidden layers
            dropout:
                Dropout probability of units in hidden layers
        """
        super(NAVARLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.num_hidden = num_hidden
        self.lstm_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        for node in range(self.num_nodes):
            self.lstm_list.append(nn.LSTM(1, num_hidden, hidden_layers, dropout=dropout, batch_first=True))
            self.fc_list.append(nn.Linear(num_hidden, num_nodes))
        self.biases = nn.Parameter(torch.ones(1, num_nodes) * 0.0001)
        self.causal_matrix = None
        pass

    def forward(self, x):
        batch_size, number_of_nodes, time_series_length = x.size()
        contributions = torch.zeros((batch_size, self.num_nodes * self.num_nodes, time_series_length))
        if torch.cuda.is_available():
            contributions = contributions.cuda()
        # we split the input into the components
        x = x.split(1, dim=1)
        # then we apply the LSTM layers and calculate the contributions
        for node in range(self.num_nodes):
            model_input = torch.transpose(x[node], 1, 2)
            lstm = self.lstm_list[node]
            fc = self.fc_list[node]
            lstm_output, _ = lstm(model_input)
            contributions[:, node * self.num_nodes:(node + 1) * self.num_nodes, :] = fc(lstm_output).transpose(1, 2)
        contributions = contributions.view([batch_size, self.num_nodes, self.num_nodes, time_series_length])
        predictions = torch.sum(contributions, dim=1) + self.biases.transpose(0, 1)
        contributions = contributions.permute(0, 3, 1, 2)
        contributions = contributions.reshape(-1, self.num_nodes * self.num_nodes, 1)
        return predictions, contributions
        
    def GC(self,):
        return self.causal_matrix
        
        
    def fit(self, save_path, X_train, Y_train, criterion, optimizer, X_val=None, Y_val=None, val_proportion=0.0, epochs=200, 
            batch_size=300, check_every=1000, lambda1=0):
        X_train = np.swapaxes(X_train,2,1) # need to change X from being formatted as (batch, time, nodes) to (batch, nodes, time)
        X_train = torch.from_numpy(X_train)
        if X_val is not None:
            X_val = np.swapaxes(X_val,2,1)
            X_val = torch.from_numpy(X_val)

        if torch.cuda.is_available():
            X_train = X_train.cuda()
            if X_val is not None:
                X_val = X_val.cuda()

        num_training_samples = X_train.size()[0]
        total_loss = 0
        loss_val = 0

        # start of training loop
        batch_counter = 0
        for t in range(1, epochs +1):
            #obtain batches
            batch_indeces_list = []
            if batch_size < num_training_samples:
                batch_perm = np.random.choice(num_training_samples, size=num_training_samples, replace=False)
                for i in range(int(num_training_samples/batch_size) + 1):
                    start = i*batch_size
                    batch_i = batch_perm[start:start+batch_size]
                    if len(batch_i) > 0:
                        batch_indeces_list.append(batch_perm[start:start+batch_size])
            else:
                batch_indeces_list = [np.arange(num_training_samples)]

            for batch_indeces in batch_indeces_list:
                batch_counter += 1
                X_batch = X_train[batch_indeces][:,:,:-1]
                Y_batch = X_train[batch_indeces][:,:,-1]
                # forward pass to calculate predictions and contributions
                predictions, contributions = self.forward(X_batch)
                # calculate the loss
                loss_pred = criterion(predictions[:,:,-1], Y_batch)
                loss_l1 = (lambda1/self.num_nodes) * torch.mean(torch.sum(torch.abs(contributions), dim=1))
                loss = loss_pred + loss_l1
                total_loss += loss
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # every 'check_every' epochs we calculate and print the validation loss
            if t % check_every == 0:
                self.eval()
                if val_proportion > 0.0:
                    val_pred, val_contributions = self.forward(X_val[:,:,:-1])
                    loss_val = criterion(val_pred, X_val[:,:,-1])
                self.train()
                print(f'iteration {t}. Loss: {total_loss/batch_counter}  Val loss: {loss_val}')
                total_loss = 0
                batch_counter = 0

        # use the trained model to calculate the causal scores
        self.eval()
        y_pred, contributions = self.forward(X_train[:,:,:-1])
        self.causal_matrix = torch.std(contributions, dim=0).view(self.num_nodes, self.num_nodes).detach().cpu().numpy()
        model_save_path = os.path.join(save_path, "final_best_model.bin")
        torch.save(self, model_save_path)
        return loss_val