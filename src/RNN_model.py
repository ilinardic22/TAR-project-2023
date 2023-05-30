import torch
import torch.nn as nn

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset

train_filepath = './data/sst_train_raw.csv'
valid_filepath = './data/sst_valid_raw.csv'
test_filepath = './data/sst_train_raw.csv'

class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        # self.cell_type = cell_type

    def forward(self, x):
        outputs, _ = self.rnn(x)
        x = self.fc1(outputs)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    
class MyDataset(Dataset):
    def __init__(self, dataframe, trait):
        self.data = dataframe
        self.trait = trait

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data.iloc[idx]['average_word_embedding']
        labels = self.data.iloc[idx][self.trait]
        return inputs, labels
    

class BaselineRunner:
    def __init__(self, args, cell_type, hidden_size, num_layers, dropout, bidirectional, train_df, test_df, trait):
        self.args = args
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.train_df = train_df
        self.test_df = test_df
        self.trait = trait

    def train(self, model, data_loader, optimizer, criterion):
        model.train()
        predictions = []
        labels_list = []
        total_loss = 0

        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions.extend(torch.round(torch.sigmoid(outputs)).tolist())
            labels_list.extend(labels.tolist())

        return total_loss / len(data_loader), predictions, labels_list

    def evaluate(self, model, data_loader, criterion):
        model.eval()
        total_loss = 0
        total_predictions = []
        total_labels = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels.float())

                total_loss += loss.item()
                predicted_labels = torch.round(torch.sigmoid(outputs)).cpu().numpy()
                total_predictions.extend(predicted_labels)
                total_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        return avg_loss, total_predictions, total_labels

    def run(self):
        train_data = self.train_df
        test_data = self.test_df

        train_dataset = MyDataset(train_data, self.trait)
        test_dataset = MyDataset(test_data, self.trait)

        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        model = RNNModel(300, self.hidden_size, self.num_layers, self.dropout, self.bidirectional)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(self.args['epochs']):
            train_loss, train_predictions, train_labels = self.train(model, train_loader, optimizer, criterion)
            test_loss, test_predictions, test_labels = self.evaluate(model, test_loader, criterion)

            train_f1 = f1_score(train_labels, train_predictions)
            test_f1 = f1_score(test_labels, test_predictions)
            train_accuracy = accuracy_score(train_labels, train_predictions)
            test_accuracy = accuracy_score(test_labels, test_predictions)

            print(f"Epoch: {epoch+1}, Test Loss: {test_loss:.4f}, Test F1-Score: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")

            