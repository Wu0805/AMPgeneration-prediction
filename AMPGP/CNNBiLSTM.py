import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:',device)


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_layers, kernel_size, num_filters,feature_dim,dropout1,dropout2):
        super(CNN_BiLSTM, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.dropout1 = nn.Dropout(dropout1)
        self.bilstm = nn.LSTM(num_filters, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        # CNN layer
        conv_out = F.relu(self.conv(x))
        # Permute for LSTM input
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.dropout1(conv_out)
        # BiLSTM layer
        lstm_out, _ = self.bilstm(conv_out)
        pooled = torch.max(lstm_out, dim=1)[0]
        pooled = self.dropout2(pooled)
        # Fully connected layer
        output = self.fc(pooled)
        # Apply sigmoid activation for binary classification
        output = torch.sigmoid(output)
        return output


def load_features(csv_file):
    df = pd.read_csv(csv_file)
    df = df.reset_index(drop=True)
    features = df.drop(columns=['ID'])
    return features



def load_labels(csv_file):
    df = pd.read_csv(csv_file)
    df = df.reset_index(drop=True)
    labels = df['label']
    return labels

X = load_features('your_feature_data')
y = load_labels('your_labels_data')
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
print(X.shape)
print(X_test.shape)

input_dim = 1
feature_dim=X.shape[1]
hidden_dim =100
output_dim = 1
dropout1=0.4
dropout2=0.6
lr = 0.0001
epochs = 50
batch_size = 128

lstm_layers = 2
kernel_size = 3  # Kernel size for CNN
num_filters = 64  # Number of filters for CNN

# Create the model
model = CNN_BiLSTM(input_dim, hidden_dim, output_dim, lstm_layers, kernel_size, num_filters,feature_dim,dropout1,dropout2)
model.to(device)
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
skf=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)#这个函数的作用是k折交叉验证，并且保证正负样本个数相同
train_losses_five=[]
val_losses_five=[]
val_acc_five=[]
val_pre_five=[]
val_recall_five=[]
val_sp_five=[]
val_se_five=[]
val_mcc_five=[]

for train_index, val_index in skf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    X_train = X_train.values.astype(np.float32)
    X_val = X_val.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    y_val = y_val.values.astype(np.float32)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device),
                                  torch.tensor(y_train, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device),
                                torch.tensor(y_val, dtype=torch.float32).to(device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    val_acc = 0
    val_pre = 0
    val_re = 0
    val_sp = 0
    val_se = 0
    val_mcc = 0


    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)


        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            val_loss1 = 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs.squeeze(), labels)
                val_loss1 += val_loss.item()
                predictions = [1 if output_value > 0.5 else 0 for output_value in outputs]
                val_preds.extend(predictions)
                val_targets.extend(labels.tolist())

        val_losse = val_loss1 / len(val_loader)
        val_losses.append(val_losse)


        val_accuracy = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds)  # Compute average validation loss
        val_recall = recall_score(val_targets, val_preds)
        tn, fp, fn, tp = confusion_matrix(val_targets, val_preds).ravel()
        val_specificity = tn / (tn + fp)  # Check if current validation loss is the best so far
        val_sensitivity = tp / (tp + fn)
        val_matthews = matthews_corrcoef(val_targets, val_preds)

        val_acc += val_accuracy
        val_pre += val_precision
        val_re += val_recall
        val_sp += val_specificity
        val_se += val_sensitivity
        val_mcc += val_matthews
        train_losses_five.append(train_losses)
        val_losses_five.append(val_losses)


        # Check if current validation accuracy is the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            # Optionally, you can also save the entire model
            torch.save(model, './model/BiLSTMCNN.pth')
            print("Validation accuracy improved. Model saved.")
        else:
            print("Validation accuracy did not improve.")

        # Print metrics
        print("Epoch {}/{}".format(epoch + 1,
                                       epochs) + ", Validation Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f},Se: {:4f}, MCC: {:.4f}".format(
                val_accuracy, val_precision, val_recall, val_specificity, val_sensitivity, val_matthews))
