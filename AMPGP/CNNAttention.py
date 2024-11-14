import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:',device)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size=hidden_size
        assert (
            self.head_dim * num_heads == hidden_size
        ), "Hidden size must be divisible by the number of heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]


        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Reshape Q, K, V
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        # Apply attention to value
        x = torch.matmul(attention, V)

        # Reshape and concatenate
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)

        # Linear layer
        x = self.fc_out(x)

        return x

class CNNAttention(nn.Module):
    def __init__(self, input_channels, hidden_size, num_heads,dropout,extract_dim):
        super(CNNAttention, self).__init__()
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(input_channels, hidden_size, kernel_size=3, padding=1)
        self.self_attention = SelfAttention(hidden_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, extract_dim)
        self.fc2 = nn.Linear(extract_dim,1)

    def forward(self, x):
        # Convolutional layer
        out = self.conv1(x.transpose(1, 2))  # Conv1d expects input in shape (batch_size, channels, sequence_length)
        out = out.transpose(1, 2)  # Transpose back to (batch_size, sequence_length, channels)

        # Self-attention
        out = self.self_attention(out, out, out)
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc1(out)
        out = self.fc2(out)

        return out.squeeze(-1)

    def get_pre_activation_features(self, x):
        out = self.conv1(x.transpose(1, 2))
        out = out.transpose(1, 2)
        out = self.self_attention(out, out, out)
        out = self.dropout(out)
        out = self.fc1(out)
        return out.squeeze()

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

input_size = X.shape[1]
hidden_size =32
num_layers=1
num_heads=1
output_dim = 1
extract_dim=40
dropout=0.6
lr = 0.0001
epochs = 20
batch_size = 128
model = CNNAttention(input_size, hidden_size,num_heads,dropout,extract_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
skf=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
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
                outputcome = torch.sigmoid(outputs)
                predictions = [1 if output_value > 0.5 else 0 for output_value in outputcome]
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
            torch.save(model, './model/CNNAttention.pth')
            print("Validation accuracy improved. Model saved.")
        else:
            print("Validation accuracy did not improve.")

        # Print metrics
        print("Epoch {}/{}".format(epoch + 1,
                                       epochs) + ", Validation Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f},Se: {:4f}, MCC: {:.4f}".format(
                val_accuracy, val_precision, val_recall, val_specificity, val_sensitivity, val_matthews))