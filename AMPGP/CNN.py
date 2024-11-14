import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:',device)


class CNN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,feature_dim,dropout):
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.feature_dim=feature_dim
        self.dropout=dropout
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        #self.fc1 = nn.Linear(16 * 45, 1)
        self.fc2 = nn.Linear(int((feature_dim - kernel_size + 1) / 2), 1)
        self.fc3 = nn.Linear(out_channels,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1,int((self.feature_dim-self.kernel_size+1)/2))
        x = self.fc2(x)
        x= x.view(-1,self.out_channels)
        x= self.fc3(x)
        return x


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

def prob_to_class(output):
    for i in range(0, len(output)):
        if output[i] >= 0.5:
           output[i] = 1
        else:
           output[i] = 0
    return(output)

X = load_features('your_feature_data')
y = load_labels('your_labels_data')
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
print(X.shape)
print(X_test.shape)
y=y.values.astype(np.float32)

f_dim=X.shape[1]
print(f_dim)
epochs = 50
batch_size = 128
dropout=0.4
out_channel=135

model = CNN(in_channels=1,out_channels=out_channel,kernel_size=3,feature_dim=f_dim,dropout=dropout).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    y_train, y_val = y[train_index], y[val_index]
    X_train = X_train.values.astype(np.float32)
    X_val = X_val.values.astype(np.float32)



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
            torch.save(model, './model/CNN.pth')
            print("Validation accuracy improved. Model saved.")
        else:
            print("Validation accuracy did not improve.")

        # Print metrics
        print("Epoch {}/{}".format(epoch + 1,
                                       epochs) + ", Validation Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f},Se: {:4f}, MCC: {:.4f}".format(
                val_accuracy, val_precision, val_recall, val_specificity, val_sensitivity, val_matthews))
