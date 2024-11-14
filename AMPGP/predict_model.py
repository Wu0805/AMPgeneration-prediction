import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:',device)
class Linear_lay(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Linear_lay, self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        out=self.fc1(x)
        out=self.fc2(out)
        out=self.sigmoid(out)
        return out

def load_labels(csv_file):
    df = pd.read_csv(csv_file)
    df = df.reset_index(drop=True)
    return labels

X=torch.load('./feature/merge_feature.pt')
y=load_labels('./feature/label.csv')
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
y=y.values.astype(np.float32)
print(X.shape)
print(X_test.shape)
input_dim=X.shape[1]
hidden_dim=64
batch_size=128
epochs=50
dropout=0.6

model =Linear_lay(input_dim,hidden_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
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
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]



    train_dataset = TensorDataset(X_train.to(device),
                                  torch.tensor(y_train, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val.to(device),
                                torch.tensor(y_val, dtype=torch.float32).to(device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # 先定义初值
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
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
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
        val_precision = precision_score(val_targets, val_preds)
        val_recall = recall_score(val_targets, val_preds)
        tn, fp, fn, tp = confusion_matrix(val_targets, val_preds).ravel()
        val_specificity = tn / (tn + fp)
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


        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            torch.save(model, './model/feature_fuse.pth')
            print("Validation accuracy improved. Model saved.")
        else:
            print("Validation accuracy did not improve.")

        # Print metrics
        print("Epoch {}/{}".format(epoch + 1,
                                   epochs) + ", Validation Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f},Se: {:4f}, MCC: {:.4f}".format(
            val_accuracy, val_precision, val_recall, val_specificity, val_sensitivity, val_matthews))

