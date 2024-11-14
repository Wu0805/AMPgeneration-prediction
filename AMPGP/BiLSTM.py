import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:',device)


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout):
        super(BiLSTM, self).__init_()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        output2 = torch.sigmoid(output)
        return output2


    def get_pre_activation_features(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out


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
scaler = StandardScaler()
X = scaler.fit_transform(X)

input_dim = X.shape[1]

hidden_dim = 182
output_dim = 1
dropout=0.65
lr = 0.004
epochs = 100
batch_size=128
model = BiLSTM(input_dim, hidden_dim, output_dim,dropout)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

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
    # 将数据上传到GPU

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device),
                                  torch.tensor(y_train, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device),
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

    # 训练模型
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

        # 计算平均训练损失
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证模型
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            val_loss1 = 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                predictions = [1 if output_value > 0.5 else 0 for output_value in outputs]
                val_preds.extend(predictions)
                val_targets.extend(labels.tolist())
                val_loss = criterion(outputs.squeeze(), labels)
                val_loss1 += val_loss.item()  # 将一个batch里的加起来

        val_losse = val_loss1 / len(val_loader)
        val_losses.append(val_losse)

        # Calculate metrics这里是常见的那些评价指标
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds)  # Compute average validation loss
        val_recall = recall_score(val_targets, val_preds)
        tn, fp, fn, tp = confusion_matrix(val_targets, val_preds).ravel()
        val_specificity = tn / (tn + fp)  # Check if current validation loss is the best so far
        val_sensitivity = tp / (tp + fn)  # 这里公式之前写的有问题
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
            torch.save(model, './model/BiLSTM.pth')
            print("Validation accuracy improved. Model saved.")
        else:
            print("Validation accuracy did not improve.")

        print("Epoch {}/{}".format(epoch + 1,
                                       epochs) + ", Validation Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f},Se: {:4f}, MCC: {:.4f}".format(
                val_accuracy, val_precision, val_recall, val_specificity, val_sensitivity, val_matthews))



