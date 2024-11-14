import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:',device)

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)
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
        self.fc2 = nn.Linear(int((feature_dim-kernel_size+1)/2),1)
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
        x = torch.sigmoid(x)
        return x

    def get_pre_activation_features(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = out.view(-1, int((self.feature_dim-self.kernel_size+1)/2))
        out = self.fc2(out)
        out = out.view(-1,self.out_channels)
        return out


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

        # Linear transformation
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

class SelfAttentionPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout):
        super(SelfAttentionPrediction, self).__init__()
        self.attention = SelfAttention(input_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        # Self-attention
        out = self.attention(x, x, x)
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out.squeeze(-1))

        return out

    def get_pre_activation_features(self, x,mask=None):
        out = self.attention(x,x,x)
        out = self.dropout(out)
        out = self.fc1(out)
        return out.squeeze()

class CNNWithSelfAttention(nn.Module):
    def __init__(self, num_classes,extract_dim):
        super(CNNWithSelfAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.self_attention = SelfAttention(64,1)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(64 * 25*2, extract_dim)
        self.fc2 = nn.Linear(extract_dim,num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x= self.dropout(x)
        x = F.relu(self.conv2(x))


        x = x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), -1)


        x = self.self_attention(x,x,x)


        x = x.view(x.size(0), -1)
        x=self.dropout(x)


        x = self.fc1(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        return x

    def get_pre_activation_features(self,x):

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))


        x = x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), -1)


        x = self.self_attention(x, x, x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


amino_acids = 'ACDEFGHIKLMNPQRSTVWY'


aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}


def one_hot_encoding(peptide_sequence, max_length=50):
    encoding = np.zeros((max_length, len(amino_acids)))

    for i, aa in enumerate(peptide_sequence):
        if i >= max_length:
            break
        encoding[i, aa_to_index[aa]] = 1

    return encoding


def encode_fasta(input_file):
    encodings = []
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()
        sequence = ''
        for line in lines:
            if line.startswith('>'):
                if sequence:
                    encoding = one_hot_encoding(sequence)
                    encodings.append(encoding)
                    sequence = ''
            else:
                sequence += line.strip()
        encoding = one_hot_encoding(sequence)
        encodings.append(encoding)

    return torch.tensor(encodings,dtype=torch.float32)

def load_features(csv_file):
    df = pd.read_csv(csv_file)
    df = df.reset_index(drop=True)  # 重置索引并丢弃旧索引
    features = df.drop(columns=['ID'])
    return features


def load_labels(csv_file):
    df = pd.read_csv(csv_file)
    df = df.reset_index(drop=True)
    labels = df['label']
    return labels

'The characteristic data of the original data are respectively:sequence_info.csv,physicochemical_info.csv,evolutionary_info.csv'


fasta_file= ''
X=encode_fasta(fasta_file)

model_path = ''
loaded_model = torch.load(model_path)
X_feature = loaded_model.get_pre_activation_features(X.to(device))
print(X_feature.shape)

X1 = load_features('./feature/sequence_info.csv')
scaler = StandardScaler()
X1 = X1.values.astype(np.float32)
X_sample1 = torch.tensor(X1, dtype=torch.float32).unsqueeze(0).to(device)

model_path = '../model/BiLSTM.pth'
loaded_model = torch.load(model_path)
X_feature1 = loaded_model.get_pre_activation_features(X_sample1)
X_feature1=X_feature1.squeeze()
print(X_feature1.shape)

X2 = load_features('./feature/physicochemical_info.csv')


X2 = X2.values.astype(np.float32)
X_sample2 = torch.tensor(X2, dtype=torch.float32).unsqueeze(1).to(device)

model_path = '../model/CNN.pth'
loaded_model = torch.load(model_path)
X_feature2 = loaded_model.get_pre_activation_features(X_sample2)
print(X_feature2.shape)

X3 = load_features('./feature/evolutionary_info.csv')

X3 = X3.values.astype(np.float32)
X_sample3 = torch.tensor(X3, dtype=torch.float32).unsqueeze(1).to(device) #添加维度的位置也很重要

model_path = '../model/Attention.pth'
loaded_model = torch.load(model_path)
X_feature3 = loaded_model.get_pre_activation_features(X_sample3)

merge_feature=torch.cat((X_feature1,X_feature2,X_feature3,X_feature),dim=1)
merge_feature2 = torch.cat((X_feature1,X_feature2,X_feature3),dim=1)
