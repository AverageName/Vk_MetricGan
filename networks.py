import torch.nn as nn
import torch.nn.functional as F


class LstmGen(nn.Module):


    def __init__(self, input_features, hidden_dim, num_layers, fc_hidden_dim):
        super(LstmGen, self).__init__()

        self.lstm = nn.LSTM(input_features, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, input_features)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.3))
        x = F.sigmoid(self.fc2(x))
        return x


class ConvDiscriminator(nn.Module):

    def __init__(self):
        super(ConvDiscriminator, self).__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=2, out_channels=15,
                                                      kernel_size=(5, 5)))
        
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=15, out_channels=25,
                                                      kernel_size=(7, 7)))
        
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channels=25, out_channels=40,
                                                      kernel_size=(9, 9)))
        
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(in_channels=40, out_channels=50,
                                                      kernel_size=(11, 11)))
        
        self.fc1 = nn.utils.spectral_norm(nn.Linear(50, 50))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(50, 10))
        self.fc3 = nn.utils.spectral_norm(nn.Linear(10, 1))
        self.bn = nn.BatchNorm2d(2)
    
    def forward(self, x):
        x = self.bn(x)
        x = F.leaky_relu(self.conv1(x), 0.3)
        x = F.leaky_relu(self.conv2(x), 0.3)
        x = F.leaky_relu(self.conv3(x), 0.3)
        x = F.leaky_relu(self.conv4(x), 0.3)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)

        x = F.leaky_relu(self.fc1(x), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = self.fc3(x)
        return x
