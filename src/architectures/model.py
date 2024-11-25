import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, feature_dim=960, num_heads=8, num_layers=1, ff_hidden_dim=128, dropout=0.1):
        super(Net, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(feature_dim, 50)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x[..., :-1]
        x = x.squeeze(1)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x[-1]  # Take the output from the last token or use pooling
        x = self.softmax(self.fc(x))
        return x


if __name__ == "__main__":
    input_tensor = torch.randn(8, 1, 256, 1921)
    model = Net()
    output = model(input_tensor)
    print(output.shape)
