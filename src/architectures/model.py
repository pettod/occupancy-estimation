import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, feature_dim=1920, num_heads=8, num_layers=2, ff_hidden_dim=2048, dropout=0.1):
        super(Net, self).__init__()
        self.feature_dim = feature_dim
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_hidden_dim,
            dropout=dropout
        )
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layer to reduce the output to a single value
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # Reshape input from (batch, 1, 256, 1921) to (batch, 256, 1921)
        x = x[..., :-1]  # 1921 to 1920
        x = x.squeeze(1)  # Remove the dimension of size 1 (from shape (batch, 1, 256, 1921) to (batch, 256, 1921))
        
        # Permute to (seq_len, batch, feature) for Transformer
        x = x.permute(1, 0, 2)  # Shape: (256, batch, 1921)
        
        # Pass through the Transformer Encoder
        x = self.transformer_encoder(x)  # Output shape: (256, batch, 1921)
        
        # Take the output from the last token or use pooling
        x = x[-1]  # Shape: (batch, 1921)
        
        # Pass through the fully connected layer to produce output of shape (batch, 1)
        x = F.relu(self.fc(x))  # Shape: (batch, 1)
        #print(x)
        return 100 * x


if __name__ == "__main__":
    # Example usage
    input_tensor = torch.randn(8, 1, 256, 1921)  # Example with batch size of 8
    model = Net()
    output = model(input_tensor)
    print(output.shape)  # Output shape should be (batch, 1), e.g., (8, 1)
