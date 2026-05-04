import torch
import torch.nn as nn
 
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerRegressor, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
 
    def forward(self, x, src_key_padding_mask=None):
        x = self.input_embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = self.fc(x)
        return output