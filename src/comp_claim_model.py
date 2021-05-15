import torch
import torch.nn as nn

def get_sample_model_config():
    layer1_config = {
        'dim':100,
        'batch_normal': True,
        'drop_out_rate': 0.8,
    }

    layer2_config = {
        'dim':10,
        'batch_normal': True,
        'batch_normal_features':10,
        'drop_out_rate': 0.8,
    }

    return [layer1_config, layer2_config]

class CompClaimModel(nn.Module):

    def __init__(self, layers_config, input_data_dim):
        super().__init__()
        
        self.layers_config = layers_config

        self.layers = []

        current_dim = input_data_dim
        for layer in self.layers_config:
            self.layers.append(nn.Linear(current_dim, layer['dim']))
            current_dim = layer['dim']
            if layer['batch_normal']:
                self.layers.append(nn.BatchNorm1d(layer['batch_normal_features']))
            
            self.layers.append(nn.Dropout(layer['drop_out_rate']))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x