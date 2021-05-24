import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class comp_claim_dataset(Dataset):
    def __init__(self, csvpath, mode = 'train'):
        self.mode = mode
        self.claim_data_raw = pd.read_csv(csvpath)

        oe_style = OrdinalEncoder()
        oe_results = oe_style.fit_transform(self.claim_data_raw[['Claim Type']])
        self.claim_data_clean = pd.DataFrame(oe_results, columns = ['Claim Type'])

        self.add_categorical_columns([
            'District Name', 
            'Current Claim Status', 
            'Claim Injury Type', 
            'WCIO Part Of Body Code',
            'WCIO Nature of Injury Code',
            'WCIO Cause of Injury Code'])

        self.add_categorical_columns_string([
            'OIICS Nature of Injury Code', 
            'OIICS Injury Source Code',
            'OIICS Event Exposure Code',
            'OIICS Secondary Source Code',
            'Alternative Dispute Resolution',
            'Gender',
            'Medical Fee Region'])

        oe_style = OrdinalEncoder()
        oe_results = oe_style.fit_transform(self.claim_data_raw[['Attorney/Representative']])
        self.output = pd.DataFrame(oe_results)

        self.inp = self.claim_data_clean.iloc[:,0:].values
        self.oup = self.output.iloc[:,0:].values
    
    def __len__(self):
        return len(self.claim_data_raw)

    def __getitem__(self, idx):
        if(self.mode == 'train'):
            return {
                'inp' : torch.Tensor(self.inp[idx]),
                'oup' : torch.Tensor(self.oup[idx])}
        else:
            return {
                'inp' : torch.Tensor(self.inp[idx])}

    def add_categorical_columns(self, columns):
        for column  in columns:
            oe_style = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            oe_results = oe_style.fit_transform(self.claim_data_raw[[column]].fillna(0))
            self.claim_data_clean[column] = oe_results

    def add_categorical_columns_string(self, columns):
        for column  in columns:
            oe_style = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            oe_results = oe_style.fit_transform(self.claim_data_raw[[column]].fillna(''))
            self.claim_data_clean[column] = oe_results

