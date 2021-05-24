from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from comp_claim_dataset import comp_claim_dataset
from comp_claim_model import CompClaimModel, get_sample_model_config
import torch.nn as nn
import torch
from tqdm import tqdm

def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss =criterion(output,y)
    loss.backward()
    optimizer.step()

    return loss, output

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Cuda Device Available")
  print("Name of the Cuda Device: ", torch.cuda.get_device_name())
  print("GPU Computational Capablity: ", torch.cuda.get_device_capability())

EPOCHS = 200
BATCH_SIZE = 20000
data = comp_claim_dataset('C:\\github\\xcs\\data\\assembled-workers-compensation-claims-beginning-2000.csv')
data_train = DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle =False)

layer_config = get_sample_model_config()

model = CompClaimModel(layer_config, 14)
model.to(device)

criterion = nn.MSELoss()
optm = Adam(model.parameter_list, lr = 0.001)

for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0
    for bidx, batch in tqdm(enumerate(data_train)):
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        loss, predictions = train(model,x_train,y_train, optm, criterion)
        for idx, i in enumerate(predictions):
            i  = torch.round(i)
            if i == y_train[idx]:
                correct += 1
        acc = (correct/len(data))
        epoch_loss+=loss
    print('Epoch {} Accuracy : {}'.format(epoch+1, acc*100))
    print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))