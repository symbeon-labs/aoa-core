import torch
import torch.nn as nn
import torch.optim as optim
from models.ofp_model import OFPExtractor

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for anchor, positive, negative in dataloader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)
        
        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Entrypoint simulado
if __name__ == '__main__':
    print('Treinadora instanciada. Dataset real necessário para inicialização.')
