import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import time

from model import BERT


def train(model, train_dataloader, val_dataloader, epochs=3, lr=5e-5, device='cuda'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        start_time = time.time()
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
            # Move data to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask)
            logits = outputs[:, 0, :]  # Consideriamo solo la rappresentazione di [CLS] token

            # Calcolare la loss
            loss = loss_fn(logits, labels)
            total_train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Calcolare il loss medio per epoca
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Valutazione sul set di validazione
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(val_dataloader):
                # Move data to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs[:, 0, :]  # Consideriamo solo la rappresentazione di [CLS] token

                # Calcolare la loss
                loss = loss_fn(logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        # Print training statistics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Salvataggio del modello migliore
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
        
        end_time = time.time()
        print(f"Epoch time: {end_time - start_time:.2f} seconds")
        
# Funzione per caricare il dataset e creare dataloaders
def load_data(input_ids, attention_mask, labels, batch_size=32):
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    # Splitting the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


# Example usage
if __name__ == "__main__":
    input_ids = torch.randint(0, 10000, (1000, 128))  # esempio di dati (batch_size, seq_len)
    attention_mask = torch.ones_like(input_ids)       # esempio di maschera di attenzione
    labels = torch.randint(0, 2, (1000,))             # esempio di etichette per la classificazione binaria
    
    train_dataloader, val_dataloader = load_data(input_ids, attention_mask, labels)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Inizializzare il modello BERT
    vocab_size = 10000  # Esempio di dimensione del vocabolario
    d_model = 512       # Dimensione del modello
    n_layers = 6        # Numero di layer del transformer
    h = 8               # Numero di teste di attenzione
    d_ff = 2048         # Dimensione della feed-forward network
    seq_len = 128       # Lunghezza massima della sequenza
    dropout = 0.1       # Dropout
    
    model = BERT(vocab_size, d_model, n_layers, h, d_ff, seq_len, dropout)
    
    # Training
    train(model, train_dataloader, val_dataloader, epochs=3, lr=5e-5, device=device)

