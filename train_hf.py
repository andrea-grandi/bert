import torch
import requests
import random
from transformers import Trainer, TrainingArguments, BertTokenizer
from model import BERT

vocab_size = 10000  # Esempio di dimensione del vocabolario
d_model = 512       # Dimensione del modello
n_layers = 6        # Numero di layer del transformer
h = 8               # Numero di teste di attenzione
d_ff = 2048         # Dimensione della feed-forward network
seq_len = 128       # Lunghezza massima della sequenza
dropout = 0.1       # Dropout

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BERT(vocab_size, d_model, n_layers, h, d_ff, seq_len, dropout)

data = requests.get("https://raw.githubusercontent.com/jamescalam/transformers/main/data/text/meditations/clean.txt")
text = data.text.split('\n')
bag = [sentence for para in text for sentence in para.split('.') if sentence != '']
bag_size = len(bag)
#print(f"Bag size: {bag_size}")

sentence_a = []
sentence_b = []
labels = []

for paragraph in text:
    sentences = [sentence for sentence in paragraph.split('.') if sentence != '']
    num_sentences = len(sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences - 2)
        sentence_a.append(sentences[start])
        if random.random() > 0.5:
            sentence_b.append(sentences[start + 1])
            labels.append(0)
        else:
            sentence_b.append(bag[random.randint(0, bag_size - 1)])
            labels.append(1)

inputs = tokenizer(sentence_a, sentence_b, padding='max_length', truncation=True, return_tensors="pt", max_length=512)

#print(f"Input IDs: {inputs.keys()}")

inputs['next_sentence_label'] = torch.LongTensor([labels]).T
#print(inputs['next_sentence_label'][:10])
