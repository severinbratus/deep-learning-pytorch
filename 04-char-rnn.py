import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


hidden_size = 128
num_layers = 1
learning_rate = 0.001
epochs = 20
sequence_length = 25

# source: https://github.com/jcjohnson/torch-rnn/tree/master/data
text = open("data/tinyshakespeare.txt", "r").read()[:6000]

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

data = [char_to_idx[char] for char in text]

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.reshape(-1, self.hidden_size))
        return out, hidden

model = CharRNN(vocab_size, hidden_size, num_layers)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=learning_rate)

def get_batches(data, seq_length):
    for i in range(0, len(data) - seq_length):
        inputs = data[i:i + seq_length]
        targets = data[i + 1:i + seq_length + 1]
        yield inputs, targets

for epoch in range(epochs):
    hidden = torch.zeros(num_layers, 1, hidden_size)
    epoch_loss = 0

    for inputs, targets in get_batches(data, sequence_length):
        inputs_one_hot = torch.zeros(sequence_length, vocab_size)
        for t, char_idx in enumerate(inputs):
            inputs_one_hot[t][char_idx] = 1.0

        inputs_one_hot = inputs_one_hot.unsqueeze(0)
        targets = torch.tensor(targets)

        outputs, hidden = model(inputs_one_hot, hidden)
        hidden = hidden.detach()

        loss = loss_fn(outputs, targets)
        epoch_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch + 1}/{epochs}; Loss: {epoch_loss:.4f}")


torch.save(model.state_dict(), "models/char-rnn.pt")


# Text generation
def generate_text(model, start_char, length):
    model.eval()
    input_char = torch.zeros(1, 1, vocab_size)
    input_char[0][0][char_to_idx[start_char]] = 1.0
    hidden = torch.zeros(num_layers, 1, hidden_size)
    generated_text = start_char

    for _ in range(length):
        output, hidden = model(input_char, hidden)
        prob = torch.softmax(output, dim=1).data
        char_idx = torch.multinomial(prob, 1).item()
        generated_text += idx_to_char[char_idx]

        input_char = torch.zeros(1, 1, vocab_size)
        input_char[0][0][char_idx] = 1.0

    return generated_text


gen_length = 500
print(f"Generating {gen_length} chars...")
print(generate_text(model, start_char="\n", length=gen_length))
