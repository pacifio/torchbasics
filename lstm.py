"""
# Prepare data

mkdir data/harry_potter_txt &&
wget "https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/data/harry_potter_txt/Book%201%20-%20The%20Philosopher's%20Stone.txt" -P data/harry_potter_txt/ &&
wget "https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/data/harry_potter_txt/Book%202%20-%20The%20Chamber%20of%20Secrets.txt" -P data/harry_potter_txt/ &&
wget "https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/data/harry_potter_txt/Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt" -P data/harry_potter_txt/ &&
wget "https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/data/harry_potter_txt/Book%204%20-%20The%20Goblet%20of%20Fire.txt" -P data/harry_potter_txt/ &&
wget "https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/data/harry_potter_txt/Book%205%20-%20The%20Order%20of%20the%20Phoenix.txt" -P data/harry_potter_txt/ &&
wget "https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/data/harry_potter_txt/Book%206%20-%20The%20Half%20Blood%20Prince.txt" -P data/harry_potter_txt/ &&
wget "https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/data/harry_potter_txt/Book%207%20-%20The%20Deathly%20Hallows.txt" -P data/harry_potter_txt/
"""


import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 5 # how many samples
sequence_length = 15 # seq len per sample
input_size = 10 # dimensions of vector for each timestep in seq per sample
hidden_size = 20
num_layers = 2

lstm = nn.LSTM(
	input_size=input_size,
	hidden_size=hidden_size,
	num_layers=num_layers,
	batch_first=True,
)

# we are generating some random data
rand = torch.rand(batch_size, sequence_length, input_size)

h0 = torch.zeros(num_layers, batch_size, hidden_size)
c0 = torch.zeros(num_layers, batch_size, hidden_size)
method_1_outs, (hn, cn) = lstm(rand, (h0, c0))

batch, seq_len, input_size = rand.shape

h = torch.zeros(num_layers, batch_size, hidden_size)
c = torch.zeros(num_layers, batch_size, hidden_size)
outs = []
for i in range(seq_len):
	token = rand[:,i,:].unsqueeze(1) # batch x 1 token X token X Input size -> unsqueeze returns back the sequence dimension
	out, (h, c) = lstm(token, (h, c))
	outs.append(out)

method_2_outs = torch.concat(outs, axis=1) #type:ignore

path_to_files = "data/harry_potter_txt/"
text_files = os.listdir(path_to_files)
all_txt = ""
for book in text_files:
	with open(os.path.join(path_to_files, book), "r") as f:
		text = f.readlines()
		text = [line for line in text if "Page" not in line]
		text = " ".join(text).replace("\n", "")
		text = [word for word in text.split(" ") if len(word) > 0]
		text = " ".join(text)
		all_txt += text

unique_chars = sorted(list(set(all_txt)))
char2idx = {c: i for i, c in enumerate(unique_chars)}
idx2char = {i: c for i, c in enumerate(unique_chars)}

class DataBuilder:
	def __init__(self, seq_len=300, text=all_txt) -> None:
		self.seq_len = seq_len
		self.text = text
		self.file_length = len(text)

	def grab_random_sample(self):
		start = np.random.randint(0, len(self.text) - self.seq_len) #type:ignore
		end = start+ self.seq_len
		text_slice = self.text[start:end]

		input_text = text_slice[:-1]
		label = text_slice[1:]

		input_text = torch.tensor([char2idx[c] for c in input_text])
		label = torch.tensor([char2idx[c] for c in label])
		return input_text, label

	def grab_random_batch(self, bach_size):
		input_texts, labels = [], []
		for _ in range(batch_size):
			input_text, label = self.grab_random_sample()
			input_texts.append(input_text)
			labels.append(label)

		input_texts = torch.stack(input_texts)
		labels = torch.stack(labels)
		return input_texts, labels

class LSTMGeneration(nn.Module):
	def __init__(self, embedding_dim=128, num_characters=len(char2idx), hidden_size=256, n_layers=3, device="cpu") -> None:
		super().__init__()
		self.embedding_dim = embedding_dim
		self.num_characters = num_characters
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.device = device

		self.embedding = nn.Embedding(
			self.num_characters,
			self.embedding_dim,
		)

		self.lstm = nn.LSTM(
			input_size=self.embedding_dim,
			hidden_size=self.hidden_size,
			num_layers=self.n_layers,
			batch_first=True
		)

		self.fc = nn.Linear(
			self.hidden_size,
			self.num_characters
		)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		batch_size, seq_len = x.shape
		x = self.embedding(x)
		h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
		c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
		output, (hn, cn) = self.lstm(x, (h0, c0))
		out = self.fc(output)
		return out

	def write(self, text, max_characters, train=True):
		idx = torch.tensor([char2idx[c] for c in text]).to(self.device)

		# TODO: if they break use the size (n_layer)
		hidden = torch.zeros(self.n_layers, self.hidden_size).to(self.device)
		cell = torch.zeros(self.n_layers, self.hidden_size).to(self.device)
		for i in range(max_characters):
			if i != 0:
				# After the first iteration, we use the last predicted char to predict the next one
				selected_idx = idx[-1].unsqueeze(0)
			else:
				# In the first iteration, we want to build up the hidden and cell state with the input chars
				selected_idx = idx
			x = self.embedding(selected_idx)
			out, (hidden, cell) = self.lstm(x, (hidden, cell))
			out = self.fc(out)

			# In the first iteration, we use all the character to build the H and C but we only use the last token for prediction
			if len(out) > 1:
				out = out[-1, :].unsqueeze(0)
			probs = self.softmax(out) # take the softmax along character dimensions to probability vector

			if train:
				idx_next = torch.multinomial(probs, num_samples=1)
			else:
				idx_next = torch.argmax(probs)
			idx = torch.cat([idx, idx_next[0]])
		gen_string = "".join([idx2char[int(c)] for c in idx])
		return gen_string


epochs = 3000
max_len = 300
evaluate_interval = 300
embedding_dim = 128
hidden_size = 256
n_layers = 3
lr = 0.003
batch_size = 128

device = "cpu"

model = LSTMGeneration(
	embedding_dim=embedding_dim,
	num_characters=len(char2idx),
	n_layers=n_layers,
	device=device
)

model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

loss_fn = nn.CrossEntropyLoss()
dataset = DataBuilder()

for epoch in range(epochs):
	input_texts, labels = dataset.grab_random_batch(batch_size)
	input_texts, labels = input_texts.to(device), labels.to(device)
	optimizer.zero_grad()

	out = model.forward(input_texts) # [batch, seq_len, n_chars]
	out = out.transpose(1, 2) # [batch, n_chars, seq_len]

	loss = loss_fn(out, labels)
	loss.backward()
	optimizer.step()

	if epoch % evaluate_interval == 0:
		print("-"*15)
		print(f"epoch {epoch}")
		print(f"loss {loss.item()}")

		generate_text = model.write("Spells ", max_characters=200)
		print("Sample Generation")
		print(generate_text)
		print("-"*15)
