import os  # Allows to access files
from collections import (
    Counter,  # Utility function to give us the counts of unique items in an iterable
)

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image  # Allows us to Load Images
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import (
    ImageFolder,  # Stream data from images stored in folders
)

img_transforms = transforms.Compose(
	[
		transforms.Resize((224, 224)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
	]
)


class DogsVsCats(Dataset):
	def __init__(self, path_to_folder, transform) -> None:
		super().__init__()
		path_to_cats = os.path.join(path_to_folder, "Cat")
		path_to_dogs = os.path.join(path_to_folder, "Dog")
		dog_files = os.listdir(path_to_dogs)
		cat_files = os.listdir(path_to_cats)
		path_to_dog_files = [os.path.join(path_to_dogs, file) for file in dog_files]
		path_to_cat_files = [os.path.join(path_to_dogs, file) for file in cat_files]
		self.training_files = path_to_dog_files + path_to_cat_files
		self.dog_label, self.cat_label = 0,1
		self.transform = transform


	def __len__(self):
		return len(self.training_files)

	def __getitem__(self, index):
		path_to_image = self.training_files[index]
		label: int = self.dog_label if "Dog" in path_to_image else self.cat_label
		image = Image.open(path_to_image)
		image = self.transform(image)
		return image, label

dogvcat = DogsVsCats("PetImages/", img_transforms)
loader: DataLoader[DogsVsCats] = DataLoader(
	dogvcat,
	batch_size=16,
	shuffle=True,
)

train_samples = int(0.9*len(dogvcat))
test_samples = len(dogvcat) - train_samples
train_dataset, test_dataset = torch.utils.data.random_split(dogvcat, lengths=[train_samples, test_samples])

trainloader = DataLoader(
	train_dataset,
	batch_size=16,
	shuffle=True
)

testloader = DataLoader(
	test_dataset,
	batch_size=16,
	shuffle=True,
)

# This is the imdb text bit

path_to_data = "aclImdb/train"
path_to_pos_fld = os.path.join(path_to_data, "pos")
path_to_neg_fld = os.path.join(path_to_data, "neg")

path_to_pos_txt = [os.path.join(path_to_pos_fld, file) for file in path_to_pos_fld]
path_to_neg_txt = [os.path.join(path_to_neg_fld, file) for file in path_to_neg_fld]

training_files = path_to_pos_txt + path_to_neg_txt

alltxt = ""

for file in training_files:
	with open(file, "r") as f:
		text = f.readlines()
		alltxt += text[0]

unique_counts = dict(Counter(alltxt))
chars = sorted([key for (key, value) in unique_counts.items()])
chars.append("<UNK>")
chars.append("<PAD>")
char2idx = {c:i for i, c in enumerate(chars)}
idx2char = {i:c for i, c in enumerate(chars)}

class IMDBDataset(Dataset):
	def __init__(self) -> None:
		super().__init__()
		path_to_data = "aclImdb/train"
		path_to_pos_fld = os.path.join(path_to_data, "pos")
		path_to_neg_fld = os.path.join(path_to_data, "neg")

		path_to_pos_txt = [os.path.join(path_to_pos_fld, file) for file in path_to_pos_fld]
		path_to_neg_txt = [os.path.join(path_to_neg_fld, file) for file in path_to_neg_fld]

		self.training_files = path_to_pos_txt + path_to_neg_txt
		self.tokenizer = char2idx

	def __len__(self) -> int:
		return len(self.training_files)

	def __getitem__(self, index):
		path_to_txt = self.training_files[index]
		with open(path_to_txt, "r") as f:
			txt = f.readlines()[0]
		tokenized = []
		for char in txt:
			if char in self.tokenizer.keys():
				tokenized.append(self.tokenizer[char])
			else:
				tokenized.append(self.tokenizer["<UNK>"])

		sample = torch.tensor(tokenized)
		label = 0 if "neg" in path_to_txt else 1
		return sample, label
