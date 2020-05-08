import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

def load_image(image_path, transform=None):
  image = Image.open(image_path).convert('RGB')
  image = image.resize([224, 224], Image.LANCZOS)
  
  if transform is not None:
    image = transform(image).unsqueeze(0)
  
  return image

class Neuraltalk2:

  def __init__(self):
    print("Defining I.A")
    # Device configuration
    self.device = torch.device('cpu')

    #vars
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    encoder_path = 'models/encoder-5-3000.pkl'
    decoder_path = 'models/decoder-5-3000.pkl'
    vocab_path = 'data/vocab.pkl'

    # Image preprocessing
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    with open(vocab_path, 'rb') as f:
      self.vocab = pickle.load(f)

    print("Building Model")
    # Build models
    self.encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    self.decoder = DecoderRNN(embed_size, hidden_size, len(self.vocab), num_layers)
    self.encoder = self.encoder.to(self.device)
    self.decoder = self.decoder.to(self.device)

    print("loading checkpoint")
    # Load the trained model parameters
    self.encoder.load_state_dict(torch.load(encoder_path))
    self.decoder.load_state_dict(torch.load(decoder_path))

  def eval_image(self, image_path):
    # Prepare an image
    image = load_image(image_path, self.transform)
    image_tensor = image.to(self.device)
    
    # Generate an caption from the image
    feature = self.encoder(image_tensor)
    sampled_ids = self.decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
      word = self.vocab.idx2word[word_id]
      if word == '<end>':
        break
      if word == '<start>':
        continue
      sampled_caption.append(word)
        
    sentence = ' '.join(sampled_caption)
    return sentence

if __name__ == '__main__':
  nt2 = Neuraltalk2()
  print(nt2.eval_image("png/kids.jpg"))