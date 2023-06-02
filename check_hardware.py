import torch

if torch.cuda.is_available():
  print('GPU available')
else:
  print('Please enable GPU.')
  
