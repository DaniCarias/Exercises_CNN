# pip install datasets transformers

from datasets import load_dataset

#We'll use the beans dataset, which is a collection of pictures of healthy and unhealthy bean leaves.
ds = load_dataset('beans')
ds

labels = ds['train'].features['labels']
labels



