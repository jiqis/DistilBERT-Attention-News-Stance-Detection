# DistilBertModel Proccess for bodies Sentences
import torch
import transformers as ppb
import pandas as pd
import numpy as np
from numpy import save
import time

#read data bodies sentence from csv

print('Read step3bodies.csv')
df = pd.read_csv("./real_dataset/step3bodies.csv")

df_bodies = df['sbodies']

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# tokenized Proccess
tokenized = df_bodies.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# find Longests Sentence on bodies
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

# padding process
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

stc = 400
arr_pad = []
posisi = 0

while posisi <=  len(padded):
   arr_pad.append(padded[posisi: posisi + stc, :])
   posisi = posisi + stc

paddx = np.array(arr_pad)

for i in range(len(paddx)):
   print('Proccess : ', i)
   padded = paddx[i]
   attention_mask = np.where(padded != 0, 1, 0)
   input_ids = torch.tensor(padded).to(torch.int64)
   attention_mask = torch.tensor(attention_mask)
   with torch.no_grad():
       last_hidden_states = model(input_ids, attention_mask=attention_mask)

   np1 = last_hidden_states[0].numpy()
   save('./npy_body/test-'+ str(i).zfill(3)+ '.npy',np1)

   time.sleep(2.4)
