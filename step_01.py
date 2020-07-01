from unit03 import fix_text
from unit03 import df_to_array

from unit03 import change_stance
from nltk import sent_tokenize
import nltk

import pandas as pd
import numpy as np

print('Read Source Dataset..')
df = pd.read_csv("./real_dataset/source_dataset.csv")

df_headline = df_to_array(fix_text(df['Headline']))
df_bodies = df_to_array(fix_text(df['articleBody']))
df_y = change_stance(df['Stance'])

#----------------------
#process for df_bodies
#----------------------

df_bodies_lg = []
i = -1
t_sentence = -1
for allrow in df_bodies:
    i = i + 1
    for row in allrow:
        df_bodies_lg.append([row, i, df_y[i]])
        t_sentence +=1

sbodies = []
numrec = []
dfy = []

for row in df_bodies_lg:
    sbodies.append(row[0])
    numrec.append(row[1])
    dfy.append(row[2])

bodiesx = {
'sbodies' : sbodies ,
'numrec' : numrec,
'dfy' : dfy
}

hasilx = pd.DataFrame(bodiesx, columns = ['sbodies', 'numrec', 'dfy'])

hasilx.to_csv('./real_dataset/step3bodies.csv')
print('Body Dataset created..')
print('Total Sentence : ' , t_sentence)

#----------------------
#process for df_bodies
#----------------------

sheadlines = []
numrech = []
dfyh = []

i = -1
for row in df_headline:
    i += 1
    sheadlines.append(row[0])
    numrech.append(i)
    dfyh.append(df_y[i])

headlinesx = {
'sheadline' : sheadlines ,
'numrech' : numrech,
'dfyh' : dfyh
}

hasilx = pd.DataFrame(headlinesx, columns = ['sheadline', 'numrech', 'dfyh'])

hasilx.to_csv('./real_dataset/step3headlines.csv')
print('Headline Dataset created..')
print('Total Sentence : ' , i)
