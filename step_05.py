#
# combine body, headline, and stance in one file
#
import numpy as np
import pandas as pd

df2 = np.load('./real_dataset/headlinesakhir.npy')
df2 = np.array(df2,dtype = 'float32')

df1 = np.load('./real_dataset/bodyakhir.npy', allow_pickle=True)

df_stance = pd.read_csv("./real_dataset/step3headlines.csv")

np_df_stance = np.array(df_stance['dfyh'])

ts = []
for i in range(len(df_stance)):
    tsbaris = []
    tsbaris.append(i)
    tsbaris.append(np.array(df1[i]))
    tsbaris.append(np.array(df2[i]))
    tsbaris.append(np_df_stance[i])
    ts.append(tsbaris)

trbaru = np.array(ts)

np.save('./real_dataset/hasil_combine.npy', trbaru)
