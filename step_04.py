#
# this program will concatenate all sentence
#
import numpy as np
import pandas as pd

print('read step3bodies.csv')
df1 = pd.read_csv('./real_dataset/step3bodies.csv')

panjang = len(df1)

print('read step3headlines.csv')
df2 = pd.read_csv('./real_dataset/step3headlines.csv')

hitungbnykkalimat = 0
listbnykkalimat = []
hitungawal = -1
no_kalimat = 0
kumpulawal = np.array([])
hasilsemua = []
for bd1 in range(99):
    nmfile = './npy_body/test-'+ str(bd1).zfill(3)+ '.npy'
    print('read ', nmfile)
    np1 = np.load(nmfile)
    for i in range(len(np1)):
        hitungawal += 1
        if (no_kalimat == df1['numrec'][hitungawal]):
            hitungbnykkalimat += 1
            kumpulawal = np.concatenate((kumpulawal, np1[i,0]), axis=0)
            if ((bd1 == 98) and (i == (len(np1) - 1))):
                listbnykkalimat.append([no_kalimat, hitungbnykkalimat])
                hasilsemua.append(kumpulawal)
        else:
            listbnykkalimat.append([no_kalimat, hitungbnykkalimat])
            hasilsemua.append(kumpulawal)
            kumpulawal = np.array([])
            kumpulawal = np.concatenate((kumpulawal, np1[i,0]), axis=0)
            no_kalimat += 1
            hitungbnykkalimat = 1

hasilsemuanp = np.array(hasilsemua)

print('Save to ./real_dataset/bodyakhir.npy')
np.save('./real_dataset/bodyakhir.npy', hasilsemuanp)

np2 = []
for bd1 in range(5):
    nmfile2 = './npy_head/test-'+ str(bd1).zfill(3)+ '.npy'
    print('read ', nmfile2)
    np1 = np.load(nmfile2)
    for i in range(len(np1)):
        np2.append(np1[i,0])

np1 = np.array(np2)

print('Save to ./real_dataset/headlinesakhir.npy')
np.save('./real_dataset/headlinesakhir.npy', np1)
