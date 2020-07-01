# unit ini hanya untuk melakukukan proses perbaikan kalimat dan pharagrap

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import string
import contractions

punct_list = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
specials = ["’", "‘", "´", "`"]

def remove_specail_signs(x):
    for ss in specials:
        x = re.sub(ss, '', x)
    return x

def fix_text(df):
    dfbaru = []
    for row in df:
        xrow = ''.join([char if ord(char) < 128 else '' for char in row])
        xrow = contractions.fix(xrow.lower())
        xrow = remove_specail_signs(xrow)
        dfbaru.append(xrow)
    return dfbaru

def delete_double_char(substr, textd):
    while True:
        textd = re.sub(substr, ' ', textd)
        if substr in textd:
            pass
        else:
            break
    return textd


def delete_dot_at_last(sentence):
    num_last = len(sentence) - 1
    if sentence[num_last] == '.' :
        new_sent = sentence[:-1]
    else:
        new_sent = sentence

    #remove punct
    new_sent = re.sub('['+string.punctuation+']', ' ', new_sent)

    new_sent = delete_double_char(' s ', new_sent)
    new_sent = delete_double_char('\n', new_sent)
    new_sent = delete_double_char('  ', new_sent)

    return new_sent

def separate_sentence(text):
    stoplist = stopwords.words('english')
    arr_sentence = []
    new_text = sent_tokenize(text)
    for row in new_text:
        new_sentence = row
        arr_sentence.append('[CLS]' + delete_dot_at_last(new_sentence) +'[SEP]')
    return arr_sentence

def df_to_array(df):
    dfbaru = []
    for row in df:
        new_row = separate_sentence(row)
        dfbaru.append(new_row)
    return dfbaru

def change_stance(df):
    dfbaru = []
    for row in df:
        if row == 'agree':
            dfbaru.append(0)
        elif row == 'disagree':
            dfbaru.append(1)
        elif row == 'discuss':
            dfbaru.append(2)
        else:
            dfbaru.append(3)
    return dfbaru
