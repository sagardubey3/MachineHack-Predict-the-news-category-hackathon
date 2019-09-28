# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 00:12:58 2019

@author: sagar
"""
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from keras.layers import Flatten, Conv1D, MaxPooling1D, Dropout, Activation


train=pd.read_excel('Data.xlsx')
train.head()

xtrain, xvalid, ytrain, yvalid = train_test_split(train['Processed_article'].values, train['SECTION'], 
                                                  stratify=train['SECTION'], 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
     
#EMBEDDING
EMBEDDING_FILE = 'glove.840B.300d.txt'
def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding='utf8'))

print('Found %s word vectors.' % len(embeddings_index))

def sent2vec(s):
    words = str(s).lower()#.decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

xtrain_glove = [sent2vec(x) for x in tqdm(xtrain)]
xvalid_glove = [sent2vec(x) for x in tqdm(xvalid)]

xtrain_glove = np.array(xtrain_glove)
xvalid_glove = np.array(xvalid_glove)

yvalid_enc = np_utils.to_categorical(yvalid)
ytrain_enc = np_utils.to_categorical(ytrain)

scl = preprocessing.StandardScaler()
xtrain_glove_scl = scl.fit_transform(xtrain_glove)
xvalid_glove_scl = scl.transform(xvalid_glove)

token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector





model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=5, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])



test = pd.read_excel('Data_Test.xlsx')

seq = token.texts_to_sequences(test['STORY'])
padded = pad_sequences(seq, maxlen=70)
pred=model.predict_classes(padded)

submission = pd.DataFrame({'SECTION':pred})
submission.to_excel('submission5.xlsx', index=False)


