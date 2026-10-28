import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

print('Step 1: Loading dataset ...')
data = fetch_20newsgroups(subset='all')
texts, labels = data.data, data.target
num_classes = len(set(labels))
print(f'Loaded {len(texts)} documents across {num_classes} classes.')

print('\nStep 2: Preprocessing text ...')
max_words = 10000
max_len = 300
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)
y = to_categorical(labels, num_classes=num_classes)
print('Text converted to padded sequences.')

print('\nStep 3: Splitting into train and test ...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=labels
)
print(f'Training samples: {len(X_train)}, Testing samples: {len(X_test)}')

print('\nStep 4: Building deep learning model ...')
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128),
    LSTM(128),
    Dense(num_classes, activation='softmax')
])
model.build(input_shape=(None, max_len))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print('\nStep 5: Training model ...')
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)
print('\nStep 6: Evaluating model on test set ...')
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy:.4f}')
