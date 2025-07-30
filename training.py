import numpy as np
import os
import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# ========== Load and Prepare Data ==========

def load_conversations():
    print("Loading data from files...")
    with open("data/movie_lines.txt", encoding='utf-8', errors='ignore') as f:
        lines = f.read().split("\n")
    with open("data/movie_conversations.txt", encoding='utf-8', errors='ignore') as f:
        conv_lines = f.read().split("\n")

    id2line = {}
    for line in lines:
        parts = line.split(" +++$+++ ")
        if len(parts) == 5:
            id2line[parts[0]] = parts[4]

    conversations = []
    for conv in conv_lines:
        parts = conv.split(" +++$+++ ")
        if len(parts) == 4:
            utterance_ids = eval(parts[3])
            for i in range(len(utterance_ids) - 1):
                input_line = id2line.get(utterance_ids[i], "")
                target_line = id2line.get(utterance_ids[i + 1], "")
                if input_line and target_line:
                    conversations.append((input_line, target_line))
    print(f"✅ Loaded {len(conversations)} conversations.")
    return conversations

conversations = load_conversations()[:20000]  # Use only first 20,000 conversations


questions, answers = zip(*conversations)

# Preprocess
def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    return text

questions = [clean_text(q) for q in questions]
answers = ['<START> ' + clean_text(a) + ' <END>' for a in answers]

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

VOCAB_SIZE = len(tokenizer.word_index) + 1

MAX_LEN = 20
encoder_input_data = pad_sequences(tokenizer.texts_to_sequences(questions), maxlen=MAX_LEN, padding='post')
decoder_input_data = pad_sequences(tokenizer.texts_to_sequences(answers), maxlen=MAX_LEN, padding='post')

# Remove <END> token from decoder target
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
decoder_target_data[:, -1] = 0

# ========== Build the Seq2Seq Model ==========

embedding_dim = 100
lstm_units = 256

# Encoder
encoder_inputs = Input(shape=(MAX_LEN,))
enc_emb = Embedding(VOCAB_SIZE, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(MAX_LEN,))
dec_emb = Embedding(VOCAB_SIZE, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Reshape target data to match output
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# ========== Train ==========

x_train_enc, x_test_enc, x_train_dec, x_test_dec, y_train, y_test = train_test_split(
    encoder_input_data, decoder_input_data, decoder_target_data, test_size=0.1)

model.fit([x_train_enc, x_train_dec], y_train, batch_size=64, epochs=10, validation_split=0.1)

# ========== Save Model and Tokenizer ==========

if not os.path.exists('model'):
    os.makedirs('model')

model.save('model/chatbot_model.h5')
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ Training complete. Model and tokenizer saved.")
