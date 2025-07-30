import re
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load and map IDs to lines
lines = open('data/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    parts = line.split(" +++$+++ ")
    if len(parts) == 5:
        id2line[parts[0]] = parts[4]

qa_pairs = []
for conv in conversations:
    ids = conv.split(" +++$+++ ")[-1].replace("[", "").replace("]", "").replace("'", "").split(", ")
    for i in range(len(ids) - 1):
        if ids[i] in id2line and ids[i+1] in id2line:
            qa_pairs.append((id2line[ids[i]], id2line[ids[i+1]]))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return text

questions, answers = zip(*qa_pairs)
questions = [clean_text(q) for q in questions]
answers = ["<sos> " + clean_text(a) + " <eos>" for a in answers]

# Tokenize
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(questions + answers)

questions_seq = tokenizer.texts_to_sequences(questions)
answers_seq = tokenizer.texts_to_sequences(answers)

max_len = 20
questions_padded = pad_sequences(questions_seq, maxlen=max_len, padding='post')
answers_padded = pad_sequences(answers_seq, maxlen=max_len, padding='post')

np.savez("data/processed_data.npz", 
         questions=questions_padded, 
         answers=answers_padded, 
         word_index=tokenizer.word_index)

import pickle
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
