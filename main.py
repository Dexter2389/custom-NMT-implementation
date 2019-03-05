from collections import Counter
import nmt
import data_utils
import model_utils
import tensorflow as tf
import nltk
nltk.download("punkt")


# Loading Lan1 Data - English Text
with open("europarl-v7.de-en.en", "r", encoding="utf-8") as f:
    lan1 = f.readlines()

# Loading Lan2 Data - German Text
with open("europarl-v7.de-en.de", "r", encoding="utf-8") as f:
    lan2 = f.readlines()

# Cleaning the text for "\n" token
lan1 = [line.strip() for line in lan1]
lan2 = [line.strip() for line in lan2]

_lan2 = []
_lan1 = []

for sent_lan1, sent_lan2 in zip(lan1, lan2):
    if len(sent_lan1) < 10:           #Appends only those sentences that are less then 50 words
        _lan2.append(sent_lan2)
        _lan1.append(sent_lan1)


lan1_preprocessed, lan1_most_common = data_utils.preprocess(
    _lan1)
lan2_preprocessed, lan2_most_common = data_utils.preprocess(
    _lan2, language="german")

lan1_preprocessed_clean, lan2_preprocessed_clean = [], []
for sent_lan1, sent_lan2 in zip(lan1_preprocessed, lan2_preprocessed):
    if sent_lan1 != [] and sent_lan2 != []:
        lan1_preprocessed_clean.append(sent_lan1)
        lan2_preprocessed_clean.append(sent_lan2)
    else:
        continue

# Creating our lookup dict for lan1 and lan2, i.e. our vocab
specials = ["<unk>", "<s>", "</s>", "<pad>"]
lan1_word2index, lan1_index2word, lan1_vocab_size = data_utils.create_vocab(
    lan1_most_common, specials)
lan2_word2index, lan2_index2word, lan2_vocab_size = data_utils.create_vocab(
    lan2_most_common, specials)

lan1_index, lan1_unknowns = data_utils.convert_to_indices(
    lan1_preprocessed_clean, lan1_word2index, reverse=True, eos=True)
lan2_index, lan2_unknowns = data_utils.convert_to_indices(
    lan2_preprocessed_clean, lan2_word2index, sos=True, eos=True)

# Training the model

# Hyperparams
num_layers_encoder = 7
num_layers_decoder = 7
rnn_size_encoder = 128
rnn_size_decoder = 128
embedding_dim = 300

batch_size = 64
epochs = 10
clip = 6
keep_probability = 0.8
learning_rate = 0.01
learning_rate_decay_step = 5000
learning_rate_decay = 0.9

# Creating the graph
with tf.device("/device:GPU:0"):
    model_utils.reset_graph()
    nmt1 = nmt.NMT(lan1_word2index, lan1_index2word, lan2_word2index, lan2_index2word, "./models/model", "TRAIN", embedding_dim=embedding_dim, num_layers_encoder=num_layers_encoder, num_layers_decoder=num_layers_decoder, batch_size=batch_size, clip=clip, keep_probability=keep_probability, learning_rate=learning_rate, epochs=epochs, rnn_size_encoder=rnn_size_encoder, rnn_size_decoder=rnn_size_decoder, learning_rate_decay=learning_rate_decay, learning_rate_decay_steps=learning_rate_decay_step)
    nmt1.build_graph()
    nmt1.train(lan1_index, lan2_index, "./models/model.ckpt")


##########
# Testing
# for line in zip(lan1_preprocessed_clean[:5], lan2_preprocessed_clean[:5]):
#    print(line,"\n")
##########
#print(lan1_vocab_size, lan2_vocab_size)
#print(len(lan1_vocab_size), ",", len(lan2_vocab_size))

"""
_de_inds, _de_unknowns = data_utils.convert_to_indices(lan2_preprocessed_clean, lan2_word2index, sos = True,  eos = True)
model_utils.reset_graph()

nmt2 = nmt.NMT(lan1_word2index, lan1_index2word, lan2_word2index, lan2_index2word, './models/model', 'INFER', num_layers_encoder = num_layers_encoder, num_layers_decoder = num_layers_decoder, batch_size = len(lan1_index[:50]), keep_probability = 1.0, learning_rate = 0.0, beam_width = 0, rnn_size_encoder = rnn_size_encoder, rnn_size_decoder = rnn_size_decoder)

nmt2.build_graph()
preds = nmt2.infer(lan1_index[:50], restore_path =  './models/model', targets = _de_inds[:50])

model_utils.sample_results(preds, lan1_index2word, lan2_index2word, lan1_word2index, lan2_word2index, _de_inds[:50], lan1_index[:50])
"""
