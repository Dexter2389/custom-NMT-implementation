import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
import model_utils


class NMT():
    def __init__(self, word2index_lan1, index2word_lan1, word2index_lan2, index2word_lan2, save_path, mode="TRAIN", num_layers_encoder=None, num_layers_decoder=None, embedding_dim=300, rnn_size_encoder=256, rnn_size_decoder=256, keep_probability=0.85, batch_size=64, beam_width=10, epochs=30, eos="</s>", sos="<s>", pad="<pad>", use_gru=False, time_major=False, clip=5, learning_rate=0.001, learning_rate_decay=0.9, learning_rate_decay_steps=125):
        self.word2index_lan1 = word2index_lan1
        self.word2index_lan2 = word2index_lan2
        self.index2word_lan1 = index2word_lan1
        self.index2word_lan2 = index2word_lan2
        self.save_path = save_path
        self.vocab_size_lan1 = len(word2index_lan1)
        self.vocab_size_lan2 = len(word2index_lan2)
        self.embedding_dim = embedding_dim
        self.mode = mode.upper()
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.rnn_size_encoder = rnn_size_encoder
        self.rnn_size_decoder = rnn_size_decoder
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.keep_probability = keep_probability
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.eos = eos
        self.sos = sos
        self.time_major = time_major
        self.clip = clip
        self.pad = pad
        self.use_gru = use_gru
        self.epochs = epochs

    def build_graph(self):
        self.add_placeholders()
        self.add_embeddings()
        self.add_lookup_ops()
        self.initialize_session()
        self.add_seq2seq()
        print("\nInitial Graph Built :-)\n")

    def add_placeholders(self):
        self.ids_lan1 = tf.placeholder(
            tf.int32, shape=[None, None], name="ids_source")
        self.ids_lan2 = tf.placeholder(
            tf.int32, shape=[None, None], name="ids_target")
        self.sequence_length_lan1 = tf.placeholder(
            tf.int32, shape=[None], name="sequence_length_source")
        self.sequence_length_lan2 = tf.placeholder(
            tf.int32, shape=[None], name="sequence_length_target")
        self.maximum_iterations = tf.reduce_max(
            self.sequence_length_lan2, name="max_dec_len")

    def create_word_embedding(self, embed_name, vocab_size, embed_dim):
        """Creates a matrix of given shape - [vocab_size, embed_dim]"""
        embedding = tf.get_variable(
            embed_name, shape=[vocab_size, embed_dim], dtype=tf.float32)
        return embedding

    def add_embeddings(self):
        """Creates the embedding matrics for both the source and target language"""
        self.embedding_lan1 = self.create_word_embedding(
            "src_embedding", self.vocab_size_lan1, self.embedding_dim)
        self.embedding_lan2 = self.create_word_embedding(
            "tgt_embedding", self.vocab_size_lan2, self.embedding_dim)

    def add_lookup_ops(self):
        """Additional lookup operations for both source and target embedding matrix"""
        self.word_embedding_lan1 = tf.nn.embedding_lookup(
            self.embedding_lan1, self.ids_lan1, name="word_embeddings_source")
        self.word_embedding_lan2 = tf.nn.embedding_lookup(
            self.embedding_lan2, self.ids_lan2, name="word_embeddings_target")

    def make_rnn_cell(self, rnn_size, keep_probability):
        """Creates a LSTM cell or GRU cell, dropout wrapped"""
        if self.use_gru:
            #cell = tf.nn.rnn_cell.GRUCell(rnn_size)
            cell = tf.contrib.rnn.GRUBlockCellV2(rnn_size)
        else:
            #cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
            cell = tf.contrib.rnn.LSTMBlockCell(rnn_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, input_keep_prob=keep_probability)
        return cell

    def make_attention_cell(self, dec_cell, rnn_size, enc_output, lengths, alignment_history=False):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=rnn_size, memory=enc_output, memory_sequence_length=lengths, name='BahdanauAttention')
        return tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell, attention_mechanism=attention_mechanism, attention_layer_size=None, output_attention=False, alignment_history=alignment_history)

    def build_encoder(self):
        """Creates an encoder block with multiple LSTM-cells on top of a Bidirectional RNN"""
        with tf.variable_scope("encoder"):
            num_bi_layer = 1
            if self.num_layers_encoder is not None:
                num_uni_layers = self.num_layers_encoder - num_bi_layer
            else:
                num_uni_layers = 1

            fw_cell = self.make_rnn_cell(
                self.rnn_size_encoder, self.keep_probability)
            bw_cell = self.make_rnn_cell(
                self.rnn_size_encoder, self.keep_probability)

            bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.word_embedding_lan1, sequence_length=self.sequence_length_lan1, dtype=tf.float32)

            bi_encoder_outputs = tf.concat(bi_encoder_outputs, -1)

            uni_cell = tf.nn.rnn_cell.MultiRNNCell([self.make_rnn_cell(
                self.rnn_size_encoder, self.keep_probability) for _ in range(num_uni_layers)])

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                uni_cell, bi_encoder_outputs, sequence_length=self.sequence_length_lan1, dtype=tf.float32, time_major=self.time_major)

            encoder_state = (bi_encoder_state[1],) + (
                (encoder_state,) if self.num_layers_encoder is None else encoder_state)

            return encoder_outputs, encoder_state

    def build_decoder(self, encoder_outputs, encoder_state):
        sos_id_2 = tf.cast(self.word2index_lan2[self.sos], tf.int32)
        eos_id_2 = tf.cast(self.word2index_lan2[self.eos], tf.int32)

        self.output_layer = Dense(
            self.vocab_size_lan2, name="output_projection")

        # Decoder Cell
        with tf.variable_scope("decoder") as decoder_scope:

            cell, decoder_initial_state = self.build_decoder_cell(
                encoder_outputs, encoder_state, self.sequence_length_lan1)

            # Train
            if self.mode != "INFER":
                if self.time_major:
                    target_input = tf.transpose(self.ids_lan2)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.word_embedding_lan2, self.sequence_length_lan2, time_major=self.time_major)

                # Decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, helper, decoder_initial_state, output_layer=self.output_layer)

                # Dynamic Decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=self.time_major, maximum_iterations=self.maximum_iterations, swap_memory=False, impute_finished=True, scope=decoder_scope)

                sample_id = outputs.sample_id
                logits = outputs.rnn_output

            else:
                start_tokens = tf.fill([self.batch_size], sos_id_2)
                end_token = eos_id_2

                if self.beam_width > 0:
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell, embedding=self.embedding_lan2, start_tokens=start_tokens,
                                                                   end_token=end_token, initial_state=decoder_initial_state, beam_width=self.beam_width, output_layer=self.output_layer)

                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        self.embedding_lan2, start_tokens, end_token)
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell, helper, decoder_initial_state, output_layer=self.output_layer)

                # Dynamic Decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=self.maximum_iterations, output_time_major=self.time_major, impute_finished=False, swap_memory=False, scope=decoder_scope)

                if self.beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids

                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def build_decoder_cell(self, encoder_outputs, encoder_state, sequence_length_lan1):
        memory = encoder_outputs
        if self.mode == "INFER" and self.beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(
                memory, multiplier=self.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width)
            sequence_length_lan1 = tf.contrib.seq2seq.tile_batch(
                sequence_length_lan1, multiplier=self.beam_width)
            batch_size = self.batch_size * self.beam_width

        else:
            batch_size = self.batch_size

        # instead of MultiRNNcell we just use a cell list and pop the very bottom cell and wrap this one with attention, then using GNMTAttentionMultiRNNCell

        cell_list = [self.make_rnn_cell(
            self.rnn_size_decoder, self.keep_probability) for _ in range(self.num_layers_decoder)]
        lstm_cell = cell_list.pop(0)

        # boolean value wheter to use it or not --> only greedy inference

        alignment_history = (self.mode == 'INFER' and self.beam_width == 0)
        attention_cell = self.make_attention_cell(
            lstm_cell, self.rnn_size_decoder, memory, sequence_length_lan1, alignment_history=alignment_history)

        cell = GNMTAttentionMultiCell(attention_cell, cell_list)

        decoder_initial_state = tuple(zs.clone(cell_state=es) if isinstance(
            zs, tf.contrib.seq2seq.AttentionWrapperState) else es for zs, es in zip(cell.zero_state(batch_size, tf.float32), encoder_state))

        return cell, decoder_initial_state

    def compute_loss(self, logits):
        """Computes Loss during Optimization"""
        target_output = self.ids_lan2
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.maximum_iterations

        target_weights = tf.sequence_mask(
            self.sequence_length_lan2, max_time, dtype=tf.float32, name="mask")

        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.contrib.seq2seq.sequence_loss(
            logits=logits, targets=target_output, weights=target_weights, average_across_timesteps=True, average_across_batch=True)

        return loss

    def train(self, inputs, targets, restore_path=None):
        assert len(inputs) == len(targets)

        self.initialize_session()
        if restore_path is not None:
            if not os.path.exists(self.save_path):
                pass
            else:
                self.restore_session(restore_path)
        
        best_score = np.inf
        nepoch_no_imprv = 0

        inputs = np.array(inputs)
        targets = np.array(targets)

        for epoch in range(self.epochs + 1):
            print("-------- Epoch {} of {} --------".format(epoch, self.epochs))

            # Shuffle the input data before every epoch to add randomness
            shuffle_indices = np.random.permutation(len(inputs))
            inputs = inputs[shuffle_indices]
            targets = targets[shuffle_indices]
            
            # Run training epoch
            score = self.run_epoch(inputs, targets, epoch)

            if score <= best_score:
                nepoch_no_imprv = 0
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                self.saver.save(self.sess, self.save_path+".ckpt")
                best_score = score
                print("\n---- New Best Score ----\n\n")
                
            else:
                # warm up epochs for the model
                if epoch > 10:
                    nepoch_no_imprv += 1
                if nepoch_no_imprv >= 5:
                    print(
                        "--- Early stopping {} epochs without improvement ---".format(nepoch_no_imprv))
                    break

    def infer(self, inputs, restore_path, targets=None):
        self.initialize_session()
        self.restore_session(restore_path)

        prediction_ids = []
        feed, _, sequence_length_lan2 = self.get_feed_dict(
            inputs, trgts=targets)
        infer_logits, s_ids = self.sess.run(
            [self.infer_logits, self.sample_words], feed_dict=feed)
        prediction_ids.append(s_ids)

        return prediction_ids

    def run_epoch(self, inputs, targets, epoch):

        start_of_epoch = time.time()

        batch_size = self.batch_size
        nbatches = (len(inputs) + batch_size - 1) // batch_size
        losses = []

        for i, (inps, trgts) in enumerate(model_utils.minibatches(inputs, targets, batch_size)):
            if inps is not None and trgts is not None:
                start_of_iteration = time.time()
                feed_dict, s1, s2 = self.get_feed_dict(inps, trgts=trgts)
                _, train_loss = self.sess.run(
                    [self.train_op, self.train_loss], feed_dict=feed_dict)
                end_of_iteration = time.time()
                if i % 1 == 0 or i == (nbatches - 1):
                    print("[{}/{} epoch] Iteration: {} of {}\tTrain_loss: {:.4f}\tTime_taken: {:.4} seconds".format(
                        epoch, self.epochs, i, nbatches-1, train_loss, end_of_iteration-start_of_iteration))
                losses.append(train_loss)

            else:
                continue

        end_of_epoch = time.time()
        avg_loss = self.sess.run(tf.reduce_mean(losses))
        print("Average Score for this Epoch: {}\tTime to comple this Epoch: {:.4f} seconds".format(avg_loss, end_of_epoch-start_of_epoch))

        return avg_loss

    def get_feed_dict(self, inps, trgts=None):
        if self.mode != "INFER":
            input_ids, sequence_length_lan1 = model_utils.pad_sequences(
                inps, self.word2index_lan1[self.pad], tail=True)

            feed = {self.ids_lan1: input_ids,
                    self.sequence_length_lan1: sequence_length_lan1}

            if trgts is not None:
                target_ids, sequence_length_lan2 = model_utils.pad_sequences(
                    trgts, self.word2index_lan2[self.pad], tail=True)
                feed[self.ids_lan2] = target_ids
                feed[self.sequence_length_lan2] = sequence_length_lan2

            return feed, sequence_length_lan1, sequence_length_lan2

        else:
            input_ids, sequence_length_lan1 = model_utils.pad_sequences(
                inps, self.word2index_lan1[self.pad], tail=True)
            feed = {self.ids_lan1: input_ids,
                    self.sequence_length_lan1: sequence_length_lan1}

            if trgts is not None:
                target_ids, sequence_length_lan2 = model_utils.pad_sequences(
                    trgts, self.word2index_lan2[self.pad], tail=True)
                feed[self.sequence_length_lan2] = sequence_length_lan2

            return feed, sequence_length_lan1, sequence_length_lan2

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def restore_session(self, restore_path):
        self.saver.restore(self.sess, restore_path)
        print("Restoring Graph from ", restore_path)

    def add_seq2seq(self):
        """
        Creates the sequence to sequence architectural model
        """
        with tf.variable_scope("dynamic_seq2seq", dtype=tf.float32):

            # Encoder
            encoder_outputs, encoder_state = self.build_encoder()

            # Decoder
            logits, sample_id, final_context_state = self.build_decoder(
                encoder_outputs, encoder_state)

            if self.mode == "TRAIN":

                # Loss
                loss = self.compute_loss(logits)
                self.train_loss = loss
                self.word_count = tf.reduce_sum(
                    self.sequence_length_lan1) + tf.reduce_sum(self.sequence_length_lan2)
                self.predict_count = tf.reduce_sum(self.sequence_length_lan2)
                self.global_step = tf.Variable(0, trainable=False)

                # Optimizer
                self.learning_rate = tf.train.exponential_decay(
                    self.learning_rate, self.global_step, decay_steps=self.learning_rate_decay_steps, decay_rate=self.learning_rate_decay, staircase=True)
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                # Gradient
                if self.clip > 0:
                    grads, vs = zip(*opt.compute_gradients(self.train_loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, self.clip)
                    self.train_op = opt.apply_gradients(
                        zip(grads, vs), global_step=self.global_step)
                else:
                    self.train_op = opt.minimize(
                        self.train_loss, global_step=self.global_step)

                # Summary
                # will add afterwards

            elif self.mode == "INFER":
                loss = None
                self.infer_logits, _, self.final_context_state, self.sample_id = logits, loss, final_context_state, sample_id
                self.sample_words = self.sample_id

# RNN cell by google for neural machine translation --> from gnmt_model.py in the nmt repository.
# It inherits from the MultiRNNCell.


class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
    """A MultiCell with GNMT attention style."""

    def __init__(self, attention_cell, cells, use_new_attention=False):
        """Creates a GNMTAttentionMultiCell.
        Args:
          attention_cell: An instance of AttentionWrapper.
          cells: A list of RNNCell wrapped with AttentionInputWrapper.
          use_new_attention: Whether to use the attention generated from current
            step bottom layer's output. Default is False.
        """
        cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention
        super(GNMTAttentionMultiCell, self).__init__(
            cells, state_is_tuple=True)

    def __call__(self, inputs, state, scope=None):
        """Run the cell with bottom layer's attention copied to all upper layers."""
        if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.variable_scope(scope or "multi_rnn_cell"):
            new_states = []

            with tf.variable_scope("cell_0_attention"):
                attention_cell = self._cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(
                    inputs, attention_state)
                new_states.append(new_attention_state)

            for i in range(1, len(self._cells)):
                with tf.variable_scope("cell_%d" % i):

                    cell = self._cells[i]
                    cur_state = state[i]

                    if self.use_new_attention:
                        cur_inp = tf.concat(
                            [cur_inp, new_attention_state.attention], -1)
                    else:
                        cur_inp = tf.concat(
                            [cur_inp, attention_state.attention], -1)

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)
