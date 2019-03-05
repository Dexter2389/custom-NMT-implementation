import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu


def minibatches(inputs, targets, minibatch_size):
    """
    Batch Generator.

    Yields x and y batch
    """
    x_batch, y_batch = [], []
    for inp, tgt, in zip(inputs, targets):
        if len(x_batch) == minibatch_size and len(y_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch.append(inp)
        y_batch.append(tgt)

    if len(x_batch) != 0:
        for inp, tgt in zip(inputs, targets):
            if len(x_batch) != minibatch_size:
                x_batch.append(inp)
                y_batch.append(tgt)
            else:
                break
        yield x_batch, y_batch


def pad_sequences(sequences, pad_token, tail=True):
    """
    Pads the sentences, so that all the sentences in a batch have the same length
    """
    max_length = max(len(x) for x in sequences)

    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        if tail:
            seq_ = seq[:max_length] + [pad_token] * \
                max(max_length - len(seq), 0)
        else:
            seq_ = [pad_token] * \
                max(max_length - len(seq), 0) + seq[:max_length]

        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def reset_graph(seed=98):
    """
    Helper Function to reset the default Graph
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def sample_results(preds, lan1_index2word, lan2_index2word, lan1_word2index, lan2_word2index, lan2_indexs, lan1_indexs):
    beam = False
    if len(np.array(preds).shape) == 4:
        beam = True

    """
    BLEU Score calculation
    """
    bleu_scores = []

    for pred, de, en, seq_length in zip(preds[0],
                                        lan2_indexs,
                                        lan1_indexs,
                                        [len(inds) for inds in lan2_indexs]):
        print('\n\n\n', 100 * '-')

        if beam:
            actual_text = [lan1_index2word[word] for word in reversed(en) if
                           word != lan1_word2index["<s>"] and word != lan1_word2index["</s>"]]
            actual_translation = [lan2_index2word[word] for word in de if
                                  word != lan2_word2index["<s>"] and word != lan2_word2index["</s>"]]
            created_translation = []
            for word in pred[:seq_length]:
                if word[0] != lan2_word2index['</s>'] and word[0] != lan2_word2index['<s>']:
                    created_translation.append(lan2_index2word[word[0]])
                    continue
                else:
                    continue

            bleu_score = sentence_bleu(
                [actual_translation], created_translation)
            bleu_scores.append(bleu_score)

            print('Actual Text:\n{}\n'.format(' '.join(actual_text)))
            print('Actual translation:\n{}\n'.format(
                ' '.join(actual_translation)))
            print('Created translation:\n{}\n'.format(
                ' '.join(created_translation)))
            print('Bleu-score:', bleu_score)
            print()

        else:

            actual_text = [lan1_index2word[word] for word in reversed(en) if
                           word != lan1_word2index["<s>"] and word != lan1_word2index["</s>"]]
            actual_translation = [lan2_index2word[word] for word in de if word !=
                                  lan2_word2index["<s>"] and word != lan2_word2index["</s>"]]
            created_translation = [lan2_index2word[word] for word in pred if word !=
                                   lan2_word2index["<s>"] and word != lan2_word2index["</s>"]][:seq_length]
            bleu_score = sentence_bleu(
                [actual_translation], created_translation)
            bleu_scores.append(bleu_score)

            print('Actual Text:\n{}\n'.format(' '.join(actual_text)))
            print('Actual translation:\n{}\n'.format(
                ' '.join(actual_translation)))
            print('Created translation:\n{}\n'.format(
                ' '.join(created_translation)))
            print('Bleu-score:', bleu_score)

    bleu_score = np.mean(bleu_scores)
    print('\n\n\nTotal Bleu Score:', bleu_score)
