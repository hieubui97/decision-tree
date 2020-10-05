# Character level language model - Dinosaurus land¶
# Welcome to Dinosaurus Island! 65 million years ago, dinosaurs existed, and in this assignment they are back. You are in charge of a special task. Leading biology researchers are creating new breeds of dinosaurs and bringing them to life on earth, and your job is to give names to these dinosaurs. If a dinosaur does not like its name, it might go berserk, so choose wisely!
#
# Luckily you have learned some deep learning and you will use it to save the day. Your assistant has collected a list of all the dinosaur names they could find, and compiled them into this dataset. (Feel free to take a look by clicking the previous link.) To create new dinosaur names, you will build a character level language model to generate new names. Your algorithm will learn the different name patterns, and randomly generate new names. Hopefully this algorithm will keep you and your team safe from the dinosaurs' wrath!
#
# By completing this assignment you will learn:
#
# How to store text data for processing using an RNN
# How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
# How to build a character-level text generation recurrent neural network
# Why clipping the gradients is important
# We will begin by loading in some functions that we have provided for you in rnn_utils. Specifically, you have access to functions such as rnn_forward and rnn_backward which are equivalent to those you've implemented in the previous assignment.



from __future__ import print_function
import numpy as np
from utils import *
import random



data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))


char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
print(ix_to_char)


### GRANDED FUNCTION :clip
def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients

np.random.seed(3)
dWax = np.random.randn(5, 3) * 10
dWaa = np.random.randn(5, 5) * 10
dWya = np.random.randn(2, 5) * 10
db = np.random.randn(5, 1) * 10
dby = np.random.randn(2, 1) * 10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, 10)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])


def sample(parameters, char_to_ix, seed):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))

    a_prev = np.zeros((n_a, 1))

    indices = []

    idx = -1

    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        np.random.seed(counter + seed)

        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        a_prev = a

        seed += 1
        counter += 1

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices

np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:", indices)
print("list of sampled characters:", [ix_to_char[i] for i in indices])

def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    gradients, a = rnn_backward(X, Y, parameters, cache)

    gradients = clip(gradients, 5)

    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]

np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12, 3, 5, 11, 22, 3]
Y = [4, 14, 11, 22, 25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate=0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])


def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27):

    n_x, n_y = vocab_size, vocab_size

    parameters = initialize_parameters(n_a, n_x, n_y)

    loss = get_initial_loss(vocab_size, dino_names)

    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))

    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        loss = smooth(loss, curr_loss)

        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            seed = 0
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters

parameters = model(data, ix_to_char, char_to_ix)

# exec('from _future_ import print_function')
# from keras.callbacks import LambdaCallback
# from keras.models import Model, load_model, Sequential
# from keras.layers import Dense, Activation, Dropout, Input, Masking
# from keras.layers import LSTM
# from keras.utils.data_utils import get_file
# from keras.preprocessing.sequence import pad_sequences
# from shakespeare_utils import *
# import sys
# import io
#
# print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
#
# model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
#
# generate_output()