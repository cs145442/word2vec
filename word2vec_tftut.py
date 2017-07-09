import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# HYPERPARAMETERS
WINDOW_SIZE = 2
EMBEDDING_DIM = 5
n_iters = 10000

# corpus_raw = 'My name is Aneesh . I like chicken wings . chicken wings are very tasty .'
corpus_raw = 'He is the king . The king is royal . She is the royal queen .'
corpus_raw = corpus_raw.lower()

words = []

for word in corpus_raw.split():
    if word != '.':
        words.append(word)

words = set(words)
word2int = {}
int2word = {}
vocab_size = len(words) 

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# inv_map = {v: k for k, v in my_map.items()}

# PREPROCESSING
raw_sentences = corpus_raw.split('.')

sentences = []

for sentence in raw_sentences:
    sentences.append(sentence.split())

data = []

for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                data.append([word, nb_word])

numeric_data = []

for datapoint in data:
    numeric_data.append([word2int[datapoint[0]], word2int[datapoint[1]]] )


def to_one_hot(data_point):
    temp = np.zeros(vocab_size)
    temp[data_point] = 1
    return temp

x_train = []
y_train = []

for data in numeric_data:
    x_train.append(to_one_hot(data[0]))
    y_train.append(to_one_hot(data[1]))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# print(x_train)

# lol

W = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b = tf.Variable(tf.random_normal([EMBEDDING_DIM]))

W1 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b1 = tf.Variable(tf.random_normal([vocab_size]))

x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

hidden_representation = tf.add(tf.matmul(x,W), b)
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W1), b1))

# print(sess.run(W).shape)

# test_feed = to_one_hot(numeric_data[0][0], vocab_size).reshape(1, vocab_size)

# print(sess.run(prediction, feed_dict={x: test_feed}))
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_loss)



# n_iters = 3
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))


vectors = sess.run(W)

# print('The vectors are ',vectors)



def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

# print(euclidean_dist(vectors[2], vectors[1]))

# vec1 = vectors[0]
# vec2 = vectors[1]

# print(euclidean_dist(vec1, vec2))

def find_closest(word_index, vectors):
    min_dist = 10000
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        # vector.all() != query_vector.all()
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


print(int2word[find_closest(word2int['king'], vectors)])
print(int2word[find_closest(word2int['queen'], vectors)])
print(int2word[find_closest(word2int['royal'], vectors)])


model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors) 

import sklearn
vectors = sklearn.preprocessing.normalize(vectors, norm='l2')

fig, ax = plt.subplots()

print(words)

for word in words:
    print(word, vectors[word2int[word]][1])
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))

ax.annotate('lol', (1, 1))

print('REDUCED',vectors)

plt.show()