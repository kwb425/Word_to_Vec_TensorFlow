# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import os
import random
import zipfile
import codecs
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#######################################################################################################################
# Word2vec with TensorBoard:
#                            1. Tensorboard features added
#                            2. Organized code
#                            3. Comments
#
#                                                                                               Advised by Kim, Wiback,
#                                                                                                     2017.05.11. v1.1.
#######################################################################################################################





## Preparation ########################################################################################################



######
# Data
######
LOGDIR = os.path.join(os.path.dirname(__file__), os.path.pardir, "log")
DATADIR = os.path.join(os.path.dirname(__file__), os.path.pardir, "data")
INPUT = os.path.join(DATADIR, "input")
LABEL = os.path.join(LOGDIR, "labels.tsv")
fid_read = codecs.open(INPUT, "r", encoding="utf-8")
words = [word.rstrip() for word in fid_read.readlines()]
fid_read.close()



########
# Params
########
""" This is what basically, we are doing.
[1 0 0 0 0 0 . . . 0]  FROM ONE HOT  ->  TO CONTINUOUS VECTOR SPACE  [0.21312 0.66112 0.28012 . . . 0.44711]
[0 1 0 0 0 0 . . . 0]  FROM ONE HOT  ->  TO CONTINUOUS VECTOR SPACE  [0.31415 0.91410 0.21312 . . . 0.23121]
[0 0 1 0 0 0 . . . 0]  FROM ONE HOT  ->  TO CONTINUOUS VECTOR SPACE  [0.91410 0.31231 0.31415 . . . 0.91410]
[. . . . . . . . . .]  FROM ONE HOT  ->  TO CONTINUOUS VECTOR SPACE  [0.31231 0.30212 0.91219 . . . 0.31231]
[. . . . . . . . . .]  FROM ONE HOT  ->  TO CONTINUOUS VECTOR SPACE  [0.30212 0.81291 0.12012 . . . 0.30212]
[. . . . . . . . . .]  FROM ONE HOT  ->  TO CONTINUOUS VECTOR SPACE  [0.91219 0.31231 0.51523 . . . 0.91219]
[. . . . . . . . . .]  FROM ONE HOT  ->  TO CONTINUOUS VECTOR SPACE  [0.12012 0.41241 0.71112 . . . 0.12012]
[0 0 0 0 0 0 . . . 1]  FROM ONE HOT  ->  TO CONTINUOUS VECTOR SPACE  [0.83191 0.91219 0.71461 . . . 0.51523]
"""
vocabulary_size = 50000  # For now, it is 50000.               # 2D tensor (vocabulary_size, embedding_size)
embedding_size = 128                                           # 2D tensor (vocabulary_size, embedding_size)
batch_size = 128                                               # How many samples for a single SGD step?
num_steps = 100001                                             # How many SGD steps?
skip_window = 1                                                # How many words to consider left and right?
num_skips = 2                                                  # How many times to reuse an input to generate a label.
num_sampled = 64                                               # Number of negative examples to sample.
"""
Below, we pick a random validation set to sample nearest neighbors,
we limit the validation samples to the words that have
a low numeric ID, which by construction are also the most frequent.
"""
valid_size = 16                                                # Random set of words to evaluate similarity on.
valid_window = 100                                             # Only pick dev samples in the head of the distribution.



###################
## Build Dictionary
###################
def build_dataset(words_param, vocabulary_size_param):
  count = [[u'UNK', -1]]
  count.extend(collections.Counter(words_param).most_common(vocabulary_size_param - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words_param:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
del words  # Hint to reduce memory.
vocabulary_size = len(reverse_dictionary)  # Changing for line-by-line mapping between the labels & embeddings



####################################
# Extracting the label from the dict
####################################
fid = codecs.open(LABEL, "w", encoding="utf-8")
for i in range(len(reverse_dictionary)):
    fid.write(u"{}\n".format(reverse_dictionary[i]))
fid.close()



########################
## Generating Batch func
########################
data_index = 0  # Dummy
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels





## Graphing the Skip-gram #############################################################################################
graph = tf.Graph()       
with graph.as_default(): 



  ############
  # Input data
  ############
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])                 # For training
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])              # For training
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # For validation
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)                 # For validation



  ###################################################
  # Ops and variables pinned to the CPU (missing GPU)
  ###################################################
  with tf.device('/cpu:0'):

    ### Look up embeddings for inputs from whole embeddings
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="word_embedding")
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)  # This is the batch input

    ### Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    tf.summary.histogram("weights", nce_weights)  # For TensorBoard visualization
    tf.summary.histogram("bias", nce_biases)      # For TensorBoard visualization

  ### Compute the average NCE loss for the batch.
  """
  tf.nce_loss automatically draws a new sample of the negative labels each
  time we evaluate the loss.
  """
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))
  tf.summary.scalar("loss", loss)                 # For TensorBoard visualization
  
  ### Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  ### Compute the cosine similarity between minibatch examples (validation set) and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  ### Add variable initializer.
  init = tf.global_variables_initializer()

  ### Writer & Saver
  saver = tf.train.Saver()                             # For model saving
  writer = tf.summary.FileWriter(LOGDIR, graph=graph)  # For TensorBoard (summaries, graphs, images, audios, embedding)

  ### Tensorboard's summaries & embedding
  summ = tf.summary.merge_all()                                                  # Adding up all the summaries
  """
  Embedding system workflow:                                           # This is why we need both
  INPUT   -> Vocabularies, (one-hot-encoded)                           # the writer (since we'll be using TensorBoard)
  WEIGHTS -> 2D tensor (vocabulary_size, embedding_size),              # &
             which can be loaded from most recent .ckpt                # the saver (since we'll have to load .ckpt)
  OUTPUT  -> To be drawn 2D tensor (INPUT * WEIGHTS)                   # at the same time.
  """
  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()            # Embedding's configurator
  embedding_config = config.embeddings.add()                                     # Configuring...
  embedding_config.tensor_name = embeddings.name  # see name="word_embedding"    # Configuring...
  embedding_config.metadata_path = LABEL                                         # Configuring...
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)  # Visualizing with the configurations





## Session ############################################################################################################
with tf.Session(graph=graph) as session:



  ################################
  # Do operations for the training
  ################################

  ### global_variables_initializer
  init.run()

  ### Go
  for step in xrange(num_steps):
    # Get batch
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    # Step
    """
    We perform one update step by evaluating the optimizer op
    (including it in the list of returned values for session.run())
    """
    if step % 100 == 0:
        _, summ_out = session.run([optimizer, summ], feed_dict=feed_dict)
        writer.add_summary(summ_out, step)
    if step % 10000 == 0:
        saver.save(session, os.path.join(LOGDIR, "model.ckpt"), step)
    session.run([optimizer], feed_dict=feed_dict)
    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 20000 == 0 and step != 0:                     # Printing nearest words of the validation word
      sim = similarity.eval()                               # Printing nearest words of the validation word
      for i in xrange(valid_size):                          # Printing nearest words of the validation word
        valid_word = reverse_dictionary[valid_examples[i]]  # Printing nearest words of the validation word
        top_k = 8  # number of nearest neighbors            # Printing nearest words of the validation word
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]       # Printing nearest words of the validation word
        log_str = 'Nearest to %s:' % valid_word             # Printing nearest words of the validation word
        for k in xrange(top_k):                             # Printing nearest words of the validation word
          close_word = reverse_dictionary[nearest[k]]       # Printing nearest words of the validation word
          log_str = '%s %s,' % (log_str, close_word)        # Printing nearest words of the validation word
        print(log_str)                                      # Printing nearest words of the validation word
