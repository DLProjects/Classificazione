import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
from math import sqrt
import numpy as np
import time

# ==============================================================
# =================== IMPORTA I DATI ===========================
# ==============================================================

from tensorflow.examples.tutorials.mnist import input_data
c = False
fashion_label = None
print("Fase iniziale: importazione dei dati...")

while(c == False):
	dataset = input("Inserire il nome del dataset: ")
	if(dataset.lower() == "mnist"):
		dati = input_data.read_data_sets("./MNIST", one_hot=True)
		c = True
	elif(dataset.lower() == "fashion-mnist"):
		dati = input_data.read_data_sets('./Fashion-MNIST', one_hot=True)
		fashion_label = {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
						5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
		c = True
		
		print("Categorie:")
		for i in range(10):
			time.sleep(0.5)
			print(list(fashion_label.keys())[i], "-", list(fashion_label.values())[i])
	else:
		print("Inserimento errato. Scegliere fra MNIST e Fashion-MNIST")

# ==============================================================
# =================== SETTA I PARAMETRI ========================
# ==============================================================

dim_immagine = int(sqrt(dati.train.images.shape[1]))
num_classi = dati.train.labels.shape[1]
num_canali = 1

learning_rate = 0.001
dim_batch = 100

dim_filtri_conv1 = 5
num_filtri_conv1 = 16

dim_filtri_conv2 = 5
num_filtri_conv2 = 32

dim_fc = 32

# ==============================================================
# ===================== FUNZIONI UTILI =========================
# ==============================================================

def crea_kernel(dim, nome):
	res = tf.Variable(tf.truncated_normal(dim, stddev=0.05), name = nome)
	return res

def crea_bias(dim, nome):
	res = tf.Variable(tf.zeros(dim), name = nome)
	return res

def crea_convoluzione(inp, kernel, bias):
	conv = tf.nn.conv2d(inp, kernel, strides=[1, 1, 1, 1], padding="SAME") + bias
	pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
	relu = tf.nn.relu(pool)
	return relu

def crea_flatten_layer(inp):
	dim = inp.get_shape()
	num_features = dim[1:4].num_elements()
	res = tf.reshape(inp, [-1, num_features])
	return res, num_features

def crea_fc(inp, weight, bias, use_relu=True):
	matmul = tf.matmul(inp, weight) + bias
	if use_relu:
		relu = tf.nn.relu(matmul)
		return relu
	return matmul

def crea_costo(out, y):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y)
	costo = tf.reduce_mean(cross_entropy, name="costo")
	return costo

def crea_optimizer(opt, costo):
	if opt == "gd":
		res = tf.train.GradientDescentOptimizer(learning_rate).minimize(costo)
	elif opt == "adam":
		res = tf.train.AdamOptimizer(learning_rate).minimize(costo)
	return res
	
def calcola_accuracy(out, y):
	cor = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
	res = tf.reduce_mean(tf.cast(cor, tf.float32), name="accuracy")
	return res

def stampa_img(inp, txt):
    s = (np.reshape(inp, (dim_immagine, dim_immagine)) * 255).astype(np.uint8)
    plt.title("La label è {}".format(txt))
    plt.imshow(s, cmap="gray")
    return plt

# ==============================================================
# ===================== STAMPA IMMAGINI ========================
# ==============================================================
    
c = False

while(c == False):
	risposta = input("Stampare un'immagine d'esempio dal dataset scelto? ")
	if(risposta.lower() == "si"):
		lab = np.argmax(dati.test.labels[0], 0)
		txt = lab if fashion_label is None else fashion_label[lab]
		stampa_img(dati.test.images[0], txt).show()
		c = True
	elif(risposta.lower() == "no"):
		c = True
	else:
		print("Inserimento errato. Scegliere fra 'si' e 'no'")

# ==============================================================
# ==================== CREA RETE NEURALE =======================
# ==============================================================

print("Creazione della rete neurale...")

c = False

while(c == False):
	risposta = input("Inserire il metodo di ottimizzazione: ")
	if risposta.lower() == "gradient descent":
		opt = "gd"
		num_epoche = 20
		c = True
	elif risposta.lower() == "adam":
		opt = "adam"
		num_epoche = 5
		c = True
	else:
		print("Inserimento errato. Scegliere fra 'Gradient Descent' e 'Adam'")

# gestisci input
x = tf.placeholder(tf.float32, [None, dim_immagine * dim_immagine])
inp = tf.reshape(x, [-1, dim_immagine, dim_immagine, num_canali])
y = tf.placeholder(tf.float32, [None, num_classi])


# primo strato di convoluzione
dim_kernel1 = [dim_filtri_conv1, dim_filtri_conv1, num_canali, num_filtri_conv1]
dim_b1 = [num_filtri_conv1]

W1 = crea_kernel(dim_kernel1, "W1")
b1 = crea_bias(dim_b1, "b1")

with tf.name_scope("conv1") as scope:
	layer1 = crea_convoluzione(inp, W1, b1)

# secondo strato di convoluzione
dim_kernel2 = [dim_filtri_conv2, dim_filtri_conv2, num_filtri_conv1, num_filtri_conv2]
dim_b2 = [num_filtri_conv2]

W2 = crea_kernel(dim_kernel2, "W2")
b2 = crea_bias(dim_b2, "b2")

with tf.name_scope("conv2") as scope:
	layer2 = crea_convoluzione(layer1, W2, b2)

# strato intermedio
with tf.name_scope("flat") as scope:
	flat, num_features = crea_flatten_layer(layer2)

# primo strato totalmente connesso
dim_W3 = [num_features, dim_fc]
dim_b3 = [dim_fc]

W3 = crea_kernel(dim_W3, "W3")
b3 = crea_bias(dim_b3, "b3")

with tf.name_scope("fc3") as scope:
	layer3 = crea_fc(flat, W3, b3)

# secondo strato totalmente connesso
dim_W4 = [dim_fc, num_classi]
dim_b4 = [num_classi]

W4 = crea_kernel(dim_W4, "W4")
b4 = crea_bias(dim_b4, "b4")

with tf.name_scope("output") as scope:
	layer4 = crea_fc(layer3, W4, b4, use_relu=False)
	out = tf.nn.softmax(layer4)


# crea la funzione di costo
with tf.name_scope("funzione_costo") as scope:
	costo = crea_costo(layer4, y)

# minimizza la funzione di costo
with tf.name_scope("train") as scope:
    optimizer = crea_optimizer(opt, costo)

# confronto fra il risultato ottenuto e quello atteso
with tf.name_scope("accuracy") as scope:
	acc = calcola_accuracy(out, y)

# ==============================================================
# ====================== AVVIO SESSIONE ========================
# ==============================================================

for elem in [W1, b1, W2, b2, W3, b3, W4, b4]:
    tf.summary.histogram(elem.op.name, elem)
    
for elem in [costo, acc]:
    tf.summary.scalar(elem.op.name, elem)

init = tf.global_variables_initializer()
summaries = tf.summary.merge_all()

print("Avvio sessione di tensorflow...")
time.sleep(2)

with tf.Session() as sess:
	sess.run(init)
	
	summary_writer = tf.summary.FileWriter("modulo3", sess.graph)
	
	print("Inizio del processo di training...")
	
	for i in range(num_epoche):
		acc_esec = 0.
		num_batch = int(dati.train.num_examples/dim_batch)		
		
		for j in range(num_batch):
			x_batch, y_batch = dati.train.next_batch(dim_batch)
			fd = {x: x_batch, y: y_batch}
			
			sess.run(optimizer, feed_dict = fd)
			acc_esec += sess.run(acc, feed_dict = fd)/num_batch
			summary_writer.add_summary(sess.run(summaries, feed_dict=fd), i*num_batch+j)
		
		print("Iterazione:", i, "- Training Accuracy: {:.3f} %".format(acc_esec * 100))
			
	print("Termine del processo di training, eseguite", num_epoche, "epoche!")
	print("Valutazione delle performance sul test-set")
	time.sleep(2)
	
	print("Test Accuracy: {:.3f} %".format(acc.eval({x: dati.test.images, y: dati.test.labels}) * 100))
