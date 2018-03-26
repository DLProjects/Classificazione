import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
from math import sqrt
import numpy as np
import time

# ==============================================================
# =================== IMPORTA I DATI* ==========================
# ==============================================================

print("Fase iniziale: importazione dei dati dal dataset QuickDraw...")
data_dir = "./QuickDraw/"

# *Parte di questa prima fase, per l'importazione dei dati, è stata rielaborata a partire da
# https://github.com/ankonzoid/Google-QuickDraw/blob/master/QuickDraw_noisy_classifier.py

l = {0: 'airplane', 1: 'banana', 2: 'hourglass', 3: 'icecream', 4: 'mountain',
	5: 'mug', 6: 'mushroom', 7: 'pineapple', 8: 'pizza', 9: 'rabbit'}
labels = list(l.values())

print("Categorie:")
for i in range(10):
	time.sleep(0.5)
	print(list(l.keys())[i], "-", labels[i])

def dense_to_one_hot(labels_dense, num_classi=10):
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classi
	labels_one_hot = np.zeros((num_labels, num_classi))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

category_filenames = []
n_remaining_category = []

for catname in labels:
	filename = os.path.join(data_dir, "full_numpy_bitmap_" + catname + ".npy")
	category_filenames.append(filename)

for i_category, category in enumerate(labels):
	data = np.load(category_filenames[i_category])
	n_total = len(data)
	n_remaining_category.append(n_total)

dim_immagine = int(sqrt(data.shape[1]))

n_take_train = min([40000, min(n_remaining_category)])
n_take_test = min([8000, min(n_remaining_category)])
	
x_train = []
y_train = []
x_test = []
y_test = []

for i_category, category in enumerate(labels):
	data = np.load(category_filenames[i_category])		
	n_data = len(data)
		
	for j, data_j in enumerate(data):
		img = np.array(data_j).reshape((dim_immagine, dim_immagine))
		if j < n_take_train:
			x_train.append(img)
			y_train.append(i_category)
		elif j - n_take_train < n_take_test:
			x_test.append(img)
			y_test.append(i_category)
		else:
			break

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = dense_to_one_hot(y_train)
y_test = dense_to_one_hot(y_test)

# ==============================================================
# =================== SETTA I PARAMETRI ========================
# ==============================================================

num_classi = len(labels)
learning_rate = 0.001
num_epoche = 100

# ==============================================================
# ===================== FUNZIONI UTILI =========================
# ==============================================================

def crea_pesi(dim, nome):
	res = tf.Variable(tf.zeros(dim), name = nome)
	return res

def crea_fc(inp, weight, bias):
	res = tf.matmul(inp, weight) + bias
	return res

def crea_costo(out, y):
	res = -tf.reduce_sum(y*tf.log(out + 1e-8), name="costo")
	return res

def crea_optimizer(opt, costo):
	if opt == "gd":
		res = tf.train.GradientDescentOptimizer(learning_rate).minimize(costo)
	elif opt == "adam":
		res = tf.train.AdamOptimizer(learning_rate).minimize(costo)
	elif opt == "adagrad":
		res = tf.train.AdagradOptimizer(learning_rate).minimize(costo)
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
	risposta = input("Stampare un'immagine d'esempio dal dataset? ")
	if(risposta.lower() == "si"):
		n = np.argmax(y_test[0], 0)
		stampa_img(x_test[0], l[n]).show()
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
		c = True
	elif risposta.lower() == "adam":
		opt = "adam"
		c = True
	elif risposta.lower() == "adagrad":
		opt = "adagrad"
		c = True
	else:
		print("Inserimento errato. Scegliere fra 'Gradient Descent', 'AdaGrad' e 'Adam'")

x = tf.placeholder(tf.float32, [None, dim_immagine, dim_immagine])
inp = tf.reshape(x, [-1, dim_immagine * dim_immagine])
y = tf.placeholder(tf.float32, [None, num_classi])

dim_W1 = [dim_immagine * dim_immagine, num_classi]
dim_b1 = [num_classi]

W1 = crea_pesi(dim_W1, "W1")
b1 = crea_pesi(dim_b1, "b1")

with tf.name_scope("output_layer") as scope:
	fc_out = crea_fc(inp, W1, b1)
	out = tf.nn.softmax(fc_out)

with tf.name_scope("funzione_costo") as scope:
    costo = crea_costo(out, y)

with tf.name_scope("train") as scope:
    optimizer = crea_optimizer(opt, costo)

with tf.name_scope("accuracy") as scope:
	acc = calcola_accuracy(out, y)

# ==============================================================
# ====================== AVVIO SESSIONE ========================
# ==============================================================

for elem in [W1, b1]:
    tf.summary.histogram(elem.op.name, elem)
    
for elem in [costo, acc]:
    tf.summary.scalar(elem.op.name, elem)

init = tf.global_variables_initializer()
summaries = tf.summary.merge_all()

print("Avvio sessione di tensorflow...")
time.sleep(2)

# inizia la sessione
with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter("modulo2", sess.graph)
    
    # allenamento
    print("Inizio del processo di training...")
	
    for i in range(num_epoche):
        fd = {x: x_train, y: y_train}
        acc_esec, _ = sess.run([acc, optimizer], feed_dict = fd)

        if i % 5 == 0 or i == num_epoche-1:
            print("Iterazione:", i, "- Training Accuracy: {:.3f} %".format(acc_esec * 100))
            summary_writer.add_summary(sess.run(summaries, feed_dict = fd), i)
    
    print("Termine del processo di training, eseguite", num_epoche, "epoche!")
    print("Valutazione delle performance sul test-set")
    time.sleep(2)

	# stampa i risultati sul test-set
    print("Test Accuracy: {:.3f} %".format(acc.eval({x: x_test, y: y_test}) * 100))

