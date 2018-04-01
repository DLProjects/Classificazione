import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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

learning_rate = 0.001
num_epoche = 35
dim_batch = 100

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

def crea_optimizer(costo):
	res = tf.train.GradientDescentOptimizer(learning_rate).minimize(costo)
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

x = tf.placeholder(tf.float32, [None, dim_immagine * dim_immagine])
y = tf.placeholder(tf.float32, [None, num_classi])

dim_W1 = [dim_immagine * dim_immagine, num_classi]
dim_b1 = [num_classi]

W1 = crea_pesi(dim_W1, "W1")
b1 = crea_pesi(dim_b1, "b1")

with tf.name_scope("output_layer") as scope:
	fc_out = crea_fc(x, W1, b1)
	out = tf.nn.softmax(fc_out)

with tf.name_scope("funzione_costo") as scope:
    costo = crea_costo(out, y)

with tf.name_scope("train") as scope:
    optimizer = crea_optimizer(costo)

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

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter("modulo1", sess.graph)
    
    # allenamento
    print("Inizio del processo di training...")
	
    for i in range(num_epoche):
        acc_esec = 0.
        num_batch = int(dati.train.num_examples/dim_batch)
	
        for j in range(num_batch):
            x_batch, y_batch = dati.train.next_batch(dim_batch)
            fd = {x: x_batch, y: y_batch}
            
            sess.run(optimizer, feed_dict = fd)
            acc_esec += sess.run(acc, feed_dict = fd)/num_batch

        if i % 5 == 0 or i == num_epoche-1:
            print("Iterazione:", i, "- Training Accuracy: {:.3f} %".format(acc_esec * 100))
            fd = {x: dati.train.images, y: dati.train.labels}
            summary_writer.add_summary(sess.run(summaries, feed_dict = fd), i)
            
    print("Termine del processo di training, eseguite", num_epoche, "epoche!")
    print("Valutazione delle performance sul test-set")
    time.sleep(2)

	# stampa i risultati sul test-set
    print("Test Accuracy: {:.3f} %".format(acc.eval({x: dati.test.images, y: dati.test.labels}) * 100))

