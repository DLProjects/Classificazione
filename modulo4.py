import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import time
import csv

# ==============================================================
# =================== IMPORTA I DATI ===========================
# ==============================================================

print("Fase iniziale: importazione dei dati...")
data_dir = "./TrafficSigns/"

def rgb2gray(images):
    return np.dot(images, [0.2989, 0.5870, 0.1140])

def preprocess(x):
	x = rgb2gray(x)
	x = x.astype(np.float32) / 255
	x = x.reshape(x.shape + (1,))
	return x

def dense_to_one_hot(labels_dense, num_classi=10):
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classi
	labels_one_hot = np.zeros((num_labels, num_classi))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

with open(data_dir + "train.p", mode='rb') as f:
	train = pickle.load(f)

with open(data_dir + "valid.p", mode='rb') as f:
	valid = pickle.load(f)
	
with open(data_dir + "test.p", mode='rb') as f:
	test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_val, y_val = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

x_train = preprocess(x_train)
x_val = preprocess(x_val)
x_test = preprocess(x_test)

num_classi = len(np.unique(y_test))

y_train = dense_to_one_hot(y_train, num_classi)
y_test = dense_to_one_hot(y_test, num_classi)
y_val = dense_to_one_hot(y_val, num_classi)

labels = []
labels_file = data_dir + "signnames.csv"

with open(labels_file) as _f:
	rows = csv.reader(_f, delimiter=',')
	next(rows, None)
	for i, row in enumerate(rows):
		labels.append((row[0], row[1]))

print("Categorie:")
for i in range(len(labels)):
	time.sleep(0.5)
	print(labels[i][0], "-", labels[i][1])

# ==============================================================
# =================== SETTA I PARAMETRI ========================
# ==============================================================

num_immagini, dim_immagine, _, num_canali = x_train.shape

learning_rate = 0.001
num_epoche = 5
dim_batch = 100

dim_filtri_conv1 = 5
num_filtri_conv1 = 16

dim_filtri_conv2 = 5
num_filtri_conv2 = 32

dim_fc = 128

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

def crea_optimizer(costo):
	res = tf.train.AdamOptimizer(learning_rate).minimize(costo)
	return res
	
def calcola_accuracy(out, y):
	cor = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
	res = tf.reduce_mean(tf.cast(cor, tf.float32), name="accuracy")
	return res

def risultati(x_data, y_data):
    tot_acc = 0
    tot_dati = len(x_data)
    sess = tf.get_default_session()
    
    for j in range(0, tot_dati, dim_batch):
        x_batch = x_data[j:j+dim_batch,:,:,:]
        y_batch = y_data[j:j+dim_batch,:]
        fd = {x: x_batch, y: y_batch}
        
        batch_acc = sess.run(acc, feed_dict = fd)
        tot_acc += (batch_acc * len(x_batch)) / tot_dati
    
    return tot_acc

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
		stampa_img(x_test[0], labels[n][1]).show()
		c = True
	elif(risposta.lower() == "no"):
		c = True
	else:
		print("Inserimento errato. Scegliere fra 'si' e 'no'")

# ==============================================================
# ==================== CREA RETE NEURALE =======================
# ==============================================================

print("Creazione della rete neurale...")

# gestisci input
x = tf.placeholder(tf.float32, [None, dim_immagine, dim_immagine, num_canali])
y = tf.placeholder(tf.float32, [None, num_classi])


# primo strato di convoluzione
dim_kernel1 = [dim_filtri_conv1, dim_filtri_conv1, num_canali, num_filtri_conv1]
dim_b1 = [num_filtri_conv1]

W1 = crea_kernel(dim_kernel1, "W1")
b1 = crea_bias(dim_b1, "b1")

with tf.name_scope("conv1") as scope:
	layer1 = crea_convoluzione(x, W1, b1)

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
    optimizer = crea_optimizer(costo)

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
    
    summary_writer = tf.summary.FileWriter("modulo4", sess.graph)
    
    print("Inizio del processo di training...")
    
    for i in range(num_epoche):
        x_data, y_data = shuffle(x_train, y_train)
        
        train_acc = 0
        tot_dati = len(x_data)
        
        for j in range(0, num_immagini, dim_batch):
            x_batch = x_data[j:j+dim_batch,:,:,:]
            y_batch = y_data[j:j+dim_batch,:]
            fd = {x: x_batch, y: y_batch}
            
            sess.run(optimizer, feed_dict = fd)
            train_acc += (sess.run(acc, feed_dict = fd) * len(x_batch)) / tot_dati
            summary_writer.add_summary(sess.run(summaries, feed_dict = fd), i*num_immagini+j)
        
        train_acc = train_acc * 100
        val_acc = risultati(x_val, y_val) * 100
        
        print("Iterazione:", i, "- Trainining Accuracy: {:.3f} % - Val Accuracy: {:.3f} %".format(train_acc, val_acc))
    
    print("Termine del processo di training, eseguite", num_epoche, "epoche!")
    print("Valutazione delle performance sul test-set")
    time.sleep(2)
    
    test_acc = risultati(x_test, y_test) * 100
    print("Test Accuracy: {:.3f} %".format(test_acc))

