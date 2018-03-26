import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import pickle
import time

# ==============================================================
# =================== IMPORTA I DATI* ==========================
# ==============================================================

print("Fase iniziale: importazione dei dati...")

# *Parte di questa prima fase, per l'importazione dei dati, è stata rielaborata a partire da
# https://github.com/thomalm/svhn-multi-digit/blob/master/01-svhn-single-preprocessing.ipynb (per il dataset SVHN)
# https://github.com/exelban/tensorflow-cifar-10/blob/master/include/data.py (per il dataset CIFAR-10)

def load_data(path):
    data = loadmat(path)
    return data['X'], data['y']

def balanced_subsample(y, s):
    sample = []
    for label in np.unique(y):
        images = np.where(y==label)[0]
        random_sample = np.random.choice(images, size=s, replace=False)
        sample += random_sample.tolist()
    return sample

def rgb2gray(images):
    return np.dot(images, [0.2989, 0.5870, 0.1140])

def dense_to_one_hot(labels_dense, num_classi=10):
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classi
	labels_one_hot = np.zeros((num_labels, num_classi))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

def estrai_cifar(data_dir, name, cifar=10):
    x = None
    y = None
    l = None

    f = open(data_dir + 'batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open(data_dir + 'data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open(data_dir + 'test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])

    return x, dense_to_one_hot(y), l

def estrai_svhn(data_dir):
	X_train, y_train = load_data(data_dir + 'train_32x32.mat')
	X_test, y_test = load_data(data_dir + 'test_32x32.mat')
	
	X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
	X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
	
	y_train[y_train == 10] = 0
	y_test[y_test == 10] = 0
	
	train_samples = balanced_subsample(y_train, 400)
	
	X_val, y_val = np.copy(X_train[train_samples]), np.copy(y_train[train_samples])
	
	X_train = np.delete(X_train, train_samples, axis=0)
	y_train = np.delete(y_train, train_samples, axis=0)
	X_test, y_test = X_test, y_test
	
	x_train = rgb2gray(X_train).astype(np.float32)
	x_test = rgb2gray(X_test).astype(np.float32)
	x_val = rgb2gray(X_val).astype(np.float32)
	
	train_mean = np.mean(x_train, axis=0)
	train_std = np.std(x_train, axis=0)
	
	x_train = (x_train - train_mean) / train_std
	x_test = (x_test - train_mean)  / train_std
	x_val = (train_mean - x_val) / train_std
	
	y_train = dense_to_one_hot(y_train)
	y_test = dense_to_one_hot(y_test)
	y_val = dense_to_one_hot(y_val)
	
	x_train = x_train.reshape(x_train.shape + (1,))
	x_test = x_test.reshape(x_test.shape + (1,))
	x_val = x_val.reshape(x_val.shape + (1,))
	
	return x_train, x_test, x_val, y_train, y_test, y_val

c = False

while(c == False):
	dataset = input("Inserire il nome del dataset: ")
	
	if(dataset.lower() == "svhn"):
		data_dir = "./SVHN/"
		x_train, x_test, x_val, y_train, y_test, y_val = estrai_svhn(data_dir)
		c = True
		
	elif(dataset.lower() == "cifar"):
		data_dir = "./CIFAR-10/"
		x_train, y_train, l_train = estrai_cifar(data_dir, "train")
		x_test, y_test, l_test = estrai_cifar(data_dir, "test")
		c = True
		
		train_samples = balanced_subsample(y_train, 1000)
		x_val, y_val = np.copy(x_train[train_samples]), np.copy(y_train[train_samples])
		
		x_train = np.delete(x_train, train_samples, axis=0)
		y_train = np.delete(y_train, train_samples, axis=0)
			
		print("Categorie:")
		for i in range(10):
			time.sleep(0.5)
			print(i, "-", l_test[i])
	else:
		print("Inserimento errato. Scegliere fra SVHN e Cifar")

# ==============================================================
# =================== SETTA I PARAMETRI ========================
# ==============================================================

num_immagini, dim_immagine, _, num_canali = x_train.shape
num_classi = y_train.shape[1]

learning_rate = 0.001
num_epoche = 5
dim_batch = 100

dim_filtri_conv1 = 5
num_filtri_conv1 = 32

dim_filtri_conv2 = 5
num_filtri_conv2 = 64

dim_filtri_conv3 = 3
num_filtri_conv3 = 128

dim_fc4 = 128
dim_fc5 = 64

l = 0.01
d = 0.75

# ==============================================================
# ===================== FUNZIONI UTILI =========================
# ==============================================================

def crea_kernel(dim, nome):
	res = tf.Variable(tf.truncated_normal(dim, stddev=0.05), name = nome)
	return res

def crea_bias(dim, nome):
	res = tf.Variable(tf.zeros(dim), name = nome)
	return res

def crea_convoluzione(inp, kernel, bias, usa_pooling=True):
	conv = tf.nn.conv2d(inp, kernel, strides=[1, 1, 1, 1], padding="SAME") + bias
	if usa_pooling == False:
		relu = tf.nn.relu(conv)
	else:
		pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
		relu = tf.nn.relu(pool)
	return relu

def crea_flatten_layer(inp):
	dim = inp.get_shape()
	num_features = dim[1:4].num_elements()
	res = tf.reshape(inp, [-1, num_features])
	return res, num_features

def crea_fc(inp, weight, bias, usa_relu=True):
	matmul = tf.matmul(inp, weight) + bias
	if usa_relu:
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

def risultati(x_data, y_data, usa_l2):
    tot_acc = 0
    tot_dati = len(x_data)
    sess = tf.get_default_session()
    
    for j in range(0, tot_dati, dim_batch):
        x_batch = x_data[j:j+dim_batch]
        y_batch = y_data[j:j+dim_batch]
        
        if usa_l2: 	fd = {x: x_batch, y: y_batch}
        else: 		fd = {x: x_batch, y: y_batch, prob: 1.0}
        
        batch_acc = sess.run(acc, feed_dict = fd)
        tot_acc += (batch_acc * len(x_batch)) / tot_dati
    
    return tot_acc

def stampa_img(inp, txt):
	if(num_canali!=1):
		s = (np.reshape(inp, (dim_immagine, dim_immagine, num_canali)) * 255).astype(np.uint8)
		plt.title("La label è {}".format(l_test[txt]))
	else:
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
		stampa_img(x_test[0], n).show()
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
	risposta = input("Inserire il tipo di regolarizzazione: ")
	if(risposta.lower() == "l2"):
		usa_l2 = True
		c = True
	elif(risposta.lower() == "dropout"):
		usa_l2 = False
		c = True
	else:
		print("Inserimento errato. Scegliere fra 'l2' e 'dropout'")

# gestisci input
x = tf.placeholder(tf.float32, [None, dim_immagine, dim_immagine, num_canali])
y = tf.placeholder(tf.float32, [None, num_classi])

if usa_l2 == False:
	prob = tf.placeholder(tf.float32)

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

# terzo strato di convoluzione
dim_kernel3 = [dim_filtri_conv3, dim_filtri_conv3, num_filtri_conv2, num_filtri_conv3]
dim_b3 = [num_filtri_conv3]

W3 = crea_kernel(dim_kernel3, "W3")
b3 = crea_bias(dim_b3, "b3")

with tf.name_scope("conv3") as scope:
	layer3 = crea_convoluzione(layer2, W3, b3, usa_pooling=False)

if usa_l2 == False:
	dropout = tf.nn.dropout(layer3, prob)
	with tf.name_scope("flat") as scope:
		flat, num_features = crea_flatten_layer(dropout)
else:
	with tf.name_scope("flat") as scope:
		flat, num_features = crea_flatten_layer(layer3)

# primo strato totalmente connesso
dim_W4 = [num_features, dim_fc4]
dim_b4 = [dim_fc4]

W4 = crea_kernel(dim_W4, "W4")
b4 = crea_bias(dim_b4, "b4")

with tf.name_scope("fc4") as scope:
	layer4 = crea_fc(flat, W4, b4)
	
# secondo strato totalmente connesso
dim_W5 = [dim_fc4, dim_fc5]
dim_b5 = [dim_fc5]

W5 = crea_kernel(dim_W5, "W5")
b5 = crea_bias(dim_b5, "b5")

with tf.name_scope("fc5") as scope:
	layer5 = crea_fc(layer4, W5, b5)

# terzo strato totalmente connesso
dim_W6 = [dim_fc5, num_classi]
dim_b6 = [num_classi]

W6 = crea_kernel(dim_W6, "W6")
b6 = crea_bias(dim_b6, "b6")

with tf.name_scope("output") as scope:
	layer6 = crea_fc(layer5, W6, b6, usa_relu=False)
	out = tf.nn.softmax(layer6)


# crea la funzione di costo
with tf.name_scope("funzione_costo") as scope:
	costo = crea_costo(layer6, y)

if usa_l2:
	with tf.name_scope("regolarizzazione") as scope:
		reg = tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5)
		reg_costo = tf.reduce_mean(costo + l * reg)
	
	with tf.name_scope("train") as scope:
		optimizer = crea_optimizer(reg_costo)
else:
	with tf.name_scope("train") as scope:
		optimizer = crea_optimizer(costo)

# confronto fra il risultato ottenuto e quello atteso
with tf.name_scope("accuracy") as scope:
	acc = calcola_accuracy(out, y)

# ==============================================================
# ====================== AVVIO SESSIONE ========================
# ==============================================================
    
for elem in [costo, acc]:
    tf.summary.scalar(elem.op.name, elem)

init = tf.global_variables_initializer()
summaries = tf.summary.merge_all()

print("Avvio sessione di tensorflow...")
time.sleep(2)

with tf.Session() as sess:
    sess.run(init)
    
    summary_writer = tf.summary.FileWriter("modulo5", sess.graph)
    
    print("Inizio del processo di training...")
    
    for i in range(num_epoche):
        x_data, y_data = shuffle(x_train, y_train)
        
        train_acc = 0
        tot_dati = len(x_data)
        
        for j in range(0, num_immagini, dim_batch):
            x_batch = x_data[j:j+dim_batch]
            y_batch = y_data[j:j+dim_batch]
            
            if usa_l2: fd = {x: x_batch, y: y_batch}
            else: fd = {x: x_batch, y: y_batch, prob: d}
            
            sess.run(optimizer, feed_dict = fd)
            train_acc += (sess.run(acc, feed_dict = fd) * len(x_batch)) / tot_dati
            summary_writer.add_summary(sess.run(summaries, feed_dict = fd), i*num_immagini+j)
        
        train_acc = train_acc * 100
        val_acc = risultati(x_val, y_val, usa_l2) * 100
        
        print("Iterazione:", i, "- Trainining Accuracy: {:.3f} % - Val Accuracy: {:.3f} %".format(train_acc, val_acc))
    
    print("Termine del processo di training, eseguite", num_epoche, "epoche!")
    print("Valutazione delle performance sul test-set")
    time.sleep(2)
    
    test_acc = risultati(x_test, y_test, usa_l2) * 100
    print("Test Accuracy: {:.3f} %".format(test_acc))
