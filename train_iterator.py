# ------------------------------------
# -- Importo le librerie necessarie
# ------------------------------------

# Librerie modello
from builtins import str

import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

print (keras.__version__)

# Altre librerie
import matplotlib
matplotlib.use("TKAgg") # Utilizzo il backend per la visualizzazione grafica delle immagini

import keras
from keras.preprocessing.image import image
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import csv

import logging
import datetime

import gc


# ----------------------------------------
# -- Costanti modello e altri parametri
# ----------------------------------------

# Costanti modello
IMAGE_WIDTH = 178
IMAGE_HEIGHT = 218
EPOCHS = 30 #epoche training
INIT_LE = 1e-3 #valore default ottimizzatore Adam
BS = 32 #batch size
IMAGE_DIMS = (IMAGE_WIDTH, IMAGE_HEIGHT, 3) #dimensioni immagine

# Costanti cartelle e file
CSV_FOLDER = "csv" #sottocartella contenente i file csv
DATASET_PATH = "./img_align_celeba"
MODEL_PATH = "./model/"
PLOT_PATH = "./plot/"
LOG_PATH = "./log/"
#LABELS_CSV_PATH = "./" + CSV_FOLDER + "/list_attr_celeba_updated.csv"
#LABELS_CSV_PATH = "./" + CSV_FOLDER + "/list_attr_celeba_updated_minor.csv"
LABELS_CSV_PATH = "./" + CSV_FOLDER + "/list_attr_celeba_zeros.csv"
#MODEL_OUT_PATH = MODEL_PATH + "trainedCelebA_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".h5"
MODEL_OUT_PATH = MODEL_PATH + "trainedCelebA.h5"
MODEL_WEIGHT_PATH = MODEL_PATH + "trainedCelebA_weight.h5"
PLOT_SAVE_PATH = PLOT_PATH + "loss_acc_plot_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".png"
PLOT_MEDIA_SAVE_PATH = PLOT_PATH + "media_plot_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".png"
LOG_FILE_NAME = LOG_PATH + "logTrainIteratorCelebA_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".log"
HISTORY_CSV = CSV_FOLDER + "/history_full.csv"
HISTORY_MEDIA_CSV = CSV_FOLDER + "/history_media.csv"
CHECKPOINT_IMAGE_PATH = "./lastStart.txt"

# Configurazione logger
logging.basicConfig(filename=LOG_FILE_NAME, level=logging.INFO)
logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] PREPARAZIONE FASE TRAINING")
logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Parametri: IMAGE_WIDTH=" + str(IMAGE_WIDTH) + ", IMAGE_HEIGHT=" + str(IMAGE_HEIGHT) + ", EPOCHS=" + str(EPOCHS) + ", BATCHSIZE=" + str(BS) + ".")

# --------------------------------
# -- Definizione funzioni utili
# --------------------------------

# Metodo di checkpoint e callback in modo da salvare il modello durante il training
# ed interrompere il training in caso di overfitting dopo 5 epoche

callback_list = [
    keras.callbacks.EarlyStopping(monitor='acc', patience=5), #controlla l'accuracy e interrompe se non si migliora dopo 5 epoch
    keras.callbacks.ModelCheckpoint(filepath=MODEL_OUT_PATH, monitor='val_loss', save_best_only=True)
]


# Funzione per la creazione del model

def build(imageWidth, imageHeight, imageDepth, classesNumber, finalAct="sigmoid"):
    """
    Funzione che costruisce un modello per il training
    :param imageWidth: larghezza immagine
    :param imageHeight: altezza immagine
    :param imageDepth: numero canali immagine (3 default per RGB)
    :param classesNumber: numero parametri da avere in output dalla rete (pari al numero di caratteristiche facciali che vogliamo estrarre)
    :param finalAct: funzione di attivazione dell'ultimo layer del modello. Per il multilabel utilizzo sigmoid come attivazione finale
    :return: il modello per il training
    """

    # inizializzo il modello come sequenziale
    model = Sequential()
    inputShape = (imageHeight, imageWidth, imageDepth)
    chanDim = -1

    # Primo blocco Conv2D, Relu, Normalization, MaxPool
    # Utilizzo 32 filtri 3*3
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=inputShape))
    # con attivazione Rectified Linear Unit
    model.add(Activation("relu"))
    # applico una batch normalization
    model.add(BatchNormalization(axis=chanDim))
    # un MaxPooling 3*3
    model.add(MaxPooling2D(pool_size=(3, 3)))
    # ed un 25% di dropout per ridurre overfitting
    model.add(Dropout(0.25))

    # Secondo blocco
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Terzo blocco
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Passo ai Fully Connected Layers
    # Trasformo il modello in un vettore
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("sigmoid"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Infine utilizzo l'attivazione per la rete
    model.add(Dense(classesNumber))
    model.add(Activation(finalAct))

    return model


# Funzione per il caricamento delle immagini tra due indici utilizzando il file csv contenente i nomi dei file

def loadImages(start, stop, csvFile):
    """
    Carica le immagini prendendo il nome dal file csv e restituisce il dataset contenente le immagini
    :param start: indice di partenza del csv
    :param stop: indice di fine del csv
    :param csvFile: file csv contenente le labels
    :return: dataset contenente stop-start immagini
    """
    dataset = []

    for i in tqdm(range(start, stop)):
        # print(DATASET_PATH + "/" + csvLabels.loc[i]["image_id"])
        # print(csvFile.loc[i]["image_id"])
        img = image.load_img(DATASET_PATH + "/" + csvFile.loc[i]["image_id"], target_size=IMAGE_DIMS)
        img = image.img_to_array(img)
        img = img / 255
        dataset.append(img)

    return dataset


# Funzione per il caricamento delle label tra due indici all'interno di una lista

def loadLabels(start, stop, csvFile):
    """
    Restituisce le label contenute del file csv partendo da start fino a stop
    :param start: indice di partenza del csv
    :param stop: indice di fine del csv
    :param csvFile: file csv contenente le labels
    :return: dataframe contenente celle del csv nelle row [start:stop]
    """
    return csvFile[start:stop]


# Funzione per la creazione del model e dell'ottimizzatore utilizzando diversi parametri
# Ottimizzatore utilizzato: Adam

def buildFirstModel():
    """
    Questa funzione viene chiamata durante la prima iterazione del training.
    Definisce il model per il train con tutti i parametri
    """
    model = build(IMAGE_HEIGHT, IMAGE_WIDTH, 3, y.shape[1], finalAct="sigmoid")
    opt = Adam(lr=INIT_LE, decay=INIT_LE / EPOCHS)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])


# --------------------------------------------------------
# -- Caricamento file CSV contenente labels e nomi file
# --------------------------------------------------------

print("[INFO] Carico file CSV...")

# Leggo il file csv creando un dataframe Pandas
csvLabels = pd.read_csv(LABELS_CSV_PATH)
# Seleziono le caratteristiche da droppare tra le labels del dataframe
featureToDrop = ['5_o_Clock_Shadow', 'Attractive', 'Chubby',
                 'High_Cheekbones', 'Oval_Face', 'Pale_Skin',
                 'Rosy_Cheeks', 'Wearing_Lipstick']
# Droppo le varie feature nel dataframe
for feature in featureToDrop:
    csvLabels = csvLabels.drop([feature], axis=1)
# Stampo un sample del dataframe
print("csvLabels full head():")
print(csvLabels.head())
logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] File CSV " + LABELS_CSV_PATH + " caricato.")


# --------------------------------------------------
# -- Inizializzo alcune variabili per il training
# --------------------------------------------------

# totalImages contiene il numero di immagini del dataset prendendo il numero di righe contenute nel dataframe csv
totalImages = csvLabels.shape[0]
print("Immagini per training: %d" % totalImages)
# Imposto il numero di immagini da utilizzare per il training per ogni iterazione
numImagesPerIteration = 500
# La prima iterazione partirà dalla prima immagine, questa variabile verrà sovrascritta con esecuzioni future9
start = 0

# Se il file di checkpoint per l'ultima immagine non esiste, è la prima iterazione
if os.path.exists(CHECKPOINT_IMAGE_PATH) == False:
    logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] File checkpoint non esistente. Lo creo.")
    # Imposto flag per la prima iterazione per la creazione del model
    firstIteration = True
    # Creo il file contenente l'immagine di partenza. In questo caso 0
    with open(CHECKPOINT_IMAGE_PATH, "w") as checkpointNumber:
        checkpointNumber.write(str(start))
else:
    # Imposto il flag per continuare il training
    firstIteration = False
    logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] File checkpoint esistente. Leggo il contenuto.")
    # Apro il file di checkpoint e leggo l'ultimo blocco di immagini utilizzato
    with open(CHECKPOINT_IMAGE_PATH, "r") as checkpointNumber:
        start = int(checkpointNumber.readline())

# Per il test del model su un numero limitato di immagini rimuovere il commento e cambiare il numero
# totalImages = 1500

logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Totale immagini train: " + str(totalImages) + ", #immagini per iterazione: " + str(numImagesPerIteration))
logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] TRAINING")

# -----------------------------------------------
# -- Inizio iterazioni per training della rete
# -----------------------------------------------

print("[INFO] Inizio iterazione per training...")

# Ciclo per le varie iterazioni
for iteration in range(int(totalImages / numImagesPerIteration)):

    print("Iterating from images %d to %d" % (start, start + numImagesPerIteration))

    # Carico le immagini del dataset partendo da start per numImagesPerIteration immagini
    dataset = loadImages(start, start + numImagesPerIteration, csvLabels)

    logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Caricate immagini per train da %d a %d. Conversione in np.array." % (start, start+numImagesPerIteration))

    # Creo un Numpy array dal dataset leggibile dalla rete
    X = np.array(dataset)
    print("Iteration %d" % iteration)
    print("X.shape: ", X.shape)

    logging.info(
        "[INFO] [{0}] X.shape: {1}".format(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), str(X.shape)))

    # Carico le labels, conterrà numImagesPerIteration rows così da permettere il loading di quelle immagini per il train
    labels = loadLabels(start, start + numImagesPerIteration, csvLabels)

    # Droppo la column image_id dalle labels del csv
    labels = labels.drop(['image_id'], axis = 1)
#    print("labels (no file name) head()")
#    print(labels.head())
    # Creo un Numpy array delle labels leggibile dalla rete
    y = np.array(labels)
    print("y.shape: ", y.shape)

    logging.info("[INFO] [{0}] Caricate labels da {1} a {2}. y.shape: {3}".format(
        datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), str(start), str(start + numImagesPerIteration),
        str(y.shape)))

    # Suddivido le numImagesPerIteration immagini e labels per il training ed il test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    print("X_train: {0}".format(str(X_train.shape)))
    print("X_test: {0}".format(str(X_test.shape)))
    print("y_train: {0}".format(str(y_train.shape)))
    print("y_test: {0}".format(str(y_test.shape)))

    logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Dataset suddiviso:")
    logging.info(
        "[INFO] [{0}] X_train: {1}".format(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), str(X_train.shape)))
    logging.info(
        "[INFO] [{0}] X_test: {1}".format(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), str(X_test.shape)))
    logging.info(
        "[INFO] [{0}] y_train: {1}".format(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), str(y_train.shape)))
    logging.info(
        "[INFO] [{0}] y_test: {1}".format(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), str(y_test.shape)))

    # -----------------------------------------------------------
    # -- Sezione prima iterazione, creo il model da utilizzare
    # -----------------------------------------------------------

    if firstIteration is True:
        print("[INFO] Prima iterazione. Creo il model per il training.")
        logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Prima iterazione. Creo model e ottimizzatore.")

        # Costruisco il modello da utilizzare per il train
        model = build(IMAGE_HEIGHT, IMAGE_WIDTH, 3, y.shape[1], finalAct="sigmoid")
        # Creo un ottimizzatore per il model
        opt = Adam(lr=INIT_LE, decay=INIT_LE / EPOCHS)
        # Cro il modello
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])

        logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Parametri model: loss=binary_crossentropy, opt=Adam, metrics=[\"acc\"]")
        logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Parametri optimizer Adam: lr=" + str(INIT_LE) + ", decay=" + str(INIT_LE/EPOCHS) + ".")

        print("Training. Iteration: %d. Number of images for training: %d" % (iteration, numImagesPerIteration))
        logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Inizio training. Iterazione %d." % (iteration))

        # Avvio il training sulle X_train immagini, per EPOCHS epoche
        H = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), batch_size=BS, verbose=1, callbacks=callback_list)
        # Creo un dizionario iterabile in quanto H restituisce errori se utilizzato in questo modo
        iterableHistory = H.history


        # Creo file CSV che conterranno loss e acc delle varie iterazioni del train
        # Se il file non esiste, lo creo inserendo le labels nella prima row
        if os.path.exists(HISTORY_CSV) == False:
            with open(HISTORY_CSV, "w") as historyCsv:
                writer = csv.writer(historyCsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
                logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Creo il file " + HISTORY_CSV)

                keys = ["Iteration", "Epoch"]
                for key in iterableHistory:
                    keys.append(key)
                writer.writerow(keys)

        # Apro il file in append e inserisco i vari valori restituiti dal train
        with open(HISTORY_CSV, "a") as historyCsv:
            writer = csv.writer(historyCsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Scrivo i valori in " + HISTORY_CSV)

            for epoch in range(len(iterableHistory["loss"])):
                values = []
                values.append(str(iteration))
                values.append(epoch)

                for key in iterableHistory:
                    values.append(iterableHistory[key][epoch])
                writer.writerow(values)


        # Creo file CSV che conterranno loss e acc medio delle varie iterazioni del train

        if os.path.exists(HISTORY_MEDIA_CSV) == False:
            with open(HISTORY_MEDIA_CSV, "w") as historyMediaCsv:
                writer = csv.writer(historyMediaCsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
                logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Creo il file " + HISTORY_MEDIA_CSV)

                keys = ["Iteration"]
                for key in iterableHistory:
                    keys.append(key)
                writer.writerow(keys)

        # Creo list contenente media valori + numero iterazione attuale
        mediaValues = []
        mediaValues.append(str(iteration))
        # aggiungo media dei valori
        for name in iterableHistory:

            sum = 0
            for i in range(len(iterableHistory[name])):
                sum += iterableHistory[name][i]
            media = sum / len(iterableHistory[name])
            mediaValues.append(media)

        # Inserisco i valori di media all'interno del file csv
        with open(HISTORY_MEDIA_CSV, "a") as historyMediaCsv:
            writer = csv.writer(historyMediaCsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Scrivo i valori in " + HISTORY_MEDIA_CSV)

            writer.writerow(mediaValues)

        logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Train iterazione %d teminato. Valori attuali:" % (iteration))
        logging.info("[INFO] [{0}] H.history: {1}".format(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), str(H.history)))

        # Imposto valore a false così da entrare nel blocco successivo per le successive iterazioni
        firstIteration = False

    # ----------------------------------------------------------------------
    # -- Sezione seconda+ iterazione, carico model e continuo il training
    # ----------------------------------------------------------------------

    elif firstIteration is False:
        logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Iterazione %d. Carico il model precedentemente creato." % (iteration))
        print("[INFO] Carico il model precedente...")

        # Carico il model per continuare il training
        model = load_model(MODEL_OUT_PATH)
        # Continuo il training su altri dati
        H = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), batch_size=BS, verbose=1, callbacks=callback_list)

        logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Train iterazione %d teminato. Valori attuali:" % (iteration))
        logging.info("[INFO] [{0}] H.history: {1}".format(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), str(H.history)))

        # Creo un dizionario iterabile in quanto H restituisce errori se utilizzato in questo modo
        iterableHistory = H.history

        # Inserisco i nuovi valori ottenuti dal training all'interno del file completo
        with open(HISTORY_CSV, "a") as historyCsv:
            writer = csv.writer(historyCsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Scrivo i valori in " + HISTORY_CSV)

            for epoch in range(len(iterableHistory["loss"])):
                values = []
                values.append(str(iteration))
                values.append(epoch)

                for key in iterableHistory:
                    values.append(iterableHistory[key][epoch])
                writer.writerow(values)

        mediaValues = []
        mediaValues.append(str(iteration))
        # aggiungo media dei valori
        for name in iterableHistory:

            sum = 0
            for i in range(len(iterableHistory[name])):
                sum += iterableHistory[name][i]
            media = sum / len(iterableHistory[name])
            mediaValues.append(media)
#            historyMedia[name].append(media)

        # Inserisco i valori medi ottenuti dal training del file csv
        with open(HISTORY_MEDIA_CSV, "a") as historyMediaCsv:
            writer = csv.writer(historyMediaCsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            logging.info("[INFO] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Scrivo i valori in " + HISTORY_MEDIA_CSV)

            writer.writerow(mediaValues)

    else:
        print("[ERROR] Questo non sarebbe dovuto accadere...")
        logging.info("[ERROR] [" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "] Errore durante iterazione #%d. Termino." % (iteration))
        raise NameError("firstIteration ha un valore non gestito.")

    # Forzo il Garbage Collector di Python a liberare memoria così da rallentare la saturazione della memoria
    gc.collect()
    # Incremento valore dell'immagine di partenza per il train
    start += numImagesPerIteration
    # Salvo il valore di start in un file così da leggerlo in caso di esecuzioni future
    with open(CHECKPOINT_IMAGE_PATH, "w") as checkpointNumber:
        checkpointNumber.write(str(start))

print("[INFO] Training terminato. Carico il model per valutazione.")

# Plot e testo vengono generati in uno script Python separato