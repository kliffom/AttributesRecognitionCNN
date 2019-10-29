# ---------------------------------
# -- Importo librerie necessarie
# ---------------------------------

from builtins import str

import keras
print (keras.__version__)

# Altre librerie
import matplotlib
matplotlib.use("TKAgg") # Utilizzo il backend per la visualizzazione grafica delle immagini

from keras.preprocessing.image import image
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import csv
import datetime


# ----------------------------
# -- Costanti da utilizzare
# ----------------------------

IMAGE_WIDTH = 178
IMAGE_HEIGHT = 218
IMAGE_DIMS = (IMAGE_WIDTH, IMAGE_HEIGHT, 3) #dimensioni immagine

EPOCHS = 30 #epoche training
INIT_LE = 1e-3 #valore default ottimizzatore Adam
BS = 32 #batch size

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
MODEL_YAML_PATH = MODEL_PATH + "trainedCelebA.yaml"
MODEL_WEIGHT_PATH = MODEL_PATH + "trainedCelebA_weight.h5"
PLOT_SAVE_PATH = PLOT_PATH + "loss_acc_plot_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".png"
PLOT_MEDIA_SAVE_PATH = PLOT_PATH + "media_plot_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".png"
LOG_FILE_NAME = LOG_PATH + "logTrainIteratorCelebA_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".log"
HISTORY_CSV = CSV_FOLDER + "/history_full.csv"
HISTORY_MEDIA_CSV = CSV_FOLDER + "/history_media.csv"
FREQUENCY_CSV = "./" + CSV_FOLDER + "/attributes_frequency.csv"
CHECKPOINT_IMAGE_PATH = "./lastStartTest.txt"
NUM_ERROR_FILE = "./errors.txt"


# --------------------------------
# -- Definizione funzioni utili
# --------------------------------

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


def loadLabels(start, stop, csvFile):
    """
    Restituisce le label contenute del file csv partendo da start fino a stop
    :param start: indice di partenza del csv
    :param stop: indice di fine del csv
    :param csvFile: file csv contenente le labels
    :return: dataframe contenente celle del csv nelle row [start:stop]
    """
    return csvFile[start:stop]

# --------------------------------------------------------------------------
# -- Caricamento il file CSV contenente le labels e creazione file output
# --------------------------------------------------------------------------

print("[INFO] Carico file CSV...")

csvLabels = pd.read_csv(LABELS_CSV_PATH)
featureToDrop = ['5_o_Clock_Shadow', 'Attractive', 'Chubby',
                 'High_Cheekbones', 'Oval_Face', 'Pale_Skin',
                 'Rosy_Cheeks', 'Wearing_Lipstick']
for feature in featureToDrop:
    csvLabels = csvLabels.drop([feature], axis=1)
print("csvLabels full head():")
print(csvLabels.head())

# creo dataframe di supporto per estrarre i nomi degli attributi rinamenti nel file csv
csvLabelsNames = csvLabels.drop(['image_id'], axis=1)
# print("Numero colonne: " + str(len(csvLabelsNames.head(1))))
# inserisco il nome degli attributi in una lista in modo da inserirli nel csv con i risultati
labelsName = list(csvLabelsNames.columns)
print("Numero colonne: " + str(len(labelsName)))

totalImages = csvLabels.shape[0]
print("Immagini per training: %d" % totalImages)
numImagesPerIteration = 500
start = 0

# Se il file di checkpoint per l'ultima immagine non esiste, è la prima iterazione
if os.path.exists(CHECKPOINT_IMAGE_PATH) == False:

    #creo il file contenente l'immagine di partenza. In questo caso 0
    with open(CHECKPOINT_IMAGE_PATH, "w") as checkpointNumber:
        checkpointNumber.write(str(start))

else:
    with open(CHECKPOINT_IMAGE_PATH, "r") as checkpointNumber:
        start = int(checkpointNumber.readline())

# Numero immagini da utilizzare per l'esecuzione (utilizzata per testare il programma su parte del dataset)
# totalImages = 2000

# Creo il file csv con le frequenze se non esiste
if os.path.exists(FREQUENCY_CSV) == False:
    with open(FREQUENCY_CSV, "a") as frequencyFile:
        writer = csv.writer(frequencyFile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        print("CSV per frequenze non esistente. Lo creo.")
        writer.writerow(labelsName) #scrivo il nome degli attributi come prima riga del csv


# ----------------------------
# -- Inizio iterazioni test
# ----------------------------

print("[INFO] Inizio iterazione per test...")
for iteration in range(int(totalImages / numImagesPerIteration)):
    print("Iterating from images %d to %d" % (start, start + numImagesPerIteration))

    # Carico le immagini partendo da start per numImagesPerIteration
    dataset = loadImages(start, start + numImagesPerIteration, csvLabels)
    # Creo Numpy array del dataset per la rete
    X = np.array(dataset)
    print("Iteration %d" % iteration)
    print("X.shape: ", X.shape)

    # Carico le labels
    labels = loadLabels(start, start + numImagesPerIteration, csvLabels)  # conterrà numImagesPerIteration rows così da permettere il loading di quelle immagini per il train

    labels = labels.drop(['image_id'], axis=1)
    #    print("labels (no file name) head()")
    #    print(labels.head())
    y = np.array(labels)
    print("y.shape: ", y.shape)

    # Suddivido dataset e labels per ottenere le immagini per il test
    # test_size, random_state e numImagesPerIteration devono essere identici a quelli nel file train_iterator
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    print("X_train: {0}".format(str(X_train.shape)))
    print("X_test: {0}".format(str(X_test.shape)))
    print("y_train: {0}".format(str(y_train.shape)))
    print("y_test: {0}".format(str(y_test.shape)))

#    print("probabNum = " + str(len(y_test[1])))
#    print("num IMG = " + str(len(X_test)))
#    POSSO LEGGERE L'ARRAY CON LE LABEL COME y_test[immagine][label]

    # Carico il model per il test
    model = load_model(MODEL_OUT_PATH)


    # ------------------------------------------------------------------
    # -- Itero tra le varie immagini caricate dal dataset per il test
    # ------------------------------------------------------------------

    for i in range(len(X_test)):

        img = X_test[i].reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        prob = model.predict(img)

        # Check valori corretti
        errori = np.zeros(len(y_test[1]))
        errorCount = 0

        # Salvo i risultati nel csv
        with open(FREQUENCY_CSV, "a") as frequencyFile:
            writer = csv.writer(frequencyFile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

            for errorIterator in range(len(y_test[1])):
                if(prob[0][errorIterator] == y_test[i][errorIterator]):
                    errori[errorIterator] = 1

                else:
                    errori[errorIterator] = 0
                    errorCount += 1

            writer.writerow(errori)
            print("Numero errori: " + str(errorCount))

            # Salvo il numero di errori commessi in un file per possibili utilizzi futuri
            with open(NUM_ERROR_FILE, "a") as errorFile:
                errorFile.write(str(errorCount) + "\n")
        
    # Incremento indice per prossima iterazione e salvo nel file
    start += numImagesPerIteration
    with open(CHECKPOINT_IMAGE_PATH, "w") as checkpointNumber:
        checkpointNumber.write(str(start))


