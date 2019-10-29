# -------------------------
# -- Librerie necessarie
# -------------------------

from keras.preprocessing.image import image
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TKAgg") #Utilizzo il backend per la visualizzazione grafica delle immagini
import matplotlib.pyplot as plt


# ------------------------------------------
# -- Costanti contenenti path e dati vari
# ------------------------------------------

MODEL_DIR = "./model/"
CSV_DIR = "csv" #sottocartella contenente i file csv
MODEL_PATH = MODEL_DIR + "trainedCelebA.h5"
IMAGE_PATH = "./test_img/4.jpg"
IMAGE_WIDTH = 178
IMAGE_HEIGHT = 218
IMAGE_DIMS = (IMAGE_WIDTH, IMAGE_HEIGHT, 3) #dimensioni immagine
LABELS_CSV_PATH = "./" + CSV_DIR + "/list_attr_celeba_zeros.csv"


# ----------------------------------------
# -- Loading immagine e labels per test
# ----------------------------------------

# Carico e pre-processo immagine
img = image.load_img(IMAGE_PATH, target_size=IMAGE_DIMS)
img = image.img_to_array(img)
img = img/255

print("[INFO] Carico file CSV...")

# Carico i labels per ottenere le etichette e droppo quelle non utilizzate nel train
csvLabels = pd.read_csv(LABELS_CSV_PATH)
featureToDrop = ['5_o_Clock_Shadow', 'Attractive', 'Chubby',
                 'High_Cheekbones', 'Oval_Face', 'Pale_Skin',
                 'Rosy_Cheeks', 'Wearing_Lipstick']
for feature in featureToDrop:
    csvLabels = csvLabels.drop([feature], axis=1)
print("csvLabels full head():")
print(csvLabels.head())

labels = np.array(csvLabels.columns[1:]) #non prendo in considerazione la column image_id


# -----------------------------
# -- Loading model e predict
# -----------------------------

# Carico il modello
model = load_model(MODEL_PATH)
# Classifico l'immagine input
prob = model.predict(img.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

print(labels.shape[0])

# --------------------------
# -- Stampo le predizioni
# --------------------------

for i in range(labels.shape[0]):
    print("{}".format(labels[i]) + " ({:.3})".format(prob[0][i]))
plt.imshow(img)
plt.show()