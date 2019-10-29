import matplotlib
matplotlib.use("TKAgg") # Utilizzo il backend per la visualizzazione grafica delle immagini
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------
# -- Costanti path file
# ------------------------

FULL_CSV = "./csv/history_full.csv"
MEDIA_CSV = "./csv/history_media.csv"
FULL_PLOT = "./plot/FullPlot.png"
MEDIA_PLOT = "./plot/MediaPlot.png"


# ----------------------
# -- Loading file csv
# ----------------------

print("Carico file CSV...")
csvMedia = pd.read_csv(MEDIA_CSV)
csvFull = pd.read_csv(FULL_CSV)

print("Media CSV:")
print(csvMedia.head())

print("Full CSV:")
print(csvFull.head())


# ---------------------------
# -- Creazione PLOT output
# ---------------------------

plt.style.use("ggplot")

# Plot completo
fullPlt = plt.figure(1)
lenFull = len(csvFull["loss"])
print("Full lenght: %d" % lenFull)

plt.plot(np.arange(0, lenFull), csvFull["loss"], label="train_loss")
plt.plot(np.arange(0, lenFull), csvFull["val_loss"], label="val_loss")
plt.plot(np.arange(0, lenFull), csvFull["acc"], label="train_acc")
plt.plot(np.arange(0, lenFull), csvFull["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs * Iterations")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="center right")
plt.savefig(FULL_PLOT)

# Plot media
mediaPlt = plt.figure(2)
lenMedia = len(csvMedia["loss"])
print("Media lenght: %d" % lenMedia)

plt.plot(np.arange(0, lenMedia), csvMedia["loss"], label="train_loss")
plt.plot(np.arange(0, lenMedia), csvMedia["val_loss"], label="val_loss")
plt.plot(np.arange(0, lenMedia), csvMedia["acc"], label="train_acc")
plt.plot(np.arange(0, lenMedia), csvMedia["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Media)")
plt.xlabel("Iterations")
plt.ylabel("Loss / Accuracy Media")
plt.legend(loc="center right")
plt.savefig(MEDIA_PLOT)
