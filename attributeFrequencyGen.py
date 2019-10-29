import pandas as pd
import csv
import matplotlib
matplotlib.use("TKAgg") # Utilizzo il backend per la visualizzazione grafica delle immagini
import matplotlib.pyplot as plt

# -----------------------------------------
# -- Costanti contenenti path per i file
# -----------------------------------------

FREQUENCY_CSV = "./csv/attributes_frequency.csv"
FREQUENCY_PERCENT_CSV = "./csv/attributes_frequency_percent.csv"
FREQ_PLOT = "./plot/attributes_frequency.png"


# ----------------------------
# -- Lettura file frequenze
# ----------------------------

csvLabels = pd.read_csv(FREQUENCY_CSV)
print("Frequency CSV sample: ")
print(csvLabels.head())

rowCount = csvLabels.shape[0]
print("Numero di righe: " + str(rowCount))

labelsName = list(csvLabels.columns)

# Creo dizionario che conterrà gli attributi predetti correttamente
trueAttributes = {}
for col in labelsName:
    trueAttributes[col] = 0

# Creo dizionario che conterrà gli attributi non predetti correttamente
falseAttributes = dict(trueAttributes)
# Creo dizionario che conterrà la frequenza degli attributi
frequencyAttributes = dict(trueAttributes)

# Itero tra i vari attributi
for i in range(rowCount):
    for col in labelsName:
        print(col + ": " + str(csvLabels.loc[i, col]))
        if csvLabels.loc[i, col] == 1:
            print("Attributo corretto")
            trueAttributes[col] += 1
        elif csvLabels.loc[i, col] == 0:
            print("Attributo errato")
            falseAttributes[col] += 1
        else:
            print("WUT")

print("Attributi corretti:")
print(trueAttributes)

print("Attributi errati:")
print(falseAttributes)

# Calcolo percentuali
for col in labelsName:
    frequencyAttributes[col] = (trueAttributes[col] * 100) / rowCount

averageFrequency = 0
for col in labelsName:
    averageFrequency += frequencyAttributes[col]
print(len(labelsName))
averageFrequency = averageFrequency / len(labelsName)

print(frequencyAttributes)
print("Average: " + str(averageFrequency) + "%")

# -----------------
# -- Output dati
# -----------------

# Salvo la percentuale in un file csv
with open(FREQUENCY_PERCENT_CSV, "a") as freqPercCsv:
    writer = csv.writer(freqPercCsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
    attributes = []
    for col in labelsName:
        attributes.append(col)
    attributes.append("Average")
    writer.writerow(attributes)

    frequency = []
    for col in labelsName:
        frequency.append(frequencyAttributes[col])
    frequency.append(averageFrequency)
    print(frequency)
    writer.writerow(frequency)

# ------------------------------------------------------------
# -- Creo PLOT contenente le percentuali per ogni attributo
# ------------------------------------------------------------

plt.style.use("ggplot")
plt.figure(figsize=(12, 12))
plt.barh(attributes, frequency, color='green')
plt.title("Feature accuracy")
plt.xlabel("%")
plt.grid(False)
plt.savefig(FREQ_PLOT)