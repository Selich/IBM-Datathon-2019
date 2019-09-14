import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model_path',
            help="Input path of model",
            required=True)
    args = parser.parse_args()
    df = pd.read_csv("../data/dataset_final.csv")
    model = keras.models.load_model(args.model_path)
    ids = pd.read_csv("../data/Kupci.csv",encoding='ISO-8859-1')['KupacID']
    sume = [50000,2000000,4000000,600000]
    for id in ids:
        for suma in sume:
            print(predict(model,id,suma,df))

def predict(model, id, suma, df):
    df = df[df['KupacID'] == int(id)]
    df = df.drop(['target','index','bankrot'],axis=1)
    features = df.values

    predictions = model.predict_classes(features)
    print(predictions)
    counts = np.bincount(predictions)
    prediction = np.argmax(counts)

    rand = np.random.uniform()
    suma = int(suma)

    if suma == 500000:
        if rand < 0.1 and prediction != 3:
            prediction += 1
    elif suma == 2000000:
        if rand < 0.3 and prediction !=3:
            prediction += 1
    elif suma == 4000000:
        if rand < 0.6 and prediction !=3:
            prediction += 1
    elif suma == 6000000:
        if rand < 0.8 and prediction !=3:
            prediction += 1
    return (id,suma,prediction)
if __name__ == "__main__":
    main()
