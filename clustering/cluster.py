import sklearn.preprocessing
import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser(
        prog=__file__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i',
        '--input_path',
        help='Input .csv file with coeficients',
        required=True
    )
    parser.add_argument(
        '-o',
        '--output_path',
        help='Output path for .csv file with classes',
        required=True
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    df = df.drop('Unnamed: 0',axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
# Convert DataFrame to matrix if all are float or convertable to float
    mat = df.values
    X = mat[:,0::3]
    Y = mat[:,1::3]
    Z = mat[:,2::3]
    mat = np.concatenate((X,Y,Z),axis=0)
    mat1 = sklearn.preprocessing.normalize(mat,axis=0, norm='max')
# Using sklearn
    km = KMeans(n_clusters=2,n_init=10)
    predicted = km.fit_predict(mat1)
    print(predicted.min(),predicted.max(),predicted.mean())
    df = pd.DataFrame.from_records(mat)
    df["target_class"] = predicted
    df.to_csv(args.output_path)
    pd.DataFrame.from_records(mat1).to_csv("../data/normalized.csv")
    print("End!")
if __name__ == "__main__":
    main()
