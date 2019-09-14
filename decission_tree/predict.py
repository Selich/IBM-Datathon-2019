import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np
def main():
    parser = argparse.ArgumentParser(
        prog=__file__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i',
        '--input_path',
        help='Path to the .csv with coefs',
        required=True
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    df = df.drop(df.columns[0], axis=1)
    print(df.head())
    Y = df["target_class"]
    X = df.drop("target_class", axis=1)
    print(X.shape)
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    clf = clf.fit(X_train, Y_train)
    Y_test_predicted = clf.predict(X_test)

    print(confusion_matrix(Y_test, Y_test_predicted))
    #get_lineage(clf, X.columns)

if __name__ == "__main__":
    main()
