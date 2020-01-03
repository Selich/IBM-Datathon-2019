import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np
from sklearn2pmml import sklearn2pmml
from sklearn2pmml import make_pmml_pipeline
def procitaj_nesredjen(df):
    df = df.drop('Unnamed: 0',axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#print(df.head())
    ids = mat[:,0]
    ids = ids.reshape((848,1))
    X = mat[:,1::3]
    X = np.append(ids, X, axis=1)
    Y = mat[:,2::3]
    Y = np.append(ids, Y, axis=1)
    Z = mat[:,3::3]
    Z = np.append(ids, Z, axis=1)
    mat = np.concatenate((X,Y,Z),axis=0)
    return mat
def get_lineage(tree, feature_names):
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]

     def recurse(left, right, child, lineage=None):
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     for child in idx:
          for node in recurse(left, right, child):
               print(node)
def main():
    parser = argparse.ArgumentParser(
        prog=__file__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i',
        '--input_path',
        help='Path to the .csv with coefs and classes',
        required=True
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    df = df.drop(df.columns[0], axis=1)
    Y = df["target_class"]
    X = df.drop("target_class", axis=1)
    print(X.shape)
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    clf = clf.fit(X_train, Y_train)
    Y_test_predicted = clf.predict(X_test)
    pipeline = make_pmml_pipeline(clf)
    sklearn2pmml(pipeline,pmml="../models/bankrot.pmml")

    #print(confusion_matrix(Y_test, Y_test_predicted))
    #df1 = pd.read_csv("../data/nov.csv")
    #A = procitaj_nesredjen(df1)
    #ids = A[:,0]
    #ids = ids.astype('int32')
    #ids = ids.reshape((ids.shape[0],1))
    #print(ids.shape)
    #features = A[:,1:]
    #labels = clf.predict(features)
    #labels = labels.reshape((labels.shape[0],1))
    #print(labels.shape)
    #final = np.append(ids,features,axis=1)
    #final = np.append(final,labels,axis=1)
    #print(final[0])
    #df_final = pd.DataFrame.from_records(final).to_csv("../data/final.csv")

if __name__ == "__main__":
    main()
