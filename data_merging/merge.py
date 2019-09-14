import pandas as pd

def main():
    final = pd.read_csv("../data/final_v3.csv")
    final = final.drop('index',axis=1)
    predict = pd.read_csv("../data/final.csv")

    merged = final.merge(predict,on="KupacID")
    merged.to_csv("../data/dataset_final.csv",index=False)
if __name__ == "__main__":
    main()
