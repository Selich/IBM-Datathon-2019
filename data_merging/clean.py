import pandas as pd
import matplotlib.pyplot as plt
def main():
    df = pd.read_csv("../data/final_predict.csv")
    df = df.drop(df.columns[2:25],axis=1)
    df = df.drop('index',axis=1)

    df["odnos"] = df['bankrot'].apply(str) + df['class'].apply(str)

    plt.(df['KupacID'],df['odnos'])
    plt.show()

if __name__ == "__main__":
    main()
