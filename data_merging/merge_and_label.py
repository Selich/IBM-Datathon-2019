import pandas as pd
def main():
    df = pd.read_csv("../data/df_risk5.csv")
#print(df.head())
    fakture = pd.read_csv("../data/Fakture05092019.csv")

# df = df.merge(fakture, on="KupacID")

    arr = list(filter(lambda s: s.find("AOP") != -1, df.columns))
#print(arr)
    df = df.drop(arr,axis=1)


    df = df.merge(fakture, on="KupacID")
    df = df.sample(frac=1).reset_index(drop=True)

    Y = pd.DataFrame(['KupacID', 'Class'])
#print(df.head())
#print(df['risk2018low'][0])
    for i in range(5000):
        risk_no = df["risk2018none"][i]
        risk_low = df["risk2018low"][i]
        risk_medium = df["risk2018medium"][i]
        iznos = df["Iznos"][i]
        if iznos >= risk_medium:
            Class = 3
        if iznos < risk_medium and iznos >= risk_low:
            Class = 2
        if iznos < risk_low and iznos >= risk_no:
            Class = 1
        if iznos < risk_no:
            Class = 0
        Y.loc[i] = [[df["KupacID"][i], Class]]
        print(i)
    Y.to_csv("../data/final_v3.csv")
if __name__ == "__main__":
    main()
