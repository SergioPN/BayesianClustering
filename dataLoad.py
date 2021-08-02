import pandas as pd
from sklearn.preprocessing import StandardScaler

class dataLoad:
    def __init__(self):
        self.df = pd.read_csv("data/penguins_size.csv").drop(columns=["island", "sex"]).dropna()

        # Scaled
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.df.drop(columns=["species"]))
        scaled = pd.DataFrame(scaled, columns=self.df.columns[1:])
        scaled["species"] = list(self.df["species"])
        self.scaled_df = scaled

        self.df.species = pd.Categorical(self.df.species)
        self.scaled_df.species = pd.Categorical(self.df.species)





