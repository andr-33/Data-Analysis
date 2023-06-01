import pandas as pd
import numpy as np
import matplotlib as plt 
from matplotlib import pyplot


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, names=None)
df.columns = headers

df.replace("?", np.nan, inplace=True)
missing_data = df.isnull()

#Manejar la data vacia
df["normalized-losses"] = df["normalized-losses"].astype("float")
df["normalized-losses"].replace(np.nan, df["normalized-losses"].mean(), inplace=True)
df["normalized-losses"] = df["normalized-losses"].astype("int")

df["stroke"] = df["stroke"].astype("float")
df["stroke"].replace(np.nan, df["stroke"].mean(), inplace=True)

df["bore"] = df["bore"].astype("float")
df["bore"].replace(np.nan, df["bore"].mean(), inplace=True)

df["horsepower"] = df["horsepower"].astype("float")
df["horsepower"].replace(np.nan, df["horsepower"].mean(), inplace=True)
df["horsepower"] = df["horsepower"].astype("int")

df["peak-rpm"] = df["peak-rpm"].astype("float")
df["peak-rpm"].replace(np.nan, df["peak-rpm"].mean(), inplace=True)


df["num-of-doors"].replace(np.nan, "four", inplace=True)

df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df["price"] = df["price"].astype("float")

#Estandarizar
df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns={"city-mpg":"city-L/100Km"}, inplace=True)
df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns={"highway-mpg":"highway-L/100Km"}, inplace=True)

#Normalizar
df["length"] = df["length"]/df["length"].max()
df["width"] = df["width"]/df["width"].max()
df["height"] = df["height"]/df["height"].max()

#Binning
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ["Low", "Medium", "High"]
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)

#Dummy Variable

dummy_v = pd.get_dummies(df["fuel-type"])
dummy_v.rename(columns={"gas":"fuel-type-gas", "diesel":"fuel-type-diesel"}, inplace=True)

df = pd.concat([df, dummy_v], axis=1)
df.drop("fuel-type", axis=1, inplace=True)

dummy_v2 = pd.get_dummies(df["aspiration"])
dummy_v2.rename(columns={"std":"aspiration-standard", "turbo":"aspiration-turbo"}, inplace=True)

df = pd.concat([df,dummy_v2], axis=1)
df.drop("aspiration", axis=1, inplace=True)


df.to_csv('~/Escritorio/clean_df.csv')
