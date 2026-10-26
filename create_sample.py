import pandas as pd

df = pd.read_csv("C:/Users/SUJAL/OneDrive/Desktop/data set/house_prices.csv")

sample = df.sample(
    n=min(2000, len(df)),
    random_state=42
)

sample.to_csv("house_prices_sample.csv", index=False)


print("Sample created successfully!")
print("Rows:", len(sample))
print("Columns:", len(sample.columns))
print(df.info)
print(df.dtypes)