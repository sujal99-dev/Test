import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,precision_score,accuracy_score
from sklearn.metrics import confusion_matrix,mean_absolute_error

data=pd.read_csv("C:\\Users\\SUJAL\\OneDrive\\Desktop\\house_prices.csv")
df=pd.DataFrame(data)
df=df.copy()
def convert_price(value):
    value = str(value).strip()

    if value == "Call for Price" or value == "":
        return np.nan   # missing value

    elif "Lac" in value:
        return float(value.replace("Lac", "").strip()) * 100000

    elif "Cr" in value:
        return float(value.replace("Cr", "").strip()) * 10000000

    else:
        try:
            return float(value)
        except:
            return np.nan
df["Amount(in rupees)"] = df["Amount(in rupees)"].apply(convert_price)
df["Amount(in rupees)"] = df["Amount(in rupees)"].replace([np.inf, -np.inf], np.nan)
df["Amount(in rupees)"] = df["Amount(in rupees)"].fillna(df["Amount(in rupees)"].mean())

df["Amount(in rupees)"] = df["Amount(in rupees)"].astype(int)
df["Bathroom"] = df["Bathroom"].str.extract(r'(\d+)')
df["Bathroom"]=pd.to_numeric(df["Bathroom"],errors="raise")
df["Bathroom"]=df["Bathroom"].fillna(df["Bathroom"].median())
df["Bathroom"]=df["Bathroom"].astype(int)
df["Balcony"] = df["Balcony"].str.extract(r'(\d+)')
df["Balcony"]=pd.to_numeric(df["Balcony"],errors="raise")
df["Balcony"]=df["Balcony"].fillna(df["Balcony"].median())
df["Balcony"]=df["Balcony"].astype(int)
df["Floor"] = df["Floor"].str.extract(r'(\d+)')
df["Floor"]=pd.to_numeric(df["Floor"],errors="raise")
df["Floor"]=df["Floor"].fillna(df["Floor"].mean())
df["Floor"]=(df["Floor"].astype(int))
df["Car Parking"] = df["Car Parking"].str.extract(r'(\d+)')
df["Car Parking"]=pd.to_numeric(df["Car Parking"],errors="raise")
df["Car Parking"]=df["Car Parking"].fillna(df["Car Parking"].mean())
df["Car Parking"]=(df["Car Parking"].astype(int))
encoded=pd.get_dummies(df,columns=["location"],drop_first=True,dtype=int)
x=df.isnull().sum()
print(x)
print(df.dtypes)

print(encoded[["Title","Car Parking","Balcony","Bathroom","Price (in rupees)"]].head(20))
print(encoded)

x_train=encoded[["Car Parking","Balcony","Bathroom"]]
y_train=df["Amount(in rupees)"]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
model=DecisionTreeRegressor()
ml=model.fit(x_train,y_train)

n3=int(input("enter car parking = "))
n4=int(input("enter balcony = "))
n5=int(input("enter bathroom = "))

m2=model.predict([[n3,n4,n5]])[0]
print(f"base on a input your house price=",round(m2))
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
