import pandas as pd
import os

os.makedirs('data/tabular', exist_ok=True)

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"

df = pd.read_csv(url)
df.columns = ['age','sex','cp','trestbps','chol','fbs',
              'restecg','thalach','exang','oldpeak','slope','ca','thal','target']

df['target'] = (df['target'] > 0).astype(int)

df.to_csv('data/tabular/heart_uci.csv', index=False)

print(f"Saved: {df.shape[0]} rows, {df.shape[1]} columns")
print(df['target'].value_counts())