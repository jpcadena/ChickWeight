import pandas as pd
import numpy as np

df = pd.read_csv('chickweight.csv', header=True, index=False)
print(df)