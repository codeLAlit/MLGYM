import pandas as pd
from Akash_Linreg import linearreg
import numpy as np
x=pd.read_csv("linreg_data_train_x.csv")
y=pd.read_csv("linreg_data_train_y.csv")
x=x.to_numpy()
y=y.to_numpy()

final_theta, train, test=linearreg(x, y, 10, 0.03, 300, 0.001)

print(final_theta)