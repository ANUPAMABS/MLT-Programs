# -*- coding: utf-8 -*-
"""FINDS

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TAO-ioORtD3xtTTp3J71sQWeatnmyPYY
"""

import pandas as pd
import numpy as np

data = pd.read_csv("FindS.csv")
X = data.iloc[:,:].values

hypo = ['0']*int(len(X[0])-1)

for i,h in enumerate(X):
  if h[-1]=="yes":
    for j in range(len(hypo)):
      if hypo[j]=="0":
        hypo[j]=h[j]
      elif h[j]!=hypo[j]:
        hypo[j]="?"

print("The specific hypothesis is: ",hypo)