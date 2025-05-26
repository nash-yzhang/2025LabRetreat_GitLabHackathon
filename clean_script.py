#%%
import numpy as np
import csv
#%%
with open('visual_neuroscience_dummy_data_4min.csv') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
#%%
print(data)