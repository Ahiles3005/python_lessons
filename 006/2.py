# 3. Прочитайте любой csv файл из папки sample_data.

import pandas as df
import os




file_path = os.path.join(os.getcwd(),'sample_data','customers-100.csv')
file = df.read_csv(file_path)

print(file)
