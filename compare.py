import pandas as pd

# 讀取 csv 檔案
data1 = pd.read_csv('../Classification/results/best_weights.csv', index_col=0)
data2 = pd.read_csv('../Classification/results/howard.csv', index_col=0)

blacky = data1.character.values
howard = data2.character.values

number = 0
while True:
    try:
        if blacky[number] != howard[number]:
            print(number + 1)
            print('blacky', blacky[number])
            print('howard', howard[number])
        number += 1
    except:
        break