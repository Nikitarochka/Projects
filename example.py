import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
path = "C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It\\bb-python\\input\\"
df_long = pd.read_excel(path + "id-features.xlsx", sheet_name='full_table')
df_short = pd.read_excel(path + "id-features.xlsx", sheet_name='short_table')
'''
# Задание 1
print(df_short.head())
# Задание 2
df1 = pd.read_table(path + "XYZ.txt", sep=' ')
print(df1.head())
# Задание 3
print(df_long.iloc[34:50, 6:9].sum())
# Задание 4
a = df_long[df_long["OF_PresME10_Wq_H"] < 2000]
print(a.iloc[10:121:10, 0::3].sum())
# Задание 5
dict_of_statistics ={
    df_long.columns[14]:['median','mean', 'var'],
    df_long.columns[22]:['median','mean', 'var']
    }
df_group = df_long.groupby('label').agg(dict_of_statistics).round(4)
row_sums = df_group.iloc[0:, 0:].sum()
last_sum = row_sums.iloc[-1]
print(last_sum.round(2))
# Задание 6
df_1 = pd.read_table(path + "XYZ.txt", sep=' ').head(10)
df_2 = pd.read_excel(path + "data.xlsx").head(10)
result = pd.concat([df_1, df_2], axis=1, ignore_index=True)
print(result)
# Задание 7
df = pd.read_table(path + "XYZ.txt", sep=' ').head(10)
matrix = df.pivot_table(index=df.columns[1], columns=df.columns[0], values=df.columns[2])
print(matrix)
# Задание 8
merge_df_1 = pd.read_excel(path + "merge_sheets.xlsx", sheet_name='Coords')
merge_df_2 = pd.read_excel(path + "merge_sheets.xlsx", sheet_name='Data')
print(pd.merge(merge_df_1, merge_df_2, on='well'))
# Задание 9
print(merge_df_1.sort_values(merge_df_1.columns[2], ascending = False))
# Задание 10
nans_df = pd.read_excel(path + "nans_df.xlsx")
nans_df1 = nans_df.fillna(-9999)
nans_df2 = nans_df.fillna(nans_df.iloc[:,1].mean())
print(nans_df1)
print(nans_df2)
# Задание 11
noize1 = np.arange(2, 10, 1)
noize2 = np.random.normal(2, 5, 1000)
noize3 = np.random.default_rng().chisquare(2,4)
noize4 = np.linspace(0,15,5)
noize5 = np.random.lognormal(15, 3, 5)
print(noize1, noize2, noize3, noize4, noize5)
# Задание 12
def find_roots(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None, None
    sqrt_discriminant = discriminant**0.5
    root1 = (-b + sqrt_discriminant) / (2 * a)
    root2 = (-b - sqrt_discriminant) / (2 * a)
    return round(root1, 2), round(root2, 2) #возвращает корни квадратного уравнения
print(find_roots(4,8,3))
# Задание 13
message = "Hello, world!"
message_list = []
for letter in message:
    message_list.append(letter)
print(message_list)
# Задание 14
data = pd.read_excel(path + "id-features.xlsx", sheet_name='full_table')
for i in range(len(data.index)):
    if i == 2:
        break
    for j in range(len(data.columns)):
        if data.columns[j] == 'Krw_Sorw':
            break
        print(i, data.columns[j])
'''
'''
target_name = ['OF_PresME10_Wq_H_OPR', 'OF_PresME10_Wq_H_WPR', 'OF_PresME10_Wq_H_WIR','OF_PresME10_Wq_H_BHP', 'OF_PresME10_Wq_H']
for i in range(len(target_name)):
    df_long = df_long.sort_values(by = target_name[i],ascending= False).reset_index(drop = True)
    plt.plot(df_long[target_name[i]], label = target_name[i])
    plt.show()
'''
"""
df_long["color"] = df_long.iloc[:, -1].map( {1:'red',2:'green',3:'blue'} )
print(df_long["color"])
"""