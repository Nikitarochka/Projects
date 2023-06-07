'''
# Увязка координат скважин с сейсмической сеткой

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# Загрузка данных сейсмической сетки в формате XYZ
data_file = "C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It\\" \
            "Увязка координат скважин сейсмической сеткой\\Extract value^ CUB_FIL4_ANON [Realized] 1; -9"
data = pd.read_csv(data_file, names=["X", "Y", "Z"], sep=' ', header=None)
data.sort_values('X', ascending=False)

# Создание таблицы данных матричного типа
matrix = pd.pivot_table(data, index='Y', columns='X', values="Z")

# Загрузка координат скважин из файла
well_file = "C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It" \
            "\\Увязка координат скважин сейсмической сеткой\\well_coord.xlsx"
wells = pd.read_excel(well_file)
well_coords = wells[['X', 'Y']].values

# Нахождение ближайшего узла для каждой скважины
seismic_coords = data[['X', 'Y']].values
distances = cdist(well_coords, seismic_coords)
min_indices = np.argmin(distances, axis=1)
well_nodes = np.array([seismic_coords[i] for i in min_indices])

# Построение тепловой карты и задание отметок скважин
sns.scatterplot(data=well_coords, x=well_coords[:, 0], y=well_coords[:, 1], color="black", s=100)
sns.scatterplot(data=well_nodes, x=well_nodes[:, 0], y=well_nodes[:, 1], color="blue", s=50)
sns.heatmap(matrix)
plt.gca().invert_yaxis()
plt.show()
'''
'''
# Визуализация накопленной добычи жидкости (LPT/LPTH)
# Нужно 10-15 секунд подождать, чтобы показал график

import pandas as pd
import plotly.graph_objs as go

# чтение файлов
df_hist = pd.read_excel("C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It\\Визуализация накопленной добычи жидкости\\LIQUID_hist.xlsx")
df_rates = pd.read_excel('C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It\\Визуализация накопленной добычи жидкости\\LIQUID_rates.xlsx')

# создание графика
fig = go.Figure()

# добавление диаграмм из расчетного файла
fig.add_trace(go.Scatter(x=df_rates.iloc[10:, 0], y=df_rates.iloc[10:, 3],  mode='lines', name="rates", opacity=0.6))

# добавление диаграммы исторических значений
fig.add_trace(go.Scatter(x=df_hist.iloc[10:, 0], y=df_hist.iloc[10:, 3], mode='lines', name='Historical', marker=dict(color='red', size=5)))

# отображение графика
fig.show()
'''
'''
# Оценка изменения объема растворенного газа для набора моделей

import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

# путь к архиву с PRT-файлами
path_prt = "C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It\\Оценка изменения объема растворенного газа для набора моделей\\prts400\\prts400\\"

# проходимся по всем файлам в папке
files = [f for f in listdir(path_prt) if isfile(join(path_prt, f))]
result = pd.DataFrame()
key = ':CURRENTLY IN PLACE       :'
for file in files:
    x = open(path_prt + file, "r")
    lines = x.readlines()
    values = []
    for line in lines:
        if key in line:
            values.append(float(line.split()[4]))
    result[file] = values
for file in files:
    plt.plot(result[file].values, label=file)
plt.show()
'''
'''
# Вероятностная оценка запасов методом Монте-Карло

import random
import matplotlib.pyplot as plt


# Функция расчета STOIIP
def calculate_stoiip(area, thickness, porosity, saturation, formation_factor, recovery):
    stoiip = (7758 * area * thickness * porosity * saturation * (1 - formation_factor) / recovery) / 1000000

    return stoiip


# Генерация случайных значений параметров с использованием нормального распределения
def generate_parameters():
    areas = []
    thicknesses = []
    porosities = []
    saturations = []
    formation_factors = []
    recoveries = []

    for i in range(1000):
        area = abs(random.gauss(100, 50))
        thickness = abs(random.gauss(50, 10))
        porosity = abs(random.gauss(0.2, 0.05))
        saturation = abs(random.gauss(0.7, 0.1))
        formation_factor = abs(random.gauss(0.3, 0.05))
        recovery = abs(random.gauss(0.5, 0.05))

        areas.append(area)
        thicknesses.append(thickness)
        porosities.append(porosity)
        saturations.append(saturation)
        formation_factors.append(formation_factor)
        recoveries.append(recovery)

    return areas, thicknesses, porosities, saturations, formation_factors, recoveries


# Функция для извлечения случайных значений параметров
def get_random_parameters(params):
    idx = random.randint(0, len(params[0]) - 1)

    return [params[i][idx] for i in range(len(params))]


# Цикл N-итераций для расчета STOIIP и заполнения списка значений запасов
N = 10000
results = []

params = generate_parameters()

for i in range(N):
    random_params = get_random_parameters(params)
    stoiip = calculate_stoiip(*random_params)

    results.append(stoiip)

# Построение диаграммы eCDF
results.sort()
y = [i / N for i in range(N)]
plt.plot(results, y, 'r--')
plt.title('eCDF')
plt.xlabel('STOIIP, млн. баррелей')
plt.ylabel('Вероятность')
plt.show()
'''
'''
# Поиск ближайших  расчетов к историческим значениям

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Парсер для чтения файла
def read_file(file):
    data = pd.read_excel(file)
    return data

# Нахождение 10 ближайших кривых по метрике MSE
def find_closest_curves(hist_data, rates_data):
    hist_cols = pd.to_numeric(hist_data.iloc[10:, 3], errors='coerce')
    rates_cols = pd.to_numeric(rates_data.iloc[10:, 3], errors='coerce')
    mse_list = []
    for col in rates_cols:
        if isinstance(col, (float, int)) and not np.isnan(col):
            mse = np.mean((hist_cols - col)**2)
            mse_list.append((col, mse))
        else:
            continue
    mse_list.sort()
    closest_curves = [x[0] for x in mse_list[:10]]
    return closest_curves

# Добавление трейсов на график
def add_traces(fig, hist_data, rates_data, closest_curves):
    # MSE
    fig.add_trace(go.Scatter(x=hist_data.iloc[10:, 0], y=closest_curves,
                             name='MSE', mode='lines', line=dict(color='blue', width=2)))

    # Исторические значения
    fig.add_trace(go.Scatter(x=hist_data.iloc[10:, 0], y=hist_data.iloc[10:, 3],
                             name='Historical', mode='lines', line=dict(color='black', width=2)))

    # Расчетные значения
    fig.add_trace(go.Scatter(x=rates_data.iloc[10:, 0], y=rates_data.iloc[10:, 3],
                             name="rates", mode='lines', line=dict(color='red', width=1), opacity=0.8))

# Чтение данных
hist_data = pd.read_excel("C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It\\Визуализация накопленной добычи жидкости\\LIQUID_hist.xlsx")
rates_data = pd.read_excel('C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It\\Визуализация накопленной добычи жидкости\\LIQUID_rates.xlsx')

# Нахождение 10 ближайших кривых
closest_curves = find_closest_curves(hist_data, rates_data)

# Создание графика
fig = go.Figure()

# Добавление трейсов на график
add_traces(fig, hist_data, rates_data, closest_curves)

# Настройки графика
fig.update_layout(title='Поиск ближайших  расчетов к историческим значениям', xaxis_title='Date', yaxis_title='LPT')

# Отображение графика
fig.show()
print(find_closest_curves(hist_data, rates_data))
'''
'''
# Построение карт энтропии

import pandas as pd
import numpy as np
import seaborn as sns

# чтение и фильтрация датасета по k-ому слою
def read_and_filter(data, k):
    filtered_data = data[data['k'] == k]
    matrix = filtered_data.pivot(index='i', columns='j', values='f')
    return matrix.values.tolist()

# частотный анализ и расчет вероятности появления события
def frequency_analysis(data):
    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / np.sum(counts)
    return dict(zip(unique, probabilities))

# оценка энтропии
def entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities))

# чтение данных из файла
data = pd.read_csv("C:\\Users\\Acer\\Documents\\Работа\\Мероприятия\\Oil case\\Задачи It\\Построение карт энтропии\\facie_cube.txt",
                   delim_whitespace=True, skiprows=5, header=None, names=['i', 'j', 'k', 'f'], dtype={'i': int, 'j': int, 'k': int, 'f': int})
data.columns = ['i', 'j', 'k', 'f']

# преобразование данных в список матриц для каждого k-ого слоя
layers = []
for k in range(1, data['k'].max() + 1):
    matrix = read_and_filter(data, k)
    layers.append(matrix)
# расчет вероятностей литологий для каждой ячейки
tensor = np.zeros_like(layers[0])
for i in range(tensor.shape[0]):
    for j in range(tensor.shape[1]):
        probabilities = []
        for layer in layers:
            value = layer[i][j]
            probabilities.append(frequency_analysis(np.array(layer).flatten())[value])
        tensor[i][j] = probabilities

# оценка энтропии для каждой ячейки
empty_df = pd.DataFrame(index=range(tensor.shape[0]), columns=range(tensor.shape[1]))
for i in range(empty_df.shape[0]):
    for j in range(empty_df.shape[1]):
        empty_df.iloc[i][j] = entropy(tensor[i][j])

# построение тепловой карты
sns.heatmap(empty_df)
'''
'''
# Построение торнадо-чарта

import plotly.graph_objs as go
import random

# Функция расчета STOIIP
def calculate_stoiip(area, thickness, porosity, saturation, formation_factor, recovery):
    stoiip = (7758 * area * thickness * porosity * saturation * (1 - formation_factor) / recovery) / 1000000
    return stoiip

# Генерация случайных значений параметров с использованием нормального распределения
def generate_parameters():
    areas = []
    thicknesses = []
    porosities = []
    saturations = []
    formation_factors = []
    recoveries = []

    for i in range(1000):
        area = abs(random.gauss(100, 50))
        thickness = abs(random.gauss(50, 10))
        porosity = abs(random.gauss(0.2, 0.05))
        saturation = abs(random.gauss(0.7, 0.1))
        formation_factor = abs(random.gauss(0.3, 0.05))
        recovery = abs(random.gauss(0.5, 0.05))

        areas.append(area)
        thicknesses.append(thickness)
        porosities.append(porosity)
        saturations.append(saturation)
        formation_factors.append(formation_factor)
        recoveries.append(recovery)

    return areas, thicknesses, porosities, saturations, formation_factors, recoveries

# Функция для извлечения случайных значений параметров
def get_random_parameters(params):
    idx = random.randint(0, len(params[0]) - 1)

    return [params[i][idx] for i in range(len(params))]


# Цикл N-итераций для расчета STOIIP и заполнения списка значений запасов
N = 10000
params = generate_parameters()
results = []

# цикл по параметрам
# цикл по параметрам
for i, param in enumerate(params):
    # список для хранения результатов расчета STOIIP при варьировании i-го параметра
    param_results = []
    # цикл по значениям i-го параметра
    for value in param:
        # создание списка базовых значений параметров
        base_values = [1, 1, 1, 1, 1, 1]
        # замена i-го значения на текущее значение
        base_values[i] = value
        # расчет STOIIP и добавление в список результатов
        STOIIP_val = calculate_stoiip(*base_values)
        param_results.append(STOIIP_val)
    # добавление результатов расчета в общий список
    results.append(param_results)

# создание списка словарей с данными для торнадо-плота
data = []
for i, param in enumerate(params):
    data.append({'values': results[i], 'label': param})

# создание торнадо-плота
fig = go.Figure(go.Treemap(
    ids=[f'param_{i}' for i in range(len(params))],
    labels=params,
    parents=[''] * len(params),
    values=[sum(results[i]) for i in range(len(params))],
    branchvalues='total',
    textinfo='label+value+percent parent',
    hovertemplate='<b>%{label}</b><br>Total STOIIP: %{value:.2f}<br>Share: %{percentParent:.1%}',
    marker=dict(
        colors=['#EF553B','#636efa', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3', '#FF6692', '#B6E880'],
        line=dict(color='#FFFFFF', width=2)
    )
))
# добавление данных для каждого параметра
for i in range(len(params)):
    y = [f'param_{i}'] * len(params[i])
    x = params[i]
    fig.add_trace(go.Bar(
        y=y,
        x=x,
        orientation='h',
        marker=dict(
            color='#EF553B',
            line=dict(color='#FFFFFF', width=2)),
        showlegend=False
))

# настройки макета
fig.update_layout(
    title='Торнадо чарт',
    xaxis=dict(title='STOIIP'),
    yaxis=dict(title='Параметры'),
    height=600,
    margin=dict(l=100, r=100, t=100, b=100),
    treemapcolorway=['#ab63fa', '#ab63fa', '#FFA15A', '#19d3f3']
)

# отображение графика
fig.show()
'''