import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("generative_models/transformer/autotrain/train_ALZHEIMER/ALZHEIMER/weights/History_128_epo=35_20250423.csv")
df = df.drop(columns=["Unnamed: 0"])  # Удаляем лишний столбец, если есть

# Установка стиля графика
plt.style.use('default')

# Построение графиков
num_cols = len(df.columns) - 1  # исключая 'epochs'
fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 3 * num_cols), sharex=True)

for ax, col in zip(axes, df.columns[1:]):  # Пропускаем 'epochs'
    ax.plot(df['epochs'], df[col], marker='o')
    ax.set_title(f'{col} over Epochs', fontsize=12)
    ax.set_ylabel(col)
    ax.grid(True)

axes[-1].set_xlabel('Epochs')

plt.tight_layout()
plt.show()