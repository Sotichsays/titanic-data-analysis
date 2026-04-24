# Titanic Data Analysis
# Автор: Игорь Сотиков
# Описание: Анализ данных пассажиров Титаника: пол, возраст, класс, выживаемость

import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv('titanic.csv')
print("Данные загружены. Размер таблицы:", df.shape)

# 2. Общая информация
print("\nПервые 5 строк:")
print(df.head())

print("\nТипы данных в колонках:")
print(df.dtypes)

# 3. Анализ выживаемости по полу
print("\nКоличество мужчин и женщин:")
print(df['Sex'].value_counts())

# 4. Средний возраст пассажиров
print("\nСредний возраст пассажиров:", round(df['Age'].mean(), 1))

# 5. Количество выживших и погибших
print("\nВыжившие (1) и погибшие (0):")
print(df['Survived'].value_counts())

# 6. Шанс выжить в зависимости от класса
print("\nШанс выжить по классу билета:")
print(df.groupby('Pclass')['Survived'].mean())

# 7. Визуализация: столбчатая диаграмма
df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Выжившие vs Погибшие на Титанике')
plt.xlabel('Статус (0 = погиб, 1 = выжил)')
plt.ylabel('Количество пассажиров')
plt.savefig('survived_chart.png')  # сохраняем график
plt.show()

print("\nАнализ завершён. График сохранён как survived_chart.png")