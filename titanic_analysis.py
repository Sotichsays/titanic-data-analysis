# Titanic Data Analysis
# Автор: Игорь Сотиков
# Описание: Анализ данных пассажиров Титаника: пол, возраст, класс, выживаемость

import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean_data(filepath):
    """Загружает данные и заполняет пропуски"""
    df = pd.read_csv(filepath)
    median_age = df['Age'].median()
    df['Age_filled'] = df['Age'].fillna(median_age)
    df['is_child'] = df['Age_filled'] < 18
    return df

def analyze_survival(df):
    """Анализирует выживаемость по разным группам"""
    results = {
        'by_class': df.groupby('Pclass')['Survived'].mean(),
        'by_sex': df.groupby('Sex')['Survived'].mean(),
        'by_child': df.groupby('is_child')['Survived'].mean()
    }
    return results

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

# 1. Проверяем пропуски в каждом столбце
print("\n" + "="*50)
print("АНАЛИЗ ПРОПУСКОВ В ДАННЫХ")
print("="*50)

print(df.isnull().sum())

# Сколько процентов пропусков в возрасте?
age_null_percent = (df['Age'].isnull().sum() / len(df)) * 100
print(f"\nПропуски в возрасте: {age_null_percent:.1f}%")

# 2. Заполняем пропуски в возрасте медианой
print("\n" + "="*50)
print("ЗАПОЛНЕНИЕ ПРОПУСКОВ")
print("="*50)

# Медиана возраста (без пропусков)
median_age = df['Age'].median()
print(f"Медианный возраст: {median_age}")

# Заполняем пропуски
df['Age_filled'] = df['Age'].fillna(median_age)

# Проверяем, что пропусков больше нет
print(f"Пропусков после заполнения: {df['Age_filled'].isnull().sum()}")

# 3. Создаём признак "ребёнок" (до 18 лет)
df['is_child'] = df['Age_filled'] < 18

# Проверяем
print("\n" + "="*50)
print("АНАЛИЗ: ДЕТИ vs ВЗРОСЛЫЕ")
print("="*50)

child_survival = df.groupby('is_child')['Survived'].mean()
print(child_survival)

# Добавим понятные названия
child_survival.index = ['Взрослые', 'Дети']
print(f"\nШанс выжить у детей: {child_survival['Дети']*100:.1f}%")
print(f"Шанс выжить у взрослых: {child_survival['Взрослые']*100:.1f}%")

# 4. Визуализация результатов
print("\n" + "="*50)
print("СТРОИМ ГРАФИКИ")
print("="*50)

import matplotlib.pyplot as plt

# Фигура с 2x2 подграфиками
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# График 1: Выживаемость по классу
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[0, 0], color=['gold', 'silver', 'brown'])
axes[0, 0].set_title('Выживаемость в зависимости от класса билета')
axes[0, 0].set_xlabel('Класс')
axes[0, 0].set_ylabel('Доля выживших')
axes[0, 0].set_ylim(0, 1)

# График 2: Выживаемость по полу
df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0, 1], color=['blue', 'pink'])
axes[0, 1].set_title('Выживаемость: мужчины vs женщины')
axes[0, 1].set_xlabel('Пол')
axes[0, 1].set_ylabel('Доля выживших')
axes[0, 1].set_ylim(0, 1)

# График 3: Выживаемость по возрасту (гистограмма)
df[df['Survived'] == 1]['Age_filled'].hist(bins=20, alpha=0.7, label='Выжившие', color='green', ax=axes[1, 0])
df[df['Survived'] == 0]['Age_filled'].hist(bins=20, alpha=0.7, label='Погибшие', color='red', ax=axes[1, 0])
axes[1, 0].set_title('Распределение возраста: выжившие vs погибшие')
axes[1, 0].set_xlabel('Возраст')
axes[1, 0].set_ylabel('Количество')
axes[1, 0].legend()

# График 4: Дети vs взрослые
child_survival.plot(kind='bar', ax=axes[1, 1], color=['orange', 'cyan'])
axes[1, 1].set_title('Дети выживают чаще?')
axes[1, 1].set_xlabel('Категория')
axes[1, 1].set_ylabel('Доля выживших')
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('titanic_analysis_charts.png', dpi=150)
plt.show()

print("Графики сохранены в файл 'titanic_analysis_charts.png'")

# 5. Сохраняем очищенные данные для будущих проектов
df.to_csv('titanic_cleaned.csv', index=False)
print("Очищенный датасет сохранён как 'titanic_cleaned.csv'")

if __name__ == "__main__":
    df = load_and_clean_data('titanic.csv')
    results = analyze_survival(df)

    print("\nРезультаты анализа:")
    print("\nПо классу билета:")
    print(results['by_class'])
    print("\nПо полу:")
    print(results['by_sex'])
    print("\nПо возрасту (ребёнок/взрослый):")
    print(results['by_child'])

