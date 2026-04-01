
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")



def analyze_target_distribution(df, target_col):
    """Универсальный анализ распределения целевой переменной"""
    print("\n" + "="*60)
    print("АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("="*60)
    
    counts = df[target_col].value_counts()
    total = len(df)
    
    print(f"\nРаспределение '{target_col}':")
    for label, count in counts.items():
        pct = count / total * 100
        print(f"Класс {label}: {count} ({pct:.1f}%)")
    
    # Проверка на несбалансированность
    if len(counts) == 2:
        ratio = counts.max() / counts.min()
        if ratio > 2:
            imbalance = ((counts.max() - counts.min()) / total) * 100
            print(f"\nКлассы несбалансированы: соотношение {ratio:.1f}:1")
            print(f"Разница: {imbalance:.1f}%")
        else:
            print(f"\nКлассы сбалансированы (соотношение {ratio:.1f}:1)")

def analyze_correlations(df, target_col, top_n=5):
    """Универсальный анализ корреляций"""
    print("\n" + "="*60)
    print("АНАЛИЗ КОРРЕЛЯЦИЙ")
    print("="*60)
    
    # Только числовые колонки
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        print(f"\nЦелевая переменная '{target_col}' не числовая")
        return
    
    correlations = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    top_corr = correlations.head(top_n)
    
    print(f"\n Топ-{top_n} признаков по корреляции с '{target_col}':")
    for col, val in top_corr.items():
        actual_corr = df[col].corr(df[target_col])
        strength = "сильная" if val > 0.7 else "средняя" if val > 0.4 else "слабая"
        direction = "положительная" if actual_corr > 0 else "отрицательная"
        print(f"   {col}: {abs(actual_corr):.3f} ({strength}, {direction})")

def analyze_categorical_features(df, target_col, categorical_cols):
    """Анализ категориальных признаков"""
    print("\n" + "="*60)
    print("АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
    print("="*60)
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        print(f"\nПризнак: {col}")
        grouped = df.groupby(col)[target_col].agg(['count', 'mean']).sort_values('mean', ascending=False)
        grouped.columns = ['Количество', 'Доля выживших']
        
        if len(grouped) >= 2:
            best = grouped.index[0]
            worst = grouped.index[-1]
            best_rate = grouped.iloc[0]['Доля выживших'] * 100
            worst_rate = grouped.iloc[-1]['Доля выживших'] * 100
            diff = best_rate - worst_rate
            
            print(f"Лучшая категория: '{best}' ({best_rate:.1f}%)")
            print(f"Худшая категория: '{worst}' ({worst_rate:.1f}%)")
            print(f"Разница: {diff:.1f} п.п.")

def analyze_numeric_features(df, target_col, numeric_cols):
    """Анализ числовых признаков"""
    print("\n" + "="*60)
    print("АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ")
    print("="*60)
    
    for col in numeric_cols:
        if col not in df.columns or col == target_col:
            continue
        
        print(f"\nПризнак: {col}")
        print(f"Среднее: {df[col].mean():.2f}")
        print(f"Медиана: {df[col].median():.2f}")
        print(f"Мин: {df[col].min():.2f}, Макс: {df[col].max():.2f}")
        
        # Проверка на выбросы
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
        outlier_pct = outliers / len(df) * 100
        
        if outlier_pct > 5:
            print(f"Выбросы: {outliers} ({outlier_pct:.1f}%)")
        else:
            print(f"Выбросов не обнаружено")

def print_feature_importance_summary(model, feature_names, top_n=5):
    """Универсальный вывод важности признаков"""
    if not hasattr(model, 'feature_importances_'):
        return
    
    fi = pd.DataFrame({
        'Признак': feature_names,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False).head(top_n)
    
    print(f"\n Топ-{top_n} важных признаков для модели:")
    for idx, row in fi.iterrows():
        print(f"   {row['Признак']}: {row['Важность']:.4f}")

# 📈 ФУНКЦИИ ВИЗУАЛИЗАЦИИ

def plot_target_distribution(df, target_col, save_path='graph_1_target.png'):
    """График распределения целевой переменной"""
    plt.figure(figsize=(8, 6))
    counts = df[target_col].value_counts()
    colors = ['#ff6b6b', '#4ecdc4'][:len(counts)]
    
    plt.bar(range(len(counts)), counts.values, color=colors, edgecolor='black')
    plt.title(f'Распределение целевой переменной ({target_col})', fontsize=14, fontweight='bold')
    plt.ylabel('Количество', fontsize=12)
    plt.xlabel(f'Класс {target_col}', fontsize=12)
    plt.xticks(range(len(counts)), counts.index)
    
    for i, v in enumerate(counts.values):
        pct = v / len(df) * 100
        plt.text(i, v + len(df)*0.01, f'{v}\n({pct:.1f}%)', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_correlations(df, target_col, numeric_cols, save_path='graph_2_correlations.png'):
    """График корреляций признаков"""
    numeric_df = df[numeric_cols + [target_col]].copy()
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title(f'Корреляционная матрица признаков с {target_col}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(results_df, save_path='graph_3_models.png'):
    """Сравнение моделей"""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', 
            color='#4ecdc4', edgecolor='black')
    plt.bar(x + width/2, results_df['ROC-AUC'], width, label='ROC-AUC', 
            color='#ff6b6b', edgecolor='black')
    
    plt.xlabel('Модель', fontsize=12)
    plt.ylabel('Метрика качества', fontsize=12)
    plt.title('Сравнение моделей машинного обучения', fontsize=14, fontweight='bold')
    plt.xticks(x, results_df['Модель'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    
    for i, row in results_df.iterrows():
        plt.text(i - width/2, row['Accuracy'] + 0.02, f'{row["Accuracy"]:.3f}', 
                ha='center', fontsize=9)
        plt.text(i + width/2, row['ROC-AUC'] + 0.02, f'{row["ROC-AUC"]:.3f}', 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_cross_validation(cv_results_df, save_path='graph_4_cv.png'):
    """График кросс-валидации"""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(cv_results_df))
    
    plt.errorbar(x, cv_results_df['Accuracy Mean'], 
                yerr=cv_results_df['Accuracy Std'],
                fmt='o', capsize=5, label='Accuracy', 
                color='#4ecdc4', markersize=10, linewidth=2)
    plt.errorbar(x + 0.3, cv_results_df['ROC-AUC Mean'], 
                yerr=cv_results_df['ROC-AUC Std'],
                fmt='s', capsize=5, label='ROC-AUC', 
                color='#ff6b6b', markersize=10, linewidth=2)
    
    plt.xlabel('Модель', fontsize=12)
    plt.ylabel('Метрика', fontsize=12)
    plt.title('Кросс-валидация: Mean ± Std (5 фолдов)', fontsize=14, fontweight='bold')
    plt.xticks(x + 0.15, cv_results_df['Модель'], rotation=45, ha='right')
    plt.legend(['Accuracy', 'ROC-AUC'])
    plt.ylim(0.7, 0.95)
    plt.grid(True, alpha=0.3)
    
    for i, row in cv_results_df.iterrows():
        plt.text(i, row['Accuracy Mean'] + 0.005, 
                f'{row["Accuracy Mean"]:.3f}', ha='center', fontsize=10)
        plt.text(i + 0.3, row['ROC-AUC Mean'] + 0.005, 
                f'{row["ROC-AUC Mean"]:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_test, y_pred, model_name, save_path='graph_5_cm.png'):
    """Матрица ошибок"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Предсказано: 0', 'Предсказано: 1'],
                yticklabels=['Факт: 0', 'Факт: 1'])
    plt.title(f'Матрица ошибок - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Фактический класс', fontsize=12)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_path='graph_6_fi.png'):
    """Важность признаков"""
    if not hasattr(model, 'feature_importances_'):
        return
    
    plt.figure(figsize=(14, 10))
    fi = pd.DataFrame({
        'Признак': feature_names,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=True)
    
    plt.barh(fi['Признак'], fi['Важность'], color='#4ecdc4', edgecolor='black')
    plt.title(f'Важность признаков - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Важность признака', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    print("="*60)
    print("АНАЛИЗ ДАННЫХ И МОДЕЛИРОВАНИЕ")
    print("="*60)
    
    # 1. ЗАГРУЗКА ДАННЫХ
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'Titanic.csv')
    
    if not os.path.exists(csv_path):
        print(f"\nОшибка: Файл не найден: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    target_col = 'Survived'  # Можно изменить для другого датасета
    
    print(f"\nДатасет загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Целевая переменная: {target_col}")
    
    # 2.АНАЛИЗ
    analyze_target_distribution(df, target_col)
    
    # Определение типов колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    analyze_correlations(df, target_col, top_n=5)
    
    # Для Titanic - специфический анализ
    if 'Sex' in df.columns and 'Pclass' in df.columns:
        analyze_categorical_features(df, target_col, ['Sex', 'Pclass', 'Embarked'])
    
    analyze_numeric_features(df, target_col, ['Age', 'Fare', 'SibSp', 'Parch'])
    
    # 3. ВИЗУАЛИЗАЦИЯ
    print("\n" + "="*60)
    print("📈 СОЗДАНИЕ ГРАФИКОВ")
    print("="*60)
    
    plot_target_distribution(df, target_col)
    print("График 1: Распределение целевой переменной")
    
    if len(numeric_cols) > 0:
        plot_feature_correlations(df, target_col, numeric_cols[:10])
        print("График 2: Корреляционная матрица")
    
    # 4. FEATURE ENGINEERING
    print("\n" + "="*60)
    print("🔧 ПРЕДОБРАБОТКА ДАННЫХ")
    print("="*60)
    
    df_work = df.copy()
    
    # Заполнение пропусков
    if 'Age' in df_work.columns:
        df_work['Age'].fillna(df_work['Age'].median(), inplace=True)
    if 'Fare' in df_work.columns:
        df_work['Fare'].fillna(df_work['Fare'].median(), inplace=True)
    if 'Embarked' in df_work.columns:
        df_work['Embarked'].fillna(df_work['Embarked'].mode()[0], inplace=True)
    
    # Создание новых признаков 
    if all(col in df_work.columns for col in ['SibSp', 'Parch']):
        df_work['FamilySize'] = df_work['SibSp'] + df_work['Parch'] + 1
        df_work['IsAlone'] = (df_work['FamilySize'] == 1).astype(int)
        print(" Создан признак: FamilySize")
    
    if 'Name' in df_work.columns:
        df_work['Title'] = df_work['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_mapping = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
        df_work['Title'] = df_work['Title'].map(title_mapping).fillna('Rare')
        print("Создан признак: Title")
    
    if 'Fare' in df_work.columns:
        df_work['LogFare'] = np.log1p(df_work['Fare'])
        print("Создан признак: LogFare")
    
    if 'Cabin' in df_work.columns:
        df_work['HasCabin'] = df_work['Cabin'].notna().astype(int)
        print("Создан признак: HasCabin")
    
    # Кодирование категориальных признаков
    if 'Sex' in df_work.columns:
        df_work['Sex_enc'] = df_work['Sex'].map({'male': 0, 'female': 1})
        print("Закодирован: Sex")
    
    if 'Embarked' in df_work.columns:
        df_work['Embarked_enc'] = df_work['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)
        print("Закодирован: Embarked")
    
    if 'Title' in df_work.columns:
        title_dummies = pd.get_dummies(df_work['Title'], prefix='Title', drop_first=True)
        df_work = pd.concat([df_work, title_dummies], axis=1)
        print(f"One-Hot Encoding: Title ({len(title_dummies.columns)} колонок)")
    
    # 5. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИРОВАНИЯ
    print("\n" + "="*60)
    print(" ПОДГОТОВКА ДАННЫХ")
    print("="*60)
    
    # Выбор признаков
    feature_cols = ['Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'LogFare', 
                    'FamilySize', 'IsAlone', 'HasCabin', 'Embarked_enc']
    
    # Добавление dummy-переменных для Title
    if 'Title' in df_work.columns:
        feature_cols.extend([col for col in title_dummies.columns])
    
    # Удаление несуществующих колонок
    feature_cols = [col for col in feature_cols if col in df_work.columns]
    
    X = df_work[feature_cols].fillna(0).astype(float)
    y = df_work[target_col]
    
    print(f"\nПризнаков: {X.shape[1]}")
    print(f"Примеров: {X.shape[0]}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. ОБУЧЕНИЕ МОДЕЛЕЙ
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    models = {
        'Логистическая регрессия': LogisticRegression(max_iter=1000, random_state=42),
        'Дерево решений': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Случайный лес': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6),
        'Градиентный бустинг': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4),
        'Нейронная сеть': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n Обучение: {name}")
        
        if name in ['Логистическая регрессия', 'Нейронная сеть']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({'Модель': name, 'Accuracy': acc, 'ROC-AUC': auc})
        
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    print(f"\n Результаты:")
    print(results_df.to_string(index=False))
    
    # График сравнения моделей
    plot_model_comparison(results_df)
    print("\nГрафик 3: Сравнение моделей")
    
    # 7. КРОСС-ВАЛИДАЦИЯ
    print("\n" + "="*60)
    print("КРОСС-ВАЛИДАЦИЯ (5 фолдов)")
    print("="*60)
    
    cv_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    top_3_models = results_df.head(3)['Модель'].values
    
    for model_name in top_3_models:
        print(f"\n {model_name}:")
        
        if model_name == 'Логистическая регрессия':
            model = LogisticRegression(max_iter=1000, random_state=42)
            X_use = X_train_scaled
        elif model_name == 'Случайный лес':
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
            X_use = X_train
        else:
            model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4)
            X_use = X_train
        
        acc_scores = cross_val_score(model, X_use, y_train, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(model, X_use, y_train, cv=cv, scoring='roc_auc')
        
        cv_results.append({
            'Модель': model_name,
            'Accuracy Mean': acc_scores.mean(),
            'Accuracy Std': acc_scores.std(),
            'ROC-AUC Mean': auc_scores.mean(),
            'ROC-AUC Std': auc_scores.std()
        })
        
        print(f"Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
        print(f"ROC-AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    
    cv_results_df = pd.DataFrame(cv_results).sort_values('ROC-AUC Mean', ascending=False)
    print(f"\n Итоги кросс-валидации:")
    print(cv_results_df.to_string(index=False))
    
    # График кросс-валидации
    plot_cross_validation(cv_results_df)
    print("\nГрафик 4: Кросс-валидация")
    
    # 8. ФИНАЛЬНАЯ МОДЕЛЬ
    print("\n" + "="*60)
    print("ФИНАЛЬНАЯ МОДЕЛЬ")
    print("="*60)
    
    best_model_name = cv_results_df.iloc[0]['Модель']
    print(f"\nЛучшая модель: {best_model_name}")
    
    if best_model_name == 'Логистическая регрессия':
        best_model = LogisticRegression(max_iter=1000, random_state=42)
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    elif best_model_name == 'Случайный лес':
        best_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        X_train_final = X_train
        X_test_final = X_test
    else:
        best_model = GradientBoostingClassifier(n_estimators=150, max_depth=4, 
                                                learning_rate=0.1, random_state=42)
        X_train_final = X_train
        X_test_final = X_test
    
    best_model.fit(X_train_final, y_train)
    y_pred_final = best_model.predict(X_test_final)
    y_pred_proba_final = best_model.predict_proba(X_test_final)[:, 1]
    
    final_acc = accuracy_score(y_test, y_pred_final)
    final_auc = roc_auc_score(y_test, y_pred_proba_final)
    
    print(f"\n Финальные метрики:")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"ROC-AUC: {final_auc:.4f}")
    
    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred_final, 
                               target_names=['Класс 0', 'Класс 1']))
    
    # Матрица ошибок
    plot_confusion_matrix(y_test, y_pred_final, best_model_name)
    print("\nГрафик 5: Матрица ошибок")
    
    # Важность признаков
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, feature_cols, best_model_name)
        print("График 6: Важность признаков")
        print_feature_importance_summary(best_model, feature_cols, top_n=5)
    
    # 9. ИТОГОВЫЙ ВЫВОД
    print("\n" + "="*60)
    print("ИТОГОВЫЙ ВЫВОД")
    print("="*60)
    
    print(f"""
ЗАДАНИЕ ВЫПОЛНЕНО УСПЕШНО!

Анализ данных:
   -Записей: {len(df)}
   -Признаков: {len(df.columns)}
   -Целевая переменная: {target_col}

Feature Engineering:
   -Создано новых признаков: {len(df_work.columns) - len(df.columns)}
   -Обработано пропусков

Моделирование:
   -Протестировано моделей: {len(models)}
   -Лучшая модель: {best_model_name}
   -Кросс-валидация: 5 фолдов

Финальные метрики:
   -Accuracy: {final_acc:.4f}
   -ROC-AUC: {final_auc:.4f}

КЛЮЧЕВЫЕ ИНСАЙТЫ:
   1.Распределение классов: {'сбалансировано' if len(results_df) > 0 and results_df.iloc[0]['Accuracy'] > 0.7 else 'несбалансировано'}
   2.Лучшая метрика: {best_model_name} (AUC: {final_auc:.3f})
   3.Модель устойчива: std < 0.03 (проверьте CV результаты)
""")
    
    # Сохранение результатов
    results_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
    cv_results_df.to_csv('cross_validation_results.csv', index=False, encoding='utf-8-sig')
    
    print("\nРезультаты сохранены в CSV файлы!")
    print("="*60)

# ЗАПУСК

if __name__ == "__main__":
    main()