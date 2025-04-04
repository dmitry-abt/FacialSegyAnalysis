from matplotlib import pyplot as plt
from data_handler import DataHandler, PCA, RobustScaler
from mlp import MLP
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns

def main():
    """
    Основная функция для выполнения всего процесса
    """
    
    # Конфигурация путей и параметров данных
    SEGY_PATH = "/Users/developer/Downloads/adele_seismic_survey_NW_australia_bind_cube_fault_detection.segy"
    INLINE_RANGE = (155, 200, 1)  # Диапазон инлайнов: старт, стоп, шаг
    XLINE_RANGE = (155, 200, 1)   # Диапазон кросслайнов
    SAMPLE_RANGE = (0, 50, 1)     # Диапазон временных срезов
    WELL_LOCATIONS = [(10, 20), (40, 10)]  # Синтетические координаты скважин (инлайн, кросслайн)
    
    # Инициализация обработчика данных
    dataHandler = DataHandler()

    # Загрузка и визуализация сейсмических данных
    print("Загрузка SEG-Y данных")
    seismic_data = dataHandler.load_segy_data(SEGY_PATH, INLINE_RANGE, XLINE_RANGE, SAMPLE_RANGE)
    
    # Визуализация 3D сейсмического куба
    dataHandler.plot_3d_seismic(seismic_data)
    # Визуализация вертикального разреза (первый инлайн)
    dataHandler.plot_section(seismic_data, 0, "Исходные сейсмические данные")
    
    # Расчет атрибутов
    print("\nРасчет сейсмических атрибутов")
    attributes = dataHandler.enhanced_attributes(seismic_data)
    
    # Загрузка фаций
    print("\nЗагрузка данных по фациям")
    # Генерация синтетических фаций на основе расположения скважин
    facies = dataHandler.load_facies(seismic_data.shape, WELL_LOCATIONS)
    # 3D визуализация распределения фаций
    dataHandler.plot_3d_facies(facies)
    # 2D срез распределения фаций
    dataHandler.plot_section(facies, 0, "Исходные фации", cmap=mcolors.ListedColormap(['black', 'red', 'green', 'blue']))
    
    # Подготовка данных для обучения модели
    print("\nПодготовка данных для обучения")
    # Преобразование 3D атрибутов в 2D матрицу
    X = attributes.reshape(-1, attributes.shape[-1])
    # Преобразование 3D фаций в 1D вектор
    y = facies.reshape(-1)
    # Выбор только точек с известными фациями (исключаем -1)
    valid_idx = y != -1
    X, y = X[valid_idx], y[valid_idx]
    
    # Балансировка классов
    X_res, y_res = dataHandler.simple_oversample(X, y)
    
    # Предобработка данных
    # Масштабирование устойчивое к выбросам
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # Уменьшение размерности с сохранением 95% дисперсии
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"Уменьшено с {X.shape[1]} до {pca.n_components_} признаков")
    
    # Разделение данных
    X_train, X_val, y_train, y_val = dataHandler.train_test_split(
        X_reduced, y_res, 
        test_size=0.2,       # 20% данных в валидационную выборку
        stratify=y_res,      # Сохраняем распределение классов
        random_state=42      # Фиксируем random state для воспроизводимости
    )
    
    #  Создание и обучение модели нейронной сети
    print("\nСоздание и обучение модели")
    model = MLP(
        layer_sizes=[X_reduced.shape[1],    # Входной слой (число признаков после PCA)
        1024, 512, 256, 128,                # Скрытые слои
        len(np.unique(y_res))],             # Выходной слой (число классов)
        activation='leaky_relu'             # Функция активации
    )
    
    # Обучение модели с указанными параметрами
    history = model.train(
        X_train, y_train,   # Обучающие данные
        X_val, y_val,       # Валидационные данные
        epochs=200,         # Максимальное число эпох
        batch_size=256,     # Размер мини-батча
        initial_lr=0.01,    # Начальная скорость обучения
        min_lr=0.0001,      # Минимальная скорость обучения
        patience=50         # Число эпох для ранней остановки
    )
    
    # Оценка результатов
    print("\nОценка модели:")
    # Визуализация кривых обучения (loss и accuracy)
    dataHandler.plot_history(history)
    
    # Получение предсказаний на валидационной выборке
    y_pred = np.argmax(model.forward(X_val, training=False)[0], axis=1)
    
    # Построение матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        model.confusion_matrix(y_val, y_pred, len(np.unique(y_res))), 
        annot=True,    # Отображать значения в ячейках
        fmt='d',       # Целочисленный формат
        cmap='Blues'   # Цветовая схема
    )
    plt.title('Матрица ошибок')
    plt.show()
    
    # Вывод отчета о классификации
    print(model.print_classification_report(y_val, y_pred))
    
    # Применение обученной модели ко всему объему данных
    print("\nПрименение модели ко всему объему данных")
    # Преобразование всего объема данных
    X_full = pca.transform(scaler.transform(attributes.reshape(-1, attributes.shape[-1])))
    # Получение прогнозов для всего объема
    preds = np.argmax(model.forward(X_full, training=False)[0], axis=1)
    
    # Создание массива предсказанных фаций того же размера, что исходные данные
    predicted_facies = np.full(facies.shape, -1, dtype=int)
    # Заполнение только тех точек, где были известны фации
    predicted_facies.reshape(-1)[valid_idx] = preds[:np.sum(valid_idx)]
    
    # Визуализация результатов
    # 3D распределение предсказанных фаций
    dataHandler.plot_3d_facies(predicted_facies)
    # 2D срез предсказанных фаций (10-й инлайн)
    dataHandler.plot_section(predicted_facies, 10, "Предсказанные фации",
                cmap=mcolors.ListedColormap(['black', 'red', 'green', 'blue']))

if __name__ == "__main__":
    main()
