from matplotlib import pyplot as plt
from data_handler import DataHandler, PCA, RobustScaler
from mlp import MLP
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns

def main():
    # Конфигурация
    SEGY_PATH = "/Users/developer/Downloads/adele_seismic_survey_NW_australia_bind_cube_fault_detection.segy"
    INLINE_RANGE = (155, 200, 1)  # Старт, стоп, шаг
    XLINE_RANGE = (155, 200, 1)
    SAMPLE_RANGE = (0, 50, 1)
    WELL_LOCATIONS = [(10, 20), (40, 10)]
    
    dataHandler = DataHandler()

    # Загрузка данных
    print("Загрузка SEG-Y данных")
    seismic_data = dataHandler.load_segy_data(SEGY_PATH, INLINE_RANGE, XLINE_RANGE, SAMPLE_RANGE)
    dataHandler.plot_3d_seismic(seismic_data)
    dataHandler.plot_section(seismic_data, 0, "Исходные сейсмические данные")
    
    # Расчет атрибутов
    print("\nРасчет сейсмических атрибутов")
    attributes = dataHandler.enhanced_attributes(seismic_data)
    
    # Загрузка фаций
    print("\nЗагрузка данных по фациям")
    facies = dataHandler.load_facies(seismic_data.shape, WELL_LOCATIONS)
    dataHandler.plot_3d_facies(facies)
    dataHandler.plot_section(facies, 0, "Исходные фации", cmap=mcolors.ListedColormap(['black', 'red', 'green', 'blue']))
    
    # Подготовка данных
    print("\nПодготовка данных для обучения")
    X = attributes.reshape(-1, attributes.shape[-1])
    y = facies.reshape(-1)
    valid_idx = y != -1
    X, y = X[valid_idx], y[valid_idx]
    
    # Балансировка классов
    X_res, y_res = dataHandler.simple_oversample(X, y)
    
    # Метод анализа главных компонент
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(RobustScaler().fit_transform(X_res))
    print(f"Уменьшено с {X.shape[1]} до {pca.n_components_} признаков")
    
    # Разделение данных
    X_train, X_val, y_train, y_val = dataHandler.train_test_split(
        X_reduced, y_res, test_size=0.2, stratify=y_res, random_state=42)
    
    # Обучение модели
    print("\nСоздание и обучение модели")
    model = MLP(
        layer_sizes=[X_reduced.shape[1], 1024, 512, 256, 128, len(np.unique(y_res))],
        activation='leaky_relu'
    )
    
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=200,
        batch_size=256,
        initial_lr=0.01,
        min_lr=0.0001,
        patience=50
    )
    
    # Оценка результатов
    print("\nОценка модели:")
    dataHandler.plot_history(history)
    
    y_pred = np.argmax(model.forward(X_val, training=False)[0], axis=1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(model.confusion_matrix(y_val, y_pred, len(np.unique(y_res))), annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.show()
    
    print(model.print_classification_report(y_val, y_pred))
    
    # Применение ко всему объему
    print("\nПрименение модели ко всему объему данных")
    X_full = pca.transform(RobustScaler().fit_transform(
        attributes.reshape(-1, attributes.shape[-1])))
    preds = np.argmax(model.forward(X_full, training=False)[0], axis=1)
    
    predicted_facies = np.full(facies.shape, -1, dtype=int)
    predicted_facies.reshape(-1)[valid_idx] = preds[:np.sum(valid_idx)]
    dataHandler.plot_3d_facies(predicted_facies)
    dataHandler.plot_section(predicted_facies, 10, "Предсказанные фации",
                cmap=mcolors.ListedColormap(['black', 'red', 'green', 'blue']))

if __name__ == "__main__":
    main()
