from matplotlib import pyplot as plt
from data_handler import DataHandler
from mlp import MLP
import numpy as np
import segyio

def main():

    # Открываем SEG-Y файл
    file_path = "/Users/developer/Downloads/adele_seismic_survey_NW_australia_bind_cube_fault_detection.segy"

    with segyio.open(file_path, "r") as segyfile:
        seismic_data = segyio.tools.cube(segyfile)  # Загружаем данные в массив

    # Визуализируем один из разрезов
    plt.imshow(seismic_data[:, :, 0], cmap='gray', aspect='auto')
    plt.colorbar(label='Амплитуда')
    plt.title('Сейсмический разрез')
    plt.show()

    # Получение размерностей
    num_ilines = len(segyfile.ilines)
    num_xlines = len(segyfile.xlines)

    print(f'Количество ILINE: {num_ilines}')
    print(f'Количество XLINE: {num_xlines}')


    """
    # Генерация данных
    print("Генерация синтетических данных")
    data_handler = DataHandler(n_samples=1500, n_features=5, n_classes=3)
    X, y = data_handler.generate_synthetic_data()

    # Визуализация данных
    print("\nВизуализация данных до классификации")
    feature_names = ['Амплитуда', 'Частота', 'Анизотропия', 'Атрибут 4', 'Атрибут 5']
    data_handler.plot_features(X, y, feature_names)
    
    # Разделение данных (60% обучение / 20% валидация / 20% тест)
    print("\nРазделение данных на обучающую и тестовую выборки")
    X_train, X_test, y_train, y_test = data_handler.train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = data_handler.train_test_split(X_train, y_train, test_size=0.25) 

    # Создание и обучение модели
    print("\nСоздание и обучение нейронной сети")
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    
    # Инициализация модели
    model = MLP(input_size=input_size, 
                hidden_sizes=[64, 32],      # Два скрытых слоя с 64 и 32 нейронами
                output_size=output_size,    # Три выходных нейрона для трех классов
                activation='relu',          # Функция активации скрытых слоев
                learning_rate=0.01,         # Скорость обучения
                optimizer='adam')           # Оптимизатор Adam

    # Обучение модели (100 эпох, размер батча 32)
    history = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

    # Визуализация обучения
    print("\nВизуализация кривой обучения")
    model.plot_learning_curve(history)

    # Оценка модели
    print("\nОценка модели на тестовых данных")
    test_accuracy = model.evaluate(X_test, y_test)

    # Визуализация результатов
    print("\nВизуализация результатов")
    model.plot_predictions(X_test, y_test, feature_indices=(0, 1))"
    """

if __name__ == "__main__":
    main()
