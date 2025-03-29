from data_handler import DataHandler

def main():
    # Генерация данных
    print("Генерация синтетических данных")
    data_handler = DataHandler(n_samples=1500, n_features=3, n_classes=3)
    X, y = data_handler.generate_synthetic_data()

    # Визуализация данных
    print("\nВизуализация данных до классификации")
    feature_names = ['Амплитуда', 'Частота', 'Анизотропия']
    data_handler.plot_features(X, y, feature_names)
    
    # Разделение данных
    print("\nРазделение данных на обучающую и тестовую выборки")
    
    # Создание и обучение модели
    print("\nСоздание и обучение нейронной сети")

    # Визуализация обучения
    print("\nВизуализация кривой обучения")
    
    # Оценка модели
    print("\nОценка модели на тестовых данных")
    
    # Визуализация результатов
    print("\nВизуализация результатов")

if __name__ == "__main__":
    main()
