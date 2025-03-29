import numpy as np
import matplotlib.pyplot as plt

class DataHandler:
    """Класс для генерации данных
    
    Параметры:
    n_samples: количество образцов данных
    n_features: количество сейсмических атрибутов
    n_classes: количество классов фаций
    random_state: seed для воспроизводимости результатов"""
    
    def __init__(self, n_samples=1000, n_features=5, n_classes=3, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_synthetic_data(self):
        """Генерация синтетических данных"""
        # Инициализация массивов для данных и меток
        X = np.zeros((self.n_samples, self.n_features))
        y = np.zeros(self.n_samples, dtype=int)
        
        # Генерация данных для каждого класса фаций
        
        # Класс 0 (песчаник) - характерные значения атрибутов
        n0 = self.n_samples // 3
        X[:n0, 0] = np.random.normal(0.8, 0.1, n0)  # Высокий амплитудный отклик
        X[:n0, 1] = np.random.normal(0.3, 0.2, n0)  # Низкая частота
        X[:n0, 2] = np.random.normal(0.5, 0.15, n0) # Средняя анизотропия
        y[:n0] = 0
        
        # Класс 1 (глина) - характерные значения атрибутов
        n1 = self.n_samples // 3
        X[n0:n0+n1, 0] = np.random.normal(0.3, 0.1, n1)  # Низкий амплитудный отклик
        X[n0:n0+n1, 1] = np.random.normal(0.7, 0.15, n1) # Высокая частота
        X[n0:n0+n1, 2] = np.random.normal(0.2, 0.1, n1)  # Низкая анизотропия
        y[n0:n0+n1] = 1
        
        # Класс 2 (карбонат) - характерные значения атрибутов
        n2 = self.n_samples - n0 - n1
        X[n0+n1:, 0] = np.random.normal(0.5, 0.2, n2)  # Средний амплитудный отклик
        X[n0+n1:, 1] = np.random.normal(0.5, 0.2, n2)  # Средняя частота
        X[n0+n1:, 2] = np.random.normal(0.8, 0.1, n2) # Высокая анизотропия
        y[n0+n1:] = 2
        
        # Генерация дополнительных атрибутов как комбинаций основных
        for i in range(3, self.n_features):
            X[:, i] = np.random.rand(self.n_samples) * 0.5 + X[:, i-3] * 0.5

        # Нормализация данных к диапазону [0, 1]
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        
        # Добавление шума
        X += np.random.normal(0, 0.05, X.shape)
        
        return X, y
    
    def plot_features(self, X, y, feature_names=None):
        """
        Визуализация распределения сейсмических атрибутов по классам фаций
        
        Параметры:
        X: массив сейсмических атрибутов
        y: метки классов фаций
        feature_names: названия атрибутов
        """
        if feature_names is None:
            feature_names = [f'Атрибут {i+1}' for i in range(X.shape[1])]
            
        plt.figure(figsize=(15, 10))
        
        # Построение гистограмм распределения для каждого атрибута
        for i in range(X.shape[1]):
            plt.subplot(2, 3, i+1)
            for c in range(self.n_classes):
                # Гистограмма значений атрибута для каждого класса
                plt.hist(X[y == c, i], bins=20, alpha=0.5, label=f'Класс {c}')
            plt.title(feature_names[i])
            plt.xlabel('Значение атрибута')
            plt.ylabel('Частота')
            plt.legend()
        
        plt.suptitle('Распределение сейсмических атрибутов по классам фаций', y=1.02)
        plt.tight_layout()
        plt.show()
