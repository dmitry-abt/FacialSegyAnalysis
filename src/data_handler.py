import numpy as np

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
        X = np.zeros((self.n_samples, self.n_features))
        y = np.zeros(self.n_samples, dtype=int)
        
        # Класс 0 (песчаник?)
        n0 = self.n_samples // 3
        X[:n0, 0] = np.random.normal(0.8, 0.1, n0)
        X[:n0, 1] = np.random.normal(0.3, 0.2, n0)
        y[:n0] = 0
        
        # Класс 1 (глина?)
        n1 = self.n_samples // 3
        X[n0:n0+n1, 0] = np.random.normal(0.3, 0.1, n1)
        X[n0:n0+n1, 1] = np.random.normal(0.7, 0.15, n1)
        y[n0:n0+n1] = 1
        
        # Класс 2 (карбонат?)
        n2 = self.n_samples - n0 - n1
        X[n0+n1:, 0] = np.random.normal(0.5, 0.2, n2)
        X[n0+n1:, 1] = np.random.normal(0.5, 0.2, n2)
        y[n0+n1:] = 2
        
        # Генерация дополнительных атрибутов как комбинаций основных
        for i in range(3, self.n_features):
            X[:, i] = np.random.rand(self.n_samples) * 0.5 + X[:, i-3] * 0.5

        # Нормализация данных к диапазону [0, 1]
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        
        # Добавление шума
        X += np.random.normal(0, 0.05, X.shape)
        
        return X, y
