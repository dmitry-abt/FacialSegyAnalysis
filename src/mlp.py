import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class MLP:
    """
    Многослойный перцептрон (MLP) для классификации сейсмических фаций
    
    Параметры:
    input_size: размер входного слоя (количество атрибутов)
    hidden_sizes: список с размерами скрытых слоев
    output_size: размер выходного слоя (количество классов)
    activation: функция активации ('relu', 'sigmoid', 'tanh')
    learning_rate: скорость обучения
    weight_init: метод инициализации весов ('xavier', 'he')
    optimizer: алгоритм оптимизации ('sgd', 'adam', 'rmsprop')
    beta1, beta2: параметры для оптимизатора Adam
    epsilon: малая константа для численной стабильности
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', learning_rate=0.01, 
                 weight_init='he', optimizer='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Инициализация весов и смещений для всех слоев
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes)-1):
            # Выбор метода инициализации весов
            if weight_init == 'xavier':
                # Инициализация Xavier (хороша для сигмоидных активаций)
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            elif weight_init == 'he':
                # Инициализация He (хороша для ReLU)
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = 0.01  # Простая инициализация малыми случайными значениями
                
            # Инициализация весов и смещений
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            bias = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Инициализация параметров для оптимизаторов
        if optimizer in ['adam', 'rmsprop']:
            # Моменты для Adam и RMSprop
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
        elif optimizer == 'momentum':
            # Моменты для SGD с моментом
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
    
    def forward(self, X):
        """
        Прямое распространение сигнала через сеть
        
        Параметры:
        X: входные данные (размер batch_size * input_size)
        
        Возвращает:
        Выход сети (вероятности классов)
        """
        self.layer_inputs = []      # Сохраняем входы каждого слоя для обратного распространения
        self.layer_outputs = []     # Сохраняем выходы каждого слоя
        current_output = X
        
        # Проход через скрытые слои
        for i, (weight, bias) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            self.layer_inputs.append(current_output)
            
            # Линейное преобразование
            z = np.dot(current_output, weight) + bias
            
            # Применение функции активации
            if self.activation == 'relu':
                current_output = self.relu(z)
            elif self.activation == 'sigmoid':
                current_output = self.sigmoid(z)
            elif self.activation == 'tanh':
                current_output = self.tanh(z)
                
            self.layer_outputs.append(current_output)
        
        # Выходной слой с softmax
        self.layer_inputs.append(current_output)
        z = np.dot(current_output, self.weights[-1]) + self.biases[-1]
        current_output = self.softmax(z)
        self.layer_outputs.append(current_output)
        
        return current_output
    
    def backward(self, X, y_true, y_pred):
        """
        Обратное распространение ошибки и вычисление градиентов
        
        Параметры:
        X: входные данные
        y_true: истинные метки в one-hot кодировке
        y_pred: предсказанные вероятности классов
        
        Возвращает:
        grads_w: градиенты по весам
        grads_b: градиенты по смещениям
        """
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        
        # Градиент для выходного слоя (кросс-энтропия + softmax)
        error = y_pred - y_true
        
        # Проход через слои в обратном порядке (от выходного к входному)
        for i in range(len(self.weights)-1, -1, -1):
            # Градиент по весам текущего слоя
            grads_w[i] = np.dot(self.layer_inputs[i].T, error)
            
            # Градиент по смещениям текущего слоя
            grads_b[i] = np.sum(error, axis=0, keepdims=True)
            
            # Если не первый слой, вычисляем градиент для предыдущего слоя
            if i > 0:
                # Градиент по выходу предыдущего слоя
                error = np.dot(error, self.weights[i].T)
                
                # Градиент через функцию активации предыдущего слоя
                if self.activation == 'relu':
                    error *= self.relu_derivative(self.layer_outputs[i-1])
                elif self.activation == 'sigmoid':
                    error *= self.sigmoid_derivative(self.layer_outputs[i-1])
                elif self.activation == 'tanh':
                    error *= self.tanh_derivative(self.layer_outputs[i-1])
        
        return grads_w, grads_b
    
    def update_weights(self, grads_w, grads_b, t):
        """
        Обновление весов сети с использованием выбранного оптимизатора
        
        Параметры:
        grads_w: градиенты по весам
        grads_b: градиенты по смещениям
        t: номер шага (используется для Adam)
        """
        if self.optimizer == 'sgd':
            # Обычный градиентный спуск
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grads_w[i]
                self.biases[i] -= self.learning_rate * grads_b[i]
                
        elif self.optimizer == 'momentum':
            # SGD с моментом
            gamma = 0.9  # коэффициент момента
            for i in range(len(self.weights)):
                # Обновление момента
                self.v_weights[i] = gamma * self.v_weights[i] + self.learning_rate * grads_w[i]
                self.v_biases[i] = gamma * self.v_biases[i] + self.learning_rate * grads_b[i]
                
                # Обновление параметров
                self.weights[i] -= self.v_weights[i]
                self.biases[i] -= self.v_biases[i]
                
        elif self.optimizer == 'rmsprop':
            # RMSprop оптимизатор
            beta = 0.9  # Коэффициент для скользящего среднего
            for i in range(len(self.weights)):
                # Обновление скользящего среднего квадратов градиентов
                self.v_weights[i] = beta * self.v_weights[i] + (1 - beta) * (grads_w[i]**2)
                self.v_biases[i] = beta * self.v_biases[i] + (1 - beta) * (grads_b[i]**2)
                
                # Обновление параметров с адаптивным learning rate
                self.weights[i] -= self.learning_rate * grads_w[i] / (np.sqrt(self.v_weights[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * grads_b[i] / (np.sqrt(self.v_biases[i]) + self.epsilon)
                
        elif self.optimizer == 'adam':
            # Adam оптимизатор
            for i in range(len(self.weights)):
                # Обновление первых моментов (средние градиентов)
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * grads_w[i]
                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * grads_b[i]
                
                # Обновление вторых моментов (нецентрированные дисперсии)
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (grads_w[i]**2)
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (grads_b[i]**2)
                
                # Коррекция смещения (bias correction)
                m_hat_w = self.m_weights[i] / (1 - self.beta1**t)
                m_hat_b = self.m_biases[i] / (1 - self.beta1**t)
                
                v_hat_w = self.v_weights[i] / (1 - self.beta2**t)
                v_hat_b = self.v_biases[i] / (1 - self.beta2**t)
                
                # Обновление параметров
                self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=True):
        """
        Обучение нейронной сети
        
        Параметры:
        X_train: обучающие данные
        y_train: метки обучающих данных
        X_val: валидационные данные (опционально)
        y_val: метки валидационных данных (опционально)
        epochs: количество эпох обучения
        batch_size: размер мини-батча
        verbose: вывод информации о процессе обучения
        
        Возвращает:
        history: словарь с историей обучения (потери и точность)
        """
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Преобразование меток в one-hot кодировку
        y_train_onehot = self.one_hot_encode(y_train)
        if y_val is not None:
            y_val_onehot = self.one_hot_encode(y_val)
        
        # Цикл обучения по эпохам
        for epoch in range(1, epochs+1):
            # Перемешивание данных в каждой эпохе
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            epoch_correct = 0
            
            # Обучение по мини-батчам
            for i in range(0, n_samples, batch_size):
                # Получение текущего мини-батча
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Прямое распространение
                y_pred = self.forward(X_batch)
                
                # Вычисление потерь и точности
                loss = self.cross_entropy_loss(y_batch, y_pred)
                epoch_loss += loss * X_batch.shape[0]
                
                # Подсчет правильных предсказаний
                predictions = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(predictions == true_labels)
                
                # Обратное распространение и обновление весов
                grads_w, grads_b = self.backward(X_batch, y_batch, y_pred)
                self.update_weights(grads_w, grads_b, epoch)
            
            # Вычисление средних потерь и точности для эпохи
            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Валидация (если есть валидационные данные)
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(y_val_onehot, val_pred)
                val_pred_labels = np.argmax(val_pred, axis=1)
                val_acc = np.mean(val_pred_labels == y_val)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            # Вывод информации о процессе обучения
            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
                msg = f"Эпоха {epoch}/{epochs} - потери: {train_loss:.4f} - точность: {train_acc:.4f}"
                if X_val is not None and y_val is not None:
                    msg += f" - вал. потери: {val_loss:.4f} - вал. точность: {val_acc:.4f}"
                print(msg)
        
        return history
    
    def predict(self, X):
        """
        Предсказание классов для входных данных
        
        Параметры:
        X: входные данные
        
        Возвращает:
        Предсказанные классы
        """
        proba = self.forward(X)
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X, y_true):
        """
        Оценка качества модели на тестовых данных
        
        Параметры:
        X: тестовые данные
        y_true: истинные метки тестовых данных
        
        Возвращает:
        Точность на тестовых данных
        """
        # Получение предсказаний
        y_pred = self.predict(X)
        
        # Вычисление точности
        accuracy = np.mean(y_pred == y_true)
        print(f"\nТочность на тестовых данных: {accuracy:.4f}")
        
        # Отчет о классификации
        print("\nОтчет о классификации:")
        print(classification_report(y_true, y_pred, target_names=['Песчаник', 'Глина', 'Карбонат']))
        
        # Визуализация матрицы ошибок
        self.plot_confusion_matrix(y_true, y_pred)
        
        return accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Визуализация матрицы ошибок
        
        Параметры:
        y_true: истинные метки
        y_pred: предсказанные метки
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        # Аннотации для классов
        class_names = ['Песчаник', 'Глина', 'Карбонат']
        
        # Построение тепловой карты матрицы ошибок
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Матрица ошибок классификации')
        plt.xlabel('Предсказанные классы')
        plt.ylabel('Истинные классы')
        plt.show()
    
    def plot_learning_curve(self, history):
        """
        Визуализация кривой обучения
        
        Параметры:
        history: словарь с историей обучения
        """
        plt.figure(figsize=(12, 5))
        
        # График потерь
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Обучающая выборка')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Валидационная выборка')
        plt.title('Функция потерь по эпохам')
        plt.xlabel('Эпохи')
        plt.ylabel('Потери')
        plt.legend()
        
        # График точности
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Обучающая выборка')
        if 'val_acc' in history:
            plt.plot(history['val_acc'], label='Валидационная выборка')
        plt.title('Точность по эпохам')
        plt.xlabel('Эпохи')
        plt.ylabel('Точность')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, X, y_true, feature_indices=(0, 1)):
        """
        Визуализация результатов модели
        
        Параметры:
        X: входные данные
        y_true: истинные метки
        feature_indices: индексы признаков для визуализации
        """
        # Получение результатов
        y_pred = self.predict(X)
        
        plt.figure(figsize=(12, 6))
        
        # Визуализация истинных классов
        plt.scatter(X[:, feature_indices[0]], X[:, feature_indices[1]], 
                    c=y_true, cmap='viridis', alpha=0.6, label='Истинные классы')
        
        # Визуализация предсказанных классов (крестиками)
        plt.scatter(X[:, feature_indices[0]], X[:, feature_indices[1]], 
                    c=y_pred, cmap='viridis', alpha=0.6, marker='x', label='Предсказанные классы')
        
        plt.title('Сравнение истинных и предсказанных классов фаций')
        plt.xlabel(f'Атрибут {feature_indices[0]+1}')
        plt.ylabel(f'Атрибут {feature_indices[1]+1}')
        plt.legend()
        plt.show()

    # Функции активации и их производные
    
    def relu(self, x):
        """Функция активации ReLU"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Производная функции ReLU"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Производная сигмоидной функции"""
        return x * (1 - x)
    
    def tanh(self, x):
        """Функция активации гиперболический тангенс"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Производная функции tanh"""
        return 1 - x**2
    
    def softmax(self, x):
        """Функция активации softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        """Функция потерь кросс-энтропии"""
        epsilon = 1e-12  # Малая константа для избежания log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Ограничение значений
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def one_hot_encode(self, y):
        """Преобразование меток классов в one-hot кодировку"""
        one_hot = np.zeros((y.size, self.output_size))
        one_hot[np.arange(y.size), y] = 1
        return one_hot
