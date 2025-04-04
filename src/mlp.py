import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class MLP:
    """
    Многослойный перцептрон (MLP) для классификации сейсмических фаций
    
    Параметры:
    layer_sizes - список размеров слоев (входной, скрытые, выходной)
    activation - функция активации ('leaky_relu' или 'relu')
    """
    def __init__(self, layer_sizes, activation='leaky_relu'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.parameters = {}
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Инициализация весов и смещений для всех слоев сети"""
        for i in range(1, len(self.layer_sizes)):
            # Инициализация весов методом He
            std = np.sqrt(2 / self.layer_sizes[i-1])  # Стандартное отклонение
            self.parameters[f'W{i}'] = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * std
            self.parameters[f'b{i}'] = np.zeros((1, self.layer_sizes[i]))
            
            # Инициализация параметров batch normalization для скрытых слоев
            if i < len(self.layer_sizes)-1:
                self.parameters[f'gamma{i}'] = np.ones((1, self.layer_sizes[i]))
                self.parameters[f'beta{i}'] = np.zeros((1, self.layer_sizes[i]))
                self.parameters[f'running_mean{i}'] = np.zeros((1, self.layer_sizes[i]))
                self.parameters[f'running_var{i}'] = np.ones((1, self.layer_sizes[i]))
    
    def forward(self, X, training=True, dropout_rate=0.3):
        """
        Прямой проход через сеть
        
        Параметры:
        X - входные данные
        training - флаг обучения (для dropout и batch norm)
        dropout_rate - вероятность отключения нейрона
        
        Возвращает:
        Выход сети и кэш промежуточных значений
        """
        cache = {'A0': X}  # Кэш для хранения промежуточных значений
        L = len([k for k in self.parameters if k.startswith('W')])  # Число слоев
        
        for l in range(1, L+1):
            # Линейное преобразование
            Z = cache[f'A{l-1}'].dot(self.parameters[f'W{l}']) + self.parameters[f'b{l}']
            cache[f'Z{l}'] = Z
            
            # Batch normalization и активация для скрытых слоев
            if l < L and f'gamma{l}' in self.parameters:
                if training:
                    # Вычисление статистик по батчу
                    mean, var = np.mean(Z, axis=0), np.var(Z, axis=0)
                    self.parameters[f'running_mean{l}'] = mean
                    self.parameters[f'running_var{l}'] = var
                else:
                    # Использование накопленных статистик
                    mean = self.parameters[f'running_mean{l}']
                    var = self.parameters[f'running_var{l}']
                
                # Нормализация
                Z_norm = (Z - mean) / np.sqrt(var + 1e-8)
                Z = self.parameters[f'gamma{l}'] * Z_norm + self.parameters[f'beta{l}']
                cache[f'Z_norm{l}'] = Z_norm
                
                # Dropout (только при обучении)
                if training and dropout_rate > 0:
                    mask = (np.random.rand(*Z.shape) > dropout_rate) / (1 - dropout_rate)
                    Z *= mask
                
                # Применение функции активации
                cache[f'A{l}'] = self.leaky_relu(Z) if self.activation == 'leaky_relu' else np.maximum(0, Z)
            else:
                # Выходной слой с softmax
                cache[f'A{L}'] = self.softmax(Z)
        
        return cache[f'A{L}'], cache
    
    def backward(self, X, y, cache, sample_weights=None, l2_lambda=0.001):
        """
        Обратный проход - вычисление градиентов
        
        Параметры:
        X - входные данные
        y - истинные метки
        cache - кэш из forward pass
        sample_weights - веса примеров
        l2_lambda - коэффициент L2-регуляризации
        
        Возвращает:
        Словарь градиентов для всех параметров
        """
        gradients = {}
        m = X.shape[0]  # Размер батча
        L = len([k for k in self.parameters if k.startswith('W')])  # Число слоев
        
        # Градиент на выходном слое (кросс-энтропия + softmax)
        dZL = cache[f'A{L}'].copy()
        dZL[range(m), y] -= 1
        
        # Взвешивание ошибок (для балансировки классов)
        if sample_weights is not None:
            dZL = (dZL.T * sample_weights).T / np.mean(sample_weights)
        else:
            dZL /= m
        
        # Градиенты для выходного слоя
        gradients[f'dW{L}'] = cache[f'A{L-1}'].T.dot(dZL) + l2_lambda*self.parameters[f'W{L}']
        gradients[f'db{L}'] = np.sum(dZL, axis=0, keepdims=True)
        
        # Обратное распространение по слоям
        for l in reversed(range(1, L)):
            # Градиент от следующего слоя
            dA = dZL.dot(self.parameters[f'W{l+1}'].T)
            
            # Градиенты для параметров batch norm
            if f'gamma{l}' in self.parameters:
                gradients[f'dgamma{l}'] = np.sum(dA * cache[f'Z_norm{l}'], axis=0, keepdims=True)
                gradients[f'dbeta{l}'] = np.sum(dA, axis=0, keepdims=True)
                dZ = dA * self.parameters[f'gamma{l}']
            else:
                dZ = dA
            
            # Градиент через функцию активации
            if self.activation == 'leaky_relu':
                dZ = dZ * self.leaky_relu_derivative(cache[f'Z{l}'])
            else:
                dZ = dZ * (cache[f'Z{l}'] > 0).astype(float)
            
            # Градиенты для весов и смещений
            gradients[f'dW{l}'] = cache[f'A{l-1}'].T.dot(dZ) + l2_lambda*self.parameters[f'W{l}']
            gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True)
            dZL = dZ  # Передача градиента следующему слою
        
        return gradients
    
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
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=500, batch_size=256, initial_lr=0.01, min_lr=0.0001,
              patience=50, verbose=True):
        """
        Обучение модели
        
        Параметры:
        X_train, y_train - обучающие данные
        X_val, y_val - валидационные данные (опционально)
        epochs - число эпох
        batch_size - размер батча
        initial_lr, min_lr - начальная и минимальная скорость обучения
        patience - число эпох для ранней остановки
        verbose - вывод прогресса
        """
        # Вычисление весов классов для балансировки
        class_weights_dict = self.compute_class_weights(y_train)
        sample_weights = np.array([class_weights_dict[y] for y in y_train])
        
        # Инициализация моментов для Adam
        m = {k: np.zeros_like(v) for k, v in self.parameters.items()}
        v = {k: np.zeros_like(v) for k, v in self.parameters.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Переменные для ранней остановки
        best_val_acc, wait = 0, 0
        best_params = None
        
        for epoch in range(epochs):
            # Уменьшение скорости обучения (cosine decay)
            lr = min_lr + 0.5*(initial_lr - min_lr)*(1 + np.cos(epoch/epochs*np.pi))
            
            # Перемешивание данных
            idx = np.random.permutation(len(X_train))
            
            # Обучение по мини-батчам
            for i in range(0, len(X_train), batch_size):
                batch_idx = idx[i:i+batch_size]
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                weights_batch = sample_weights[batch_idx]
                
                # Прямой и обратный проход
                AL, cache = self.forward(X_batch, training=True)
                grads = self.backward(X_batch, y_batch, cache, weights_batch)
                
                # Adam с gradient clipping
                total_norm = 0
                for grad in grads.values():
                    total_norm += np.sum(grad**2)
                total_norm = np.sqrt(total_norm)
                
                clip_coef = min(1.0, 1.0/(total_norm + 1e-6))
                
                # Обновление параметров с Adam
                for k in self.parameters:
                    if k.startswith(('W', 'b', 'gamma', 'beta')):
                        grads[f'd{k}'] *= clip_coef
                        m[k] = beta1*m[k] + (1-beta1)*grads[f'd{k}']
                        v[k] = beta2*v[k] + (1-beta2)*(grads[f'd{k}']**2)
                        m_hat = m[k]/(1 - beta1**(epoch+1))
                        v_hat = v[k]/(1 - beta2**(epoch+1))
                        self.parameters[k] -= lr * m_hat/(np.sqrt(v_hat) + eps)
            
            # Оценка на обучающей выборке
            train_pred = np.argmax(self.forward(X_train, training=False)[0], axis=1)
            train_acc = np.mean(train_pred == y_train)
            train_loss = self.cross_entropy_loss(y_train, self.forward(X_train, training=False)[0], sample_weights)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Оценка на валидационной выборке
            if X_val is not None:
                val_pred = np.argmax(self.forward(X_val, training=False)[0], axis=1)
                val_acc = np.mean(val_pred == y_val)
                val_loss = self.cross_entropy_loss(y_val, self.forward(X_val, training=False)[0])
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Проверка для ранней остановки
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {k: v.copy() for k, v in self.parameters.items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        self.parameters = best_params
                        if verbose:
                            print(f"\nРанняя остановка на эпохе {epoch+1}")
                        break
            
            # Вывод прогресса обучения
            if verbose and (epoch % 10 == 0 or epoch == epochs-1):
                msg = f"Эпоха {epoch+1}/{epochs}: LR={lr:.5f}, Train Loss={train_loss:.4f}, Acc={train_acc:.4f}"
                if X_val is not None:
                    msg += f", Val Loss={val_loss:.4f}, Acc={val_acc:.4f}"
                print(msg)
        
        return self.history
    
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
        self.print_classification_report(y_true, y_pred)

        # Визуализация матрицы ошибок
        self.plot_confusion_matrix(y_true, y_pred)
        
        return accuracy

    def confusion_matrix(self, y_true, y_pred, n_classes):
        """
        Матрица ошибок
        
        Параметры:
        y_true: истинные метки
        y_pred: предсказанные метки
        n_classes: количество классов фаций

        Возвращает:
        Матрицу ошибок
        """
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        return cm

    def print_classification_report(self, y_true, y_pred):
        """
        Отчет о классификации
        
        Параметры:
        y_true: истинные метки
        y_pred: предсказанные метки
        """
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        # Матрица ошибок
        cm = self.confusion_matrix(y_true, y_pred, n_classes)
        
        # Расчет метрик
        # Заголовок отчета
        print("{:10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
            "Класс", "Precision", "Recall", "F1-score", "Support"))

        # Расчет и вывод метрик для каждого класса
        for i in range(n_classes):
            tp = cm[i, i]               # True positives
            fp = np.sum(cm[:, i]) - tp  # False positives
            fn = np.sum(cm[i, :]) - tp  # False negatives
            
            # Расчет precision, recall, f1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = np.sum(y_true == i)
            
            # Вывод строки отчета
            print("{:10s} {:10.2f} {:10.2f} {:10.2f} {:10d}".format(
                str(i), precision, recall, f1, support))

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Визуализация матрицы ошибок
        
        Параметры:
        y_true: истинные метки
        y_pred: предсказанные метки
        """
        
        # Аннотации для классов
        class_names = ['Песчаник', 'Глина', 'Карбонат']
        n_classes = len(class_names)

        # Матрица ошибок
        cm = self.confusion_matrix(y_true, y_pred, n_classes)
                
        plt.figure(figsize=(8, 6))
        
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
    
    def cross_entropy_loss(self, y_true, y_pred, sample_weights=None, smoothing=0.1):
        """Функция потерь кросс-энтропии"""
        n_samples = y_true.shape[0]
        n_classes = y_pred.shape[1]
        
        # Преобразуем y_true в one-hot encoding
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(n_samples), y_true] = 1
        
        # Применяем label smoothing
        confidence = 1.0 - smoothing
        smooth_targets = y_true_onehot * confidence + (1 - y_true_onehot) * smoothing / (n_classes - 1)
        
        # Вычисляем логарифмы с защитой от log(0)
        log_probs = np.log(np.clip(y_pred, 1e-10, 1.0))
        
        # Вычисляем потери
        loss = -np.sum(smooth_targets * log_probs, axis=1)
        
        if sample_weights is not None:
            loss *= sample_weights
            
        return np.mean(loss)
    
    def one_hot_encode(self, y):
        """Преобразование меток классов в one-hot кодировку"""
        one_hot = np.zeros((y.size, self.output_size))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def leaky_relu(self, x, alpha=0.01):
        """Функция активации Leaky ReLU"""
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        """Производная функции Leaky ReLU"""
        return np.where(x > 0, 1, alpha)

    def compute_class_weights(self, y):
        """Вычисление весов классов для балансировки"""
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        
        weights = n_samples / (n_classes * counts)
        return {c: w for c, w in zip(classes, weights)}
