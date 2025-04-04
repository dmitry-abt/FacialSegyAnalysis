import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import segyio
from tqdm import tqdm

class DataHandler:
    """Основной класс для работы с данными: генерация, загрузка, обработка и визуализация
    
    Параметры:
    n_samples: количество образцов данных
    n_features: количество сейсмических атрибутов
    n_classes: количество классов фаций
    random_state: seed для воспроизводимости результатов"""
    
    def __init__(self, n_samples=1000, n_features=5, n_classes=3, random_state=42):
        """Инициализация"""
        self.n_samples = n_samples          # Количество образцов данных
        self.n_features = n_features        # Количество признаков (атрибутов)
        self.n_classes = n_classes          # Количество классов фаций
        self.random_state = random_state    # Seed для воспроизводимости
        np.random.seed(random_state)
    
    def generate_synthetic_data(self):
        """Генерация синтетических сейсмических данных с характеристиками разных фаций"""
        X = np.zeros((self.n_samples, self.n_features))  # Матрица признаков
        y = np.zeros(self.n_samples, dtype=int)          # Вектор меток классов
        
        # Генерация данных для каждого класса фаций
        
        # Класс 0 (песчаник) - характерные значения атрибутов
        n0 = self.n_samples // 3                    # Количество образцов для класса 0
        X[:n0, 0] = np.random.normal(0.8, 0.1, n0)  # Высокий амплитудный отклик
        X[:n0, 1] = np.random.normal(0.3, 0.2, n0)  # Низкая частота
        X[:n0, 2] = np.random.normal(0.5, 0.15, n0) # Средняя анизотропия
        y[:n0] = 0                                  # Метки класса 0
        
        # Класс 1 (глина) - характерные значения атрибутов
        n1 = self.n_samples // 3                         # Количество образцов для класса 1
        X[n0:n0+n1, 0] = np.random.normal(0.3, 0.1, n1)  # Низкий амплитудный отклик
        X[n0:n0+n1, 1] = np.random.normal(0.7, 0.15, n1) # Высокая частота
        X[n0:n0+n1, 2] = np.random.normal(0.2, 0.1, n1)  # Низкая анизотропия
        y[n0:n0+n1] = 1                                  # Метки класса 1
        
        # Класс 2 (карбонат) - характерные значения атрибутов
        n2 = self.n_samples - n0 - n1                  # Оставшиеся образцы
        X[n0+n1:, 0] = np.random.normal(0.5, 0.2, n2)  # Средний амплитудный отклик
        X[n0+n1:, 1] = np.random.normal(0.5, 0.2, n2)  # Средняя частота
        X[n0+n1:, 2] = np.random.normal(0.8, 0.1, n2)  # Высокая анизотропия
        y[n0+n1:] = 2                                  # Метки класса 2
        
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

    def train_test_split(self, X, y, test_size=0.2, stratify=None, random_state=None):
        """
        Разделение данных на обучающую и тестовую выборки
        
        Параметры:
        X: входные данные
        y: метки
        test_size: доля тестовых данных  
        stratify: если не None, сохраняет распределение классов
        random_state: seed для воспроизводимости
        
        Возвращает:
        X_train, X_test, y_train, y_test: разделенные данные
        """
        if random_state is not None:
            np.random.seed(random_state)
    
        n_samples = len(X)
        n_test = int(n_samples * test_size)  # Количество тестовых образцов
        
        if stratify is not None:
            # Стратифицированное разбиение (сохранение пропорций классов)
            classes, class_counts = np.unique(stratify, return_counts=True)
            test_counts = (class_counts * test_size).astype(int)  # Количество тестовых образцов для каждого класса
            
            test_indices = []   # Индексы тестовых данных
            train_indices = []  # Индексы обучающих данных
            
            for cls, test_count in zip(classes, test_counts):
                cls_indices = np.where(y == cls)[0] # Индексы текущего класса
                np.random.shuffle(cls_indices)
                
                # Разделяем на тестовые и обучающие
                test_indices.extend(cls_indices[:test_count])
                train_indices.extend(cls_indices[test_count:])
            
            # Перемешиваем итоговые списки индексов
            np.random.shuffle(test_indices)
            np.random.shuffle(train_indices)
        else:
            # Случайное разбиение
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
        
        # Разделение данных
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test

    def load_segy_data(self, filepath, inline_range=None, xline_range=None, sample_range=None):
        """Загрузка 3D сейсмических данных из SEG-Y"""
        with segyio.open(filepath, "r", strict=False) as segyfile:
            # Устанавливаем диапазоны по умолчанию, если не заданы
            inline_range = inline_range or (0, len(segyfile.ilines), 1)
            xline_range = xline_range or (0, len(segyfile.xlines), 1)
            sample_range = sample_range or (0, len(segyfile.samples), 1)
            
            # Инициализация массива для данных
            data = np.zeros((len(range(*inline_range)),
                            len(range(*xline_range)),
                            len(range(*sample_range))))
            
            # Чтение данных с прогресс-баром
            for i, iline in tqdm(enumerate(range(*inline_range)), desc="Чтение инлайнов"):
                for j, xline in enumerate(range(*xline_range)):
                    # Получаем индекс трассы
                    trace_idx = segyfile.xlines[xline] + segyfile.ilines[iline]
                    # Записываем данные трассы в массив
                    data[i, j, :] = segyfile.trace[trace_idx][slice(*sample_range)]
            
            return data

    def enhanced_attributes(self, data, window_size=9):
        """Расширенные сейсмические атрибуты с исправленным расчетом градиента"""
        n_inlines, n_xlines, n_samples = data.shape
        n_attributes = 10
        attributes = np.zeros((n_inlines, n_xlines, n_samples, n_attributes))
        
        # Предварительно вычисляем градиент по всему объему для скорости
        grad_z = np.gradient(data, axis=2)  # Градиент по вертикальной оси (времени/глубине)
        
        # Расчет атрибутов для каждой точки с прогресс-баром
        for i in tqdm(range(n_inlines), desc="Расчет атрибутов"):
            for j in range(n_xlines):
                for k in range(window_size//2, n_samples-window_size//2):
                    window = data[i, j, k-window_size//2:k+window_size//2+1]  # Окно данных
                    
                    # 1. Средняя амплитуда
                    attributes[i,j,k,0] = np.mean(np.abs(window))
                    
                    # 2. Энергия
                    attributes[i,j,k,1] = np.sum(window**2)
                    
                    # 3. Спектральный центроид
                    fft = np.abs(np.fft.rfft(window))
                    freq = np.fft.rfftfreq(len(window))
                    if np.sum(fft) > 0:
                        attributes[i,j,k,2] = np.sum(freq * fft) / np.sum(fft)
                    
                    # 4. Огибающая (Гильберт)
                    analytic = np.fft.ifft(np.fft.fft(window) * 2)
                    analytic[len(window)//2+1:] = 0
                    attributes[i,j,k,3] = np.mean(np.abs(analytic))
                    
                    # 5. Мгновенная фаза
                    attributes[i,j,k,4] = np.mean(np.angle(analytic))
                    
                    # 6. Стандартное отклонение
                    attributes[i,j,k,5] = np.std(window)
                    
                    # 7. Медиана
                    attributes[i,j,k,6] = np.median(window)
                    
                    # 8. Градиент (первая производная)
                    attributes[i,j,k,7] = np.mean(np.diff(window))
                    
                    # 9. Вертикальный градиент (исправленная версия)
                    attributes[i,j,k,8] = grad_z[i,j,k]  # Используем предварительно вычисленный градиент
                    
                    # 10. Пороговая энергия
                    attributes[i,j,k,9] = np.mean(window > np.mean(window))
        
        return attributes

    def load_facies(self, shape, well_locations, n_classes=4):
        """Генерация синтетических фаций на основе расположения скважин"""
        facies = np.full(shape, -1, dtype=int)  # -1 означает отсутствие данных
        for iline, xline in well_locations:
            # Случайная глубина для скважины
            depth = np.random.randint(shape[2]//3, shape[2])
            # Случайное распределение фаций по глубине
            facies[iline, xline, :depth] = np.random.randint(0, n_classes, depth)
        return facies

    def plot_section(self, data, iline, title, cmap='seismic', vmin=None, vmax=None):
        """2D визуализация вертикального разреза"""
        plt.figure(figsize=(15, 6))
        plt.imshow(data[iline, :, :].T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Crossline")
        plt.ylabel("Time/Depth")
        plt.show()

    def plot_3d_seismic(self, data, iline_step=5, xline_step=5, sample_step=5, threshold=0.5):
        """3D визуализация сейсмического куба"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Нормализация для цветовой карты
        norm = mcolors.Normalize(vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))
        
        # Создаем сетку для 3D точек
        ilines = range(0, data.shape[0], iline_step)
        xlines = range(0, data.shape[1], xline_step)
        samples = range(0, data.shape[2], sample_step)
        
        x, y, z = np.meshgrid(ilines, xlines, samples, indexing='ij')
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = data[::iline_step, ::xline_step, ::sample_step].flatten()
        
        mask = np.abs(values) > threshold * np.max(np.abs(data))
        # 3D scatter plot с цветовой кодировкой амплитуд
        sc = ax.scatter(x[mask], y[mask], z[mask], c=values[mask], cmap='seismic', norm=norm, alpha=0.3, s=1)
        
        plt.colorbar(sc, label='Амплитуда')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.set_zlabel('Время/Глубина')
        ax.set_title('3D визуализация сейсмических данных')
        plt.tight_layout()
        plt.show()

    def plot_3d_facies(self, facies, iline_step=3, xline_step=3, sample_step=3):
        """3D визуализация фаций"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Цветовая карта для фаций
        colors = ['black', 'red', 'green', 'blue', 'yellow']
        cmap = mcolors.ListedColormap(colors[:len(np.unique(facies))])
        
        # Создаем сетку для 3D точек
        ilines = range(0, facies.shape[0], iline_step)
        xlines = range(0, facies.shape[1], xline_step)
        samples = range(0, facies.shape[2], sample_step)
        
        x, y, z = np.meshgrid(ilines, xlines, samples, indexing='ij')
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        f = facies[::iline_step, ::xline_step, ::sample_step].flatten()
        
        # Отображаем только точки с определенными фациями (не -1)
        mask = f != -1
        sc = ax.scatter(x[mask], y[mask], z[mask], c=f[mask], cmap=cmap, alpha=0.5, s=2)
        
        cbar = plt.colorbar(sc, ticks=np.unique(f[mask]))
        cbar.ax.set_yticklabels([f'Facies {i}' for i in np.unique(f[mask])])
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.set_zlabel('Время/Глубина')
        ax.set_title('3D визуализация фаций')
        plt.tight_layout()
        plt.show()

    def plot_history(self, history):
        """Графики обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График функции потерь
        ax1.plot(history['train_loss'], label='Train')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation')
        ax1.set_title('Loss')
        ax1.legend()
        
        # График точности
        ax2.plot(history['train_acc'], label='Train')
        if 'val_acc' in history:
            ax2.plot(history['val_acc'], label='Validation')
        ax2.set_title('Accuracy')
        ax2.legend()
        
        plt.show()

    def simple_oversample(self, X, y):
        """Увеличение выборки (oversampling) для балансировки классов"""
        classes = np.unique(y)                          # Уникальные классы
        class_counts = [sum(y == c) for c in classes]   # Количество образцов каждого класса
        max_count = max(class_counts)                   # Максимальное количество в классе
        
        X_resampled = []  # Список для новых признаков
        y_resampled = []  # Список для новых меток
        
        for c in classes:
            # Индексы текущего класса
            idx = np.where(y == c)[0]
            # Сколько нужно добавить примеров
            need = max_count - len(idx)
            # Случайно выбираем существующие примеры для дублирования
            dup_idx = np.random.choice(idx, size=need)
            
            # Добавляем оригинальные и дублированные примеры
            X_resampled.append(X[idx])
            X_resampled.append(X[dup_idx])
            y_resampled.append(y[idx])
            y_resampled.append(y[dup_idx])
        
        return np.concatenate(X_resampled), np.concatenate(y_resampled)

class RobustScaler:
    """Масштабирование данных устойчивое к выбросам"""
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
    
    def fit(self, X):
        """Вычисление параметров масштабирования"""
        self.center_ = np.median(X, axis=0)  # Медиана по каждому признаку
        
        # Вычисляем интерквартильный размах (IQR) как масштаб
        q_min, q_max = np.percentile(X, self.quantile_range, axis=0)
        self.scale_ = q_max - q_min
        
        # Защита от нулевого масштаба
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """Применение масштабирования к данным"""
        return (X - self.center_) / self.scale_
    
    def fit_transform(self, X):
        """Вычисление параметров и масштабирование за один шаг"""
        return self.fit(X).transform(X)

class PCA:
    """Анализ главных компонент с поддержкой n_components в долях"""
    def __init__(self, n_components=None):
        self.n_components = n_components        # Количество компонент или доля дисперсии
        self.components_ = None                 # Главные компоненты
        self.mean_ = None                       # Средние значения
        self.explained_variance_ratio_ = None   # Доля объясненной дисперсии
    
    def fit(self, X):
        """Обучение PCA: вычисление главных компонент"""
        # Центрирование данных
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Сингулярное разложение
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Объясненная дисперсия
        explained_variance = (s ** 2) / (X.shape[0] - 1)
        total_variance = explained_variance.sum()
        self.explained_variance_ratio_ = explained_variance / total_variance
        
        # Автоматический выбор числа компонент
        if isinstance(self.n_components, float):
            # Если задана доля дисперсии, находим минимальное число компонент
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = np.argmax(cumulative_variance >= self.n_components) + 1
        else:
            # Иначе используем заданное число или все компоненты
            self.n_components_ = self.n_components or X.shape[1]
        
        # Сохраняем главные компоненты
        self.components_ = Vt[:self.n_components_]
        return self
    
    def transform(self, X):
        """Преобразование данных в пространство главных компонент"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """Обучение и преобразование за один шаг"""
        self.fit(X)
        return self.transform(X)
