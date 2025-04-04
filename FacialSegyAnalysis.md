**Тестовое задание: Реализация фациальной классификации на основе сейсмических данных**

### Описание задания

Необходимо реализовать метод классификации сейсмических фаций на основе предоставленных методик.
Можно выбрать один из трёх методов:

1. **Нейронная сеть обратного распространения (реализовать с нуля, без использования PyTorch/TensorFlow)**
2. **K-средняя классификация (реализовать с нуля, без использования Scikit-learn)**
3. **Байесовская классификация (реализовать с нуля, без использования Scikit-learn)**

Данные для тестирования отсутствуют, их необходимо синтезировать самостоятельно.

### Требования к реализации

- **Язык программирования:** Python (рекомендуется, но допускаются альтернативные варианты).
- **Библиотеки:** NumPy, SciPy, Pandas, Matplotlib, ObsPy, Segyio.
- **Входные данные:**
  - Синтетические сейсмические атрибуты и данные вдоль траектории скважины.
  - Можно использовать публичный набор данных, например, **Madagascar** (https://wiki.seg.org/wiki/Kerry-3D).
- **Выходные данные:**
  - Классифицированные фации, визуализация результатов, метрики качества (точность, полнота, F1-score и др.).

### Пример загрузки сейсмических данных в формате SEG-Y

```python
import segyio
import numpy as np
import matplotlib.pyplot as plt

# Открываем SEG-Y файл
file_path = "example.sgy"
with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
    seismic_data = segyio.tools.cube(segyfile)  # Загружаем данные в массив

# Визуализируем один из разрезов
plt.imshow(seismic_data[:, :, 0], cmap='gray', aspect='auto')
plt.colorbar(label='Амплитуда')
plt.title('Сейсмический разрез')
plt.show()
```

### Описание методов

#### **1. Нейронная сеть обратного распространения (без PyTorch/TensorFlow)**

- Реализовать многослойный перцептрон (MLP) с нуля на NumPy.
- Используемая функция ошибки: Cross-Entropy Loss (для многоклассовой классификации) или MSE.
- Оптимизация: градиентный спуск (SGD, Adam, RMSprop - реализовать вручную).
- Выход: метки классов сейсмических фаций, визуализация обучения.

#### **2. K-средняя классификация (без Scikit-learn)**

- Реализовать алгоритм K-средних с нуля.
- Данные обучения используются для расчёта центров кластеров.
- Определение оптимального количества кластеров (Elbow Method, Silhouette Score - реализовать вручную).
- Итерационный процесс объединения точек в кластеры на основе Евклидова расстояния.
- Выход: распределение точек по кластерам, визуализация результатов.

#### **3. Байесовская классификация (без Scikit-learn)**

- Реализовать байесовский классификатор с нуля.
- Обучение на основе расчёта средних значений и стандартных отклонений каждого класса.
- Использование вероятностной модели для предсказания классов.
- Определение качественной связи между параметрами резервуара и активными сейсмическими атрибутами.
- Выход: вероятность принадлежности точки к определённому классу, визуализация вероятностного распределения.

### Ожидаемые результаты

- Генерация синтетических данных (сейсмический разрез, атрибуты, скважины).
- Реализация одного из предложенных методов классификации.
- Визуализация данных до и после классификации.
- Оценка качества классификации.

### Критерии оценки

1. **Корректность реализации** — точность работы выбранного метода.
2. **Генерация синтетических данных** — реалистичность и обоснованность модели.
3. **Чистота и читаемость кода** — следование принципам хорошего кода.
4. **Документирование** — комментарии и описание шагов выполнения.
5. **Визуализация** — информативность представления результатов.

**Дополнительная информация**

- Возможность расширения задания за счёт использования ансамблевых методов.
- Интерактивные графики (Plotly, Seaborn) будут плюсом.
- Возможность импорта данных из внешних источников (например, файлов SEG-Y) — приветствуется.

**Формат сдачи:**

- Ссылка на github репозиторий с кодом + README с инструкцией по запуску.
- Примеры входных и выходных данных.
- Краткий отчёт (PDF/Jupyter Notebook) с объяснением реализованного метода.

