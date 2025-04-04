from matplotlib import pyplot as plt
from data_handler import DataHandler
#from mlp import MLP
import numpy as np
import matplotlib.colors as mcolors

def main():
    # Конфигурация
    SEGY_PATH = "/Users/developer/Downloads/adele_seismic_survey_NW_australia_bind_cube_fault_detection.segy"
    INLINE_RANGE = (155, 200, 1)  # Старт, стоп, шаг
    XLINE_RANGE = (155, 200, 1)
    SAMPLE_RANGE = (0, 50, 1)
    WELL_LOCATIONS = [(10, 20), (40, 10)]
    
    dataHandler = DataHandler()

    # 1. Загрузка данных
    print("Загрузка SEG-Y данных")
    seismic_data = dataHandler.load_segy_data(SEGY_PATH, INLINE_RANGE, XLINE_RANGE, SAMPLE_RANGE)
    dataHandler.plot_3d_seismic(seismic_data)
    dataHandler.plot_section(seismic_data, 0, "Исходные сейсмические данные")
    
    # 2. Расчет атрибутов
    print("\nРасчет сейсмических атрибутов")
    attributes = dataHandler.enhanced_attributes(seismic_data)
    
    # 3. Загрузка фаций
    print("\nЗагрузка данных по фациям")
    facies = dataHandler.load_facies(seismic_data.shape, WELL_LOCATIONS)
    dataHandler.plot_3d_facies(facies)
    dataHandler.plot_section(facies, 0, "Исходные фации", cmap=mcolors.ListedColormap(['black', 'red', 'green', 'blue']))
    
    # 4. Подготовка данных
    print("\nПодготовка данных для обучения")
    X = attributes.reshape(-1, attributes.shape[-1])
    y = facies.reshape(-1)
    valid_idx = y != -1
    X, y = X[valid_idx], y[valid_idx]
    
    # Балансировка классов
    X_res, y_res = dataHandler.simple_oversample(X, y)
    
if __name__ == "__main__":
    main()
