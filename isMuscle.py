import numpy as np
import pandas as pd
import tsfel
import joblib
import pickle
import scipy.stats as stats
from sklearn import preprocessing


def isMuscle(acc_filepath, gyro_filepath, fs=50, window_size=250):
    """Принимает на вход данные с акселерометра и гироскопа.
     На выходе выдает неразмеченный вектор признаков для классификатора."""

    #Считывание с датчиков
    df1 = acc_filepath
    df2 = gyro_filepath

    df1 = df1.drop(df1.columns[[0]], axis=1)
    df2 = df2.drop(df2.columns[[0]], axis=1)
    df1 = df1.drop(df1.columns[[0]], axis=1)
    df2 = df2.drop(df2.columns[[0]], axis=1)

    #df1 = pd.DataFrame((df_acc['acc_x'] ** 2 + df_acc['acc_y'] ** 2 + df_acc['acc_z'] ** 2) ** 0.5)
    #df2 = pd.DataFrame((df_gyr['gyro_x'] ** 2 + df_gyr['gyro_y'] ** 2 + df_gyr['gyro_z'] ** 2) ** 0.5)
    print(df1)
    df1.columns = [['acc_x','acc_y','acc_z']]
    df2.columns = [['gyr_x','gyr_y','gyr_z']]

    # Удаление выбросов по квантилям
    Q1 = df1.quantile(q=.25)
    Q3 = df1.quantile(q=.75)
    IQR = df1.apply(stats.iqr)
    df1 = df1[~((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).any(axis=1)]

    Q1 = df2.quantile(q=.25)
    Q3 = df2.quantile(q=.75)
    IQR = df2.apply(stats.iqr)
    df2 = df2[~((df2 < (Q1 - 1.5 * IQR)) | (df2 > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Объединяем в один датафрейм
    df = pd.merge(df1, df2, left_index=True, right_index=True)


    # Выбор признаков которые будут извлечены
    cfg1 = tsfel.get_features_by_domain('statistical')
    cfg2 = tsfel.get_features_by_domain('temporal')
    cfg3 = tsfel.get_features_by_domain('spectral')

    # Извлечение признаков
    X_train_stat = tsfel.time_series_features_extractor(cfg1, df, fs=fs, window_size=window_size,
                                                 header_names=['acc_x','acc_y','acc_z', 'gyr_x', 'gyr_y', 'gyr_z'])

    X_train_temp = tsfel.time_series_features_extractor(cfg2, df, fs=fs, window_size=window_size,
                                                        header_names=['acc_x','acc_y','acc_z', 'gyr_x', 'gyr_y', 'gyr_z'])

    X_train_spec = tsfel.time_series_features_extractor(cfg3, df, fs=fs, window_size=window_size,
                                                        header_names=['acc_x','acc_y','acc_z', 'gyr_x', 'gyr_y', 'gyr_z'])

    X_train = pd.concat([X_train_stat, X_train_temp, X_train_spec], axis=1, join='inner')

    # Нормализация признаков
    nX_train = pd.DataFrame(preprocessing.normalize(X_train))
    nX_train.columns = X_train.columns

    #nX_train['class'] = np.nan

    df1 = pd.read_csv('/home/lab/Рабочий стол/amalthea_server/amalthea_server/amalthea/libs/Muscule_fatigue/Shapka.csv')
    data = nX_train[nX_train.columns.intersection(df1.columns)]
    data2 = data.tail(1)


    # Загружаем модель
    with open('RandomForest.pkl', 'rb') as file:
        classifier = pickle.load(file)

    #RESULT = classifier.score(data2)
    RESULT = classifier.predict(data2)
    RESULT = str(RESULT)

    return RESULT

