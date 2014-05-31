__author__ = 'roz016'

import numpy as np
import math


#####################################################
############### Regression Functions ################
#####################################################

def simple_linear_regression(df, y_name, x_name):

    df = df[[x_name, y_name]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x_name])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    b = y * x.T * np.linalg.inv(x*x.T)
    return b


def weighted_simple_linear_regression(df, y_name, x_name, x_centre, width):

    df = df[[x_name, y_name]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x_name])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    w = np.matrix(np.zeros(shape=(x0.shape[0], x0.shape[0])))

    for i in range(x0.shape[0]):
        for j in range(x0.shape[0]):
            if i == j:
                if math.fabs(x0[i]-x_centre) < width:
                    w[i, j] = 1.0
                else:
                    w[i, j] = 0.0

    b = y * w * x.T * np.linalg.inv(x * w * x.T)
    return b


def double_linear_regression(df, y_name, x0_name, x1_name):

    df = df[[x0_name, x1_name, y_name]]

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x0_name])
    x1 = np.array(df[x1_name])
    x = np.matrix(np.vstack((x0, x1, np.ones(x0.shape[0]))))

    b = y * x.T * np.linalg.inv(x*x.T)
    return b


def weighted_double_linear_regression(df, y_name, x0_name, x1_name, x_centre, width):

    df = df[[x0_name, x1_name, y_name]]

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x0_name])
    x1 = np.array(df[x1_name])
    x = np.matrix(np.vstack((x0, x1, np.ones(x0.shape[0]))))

    w = np.matrix(np.zeros(shape=(x0.shape[0], x0.shape[0])))

    for i in range(x0.shape[0]):
        for j in range(x0.shape[0]):
            if i == j:
                if math.fabs(x0[i]-x_centre) < width:
                    w[i, j] = 1.0
                else:
                    w[i, j] = 0.0

    b = y * w * x.T * np.linalg.inv(x * w * x.T)
    return b


def direction_weighted_simple_linear_regression(df, y_name, x_name, wind_dir_name, wind_dir_centre, wind_dir_span):
    df = df[[x_name, wind_dir_name, y_name]]

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x_name])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    w = np.matrix(np.zeros(shape=(x0.shape[0], x0.shape[0])))

    for i in range(x0.shape[0]):
        for j in range(x0.shape[0]):
            if i == j:
                distance = degrees_distance(wind_dir_centre, df[wind_dir_name].iloc[i])
                if distance < wind_dir_span:
                    # Kernel cuadratic
                    w[i, j] = 70.0/81.0 * (1-math.fabs(distance/wind_dir_span)**3)**3
                else:
                    w[i, j] = 0.0

    b = y * w * x.T * np.linalg.inv(x * w * x.T)
    return b


def direction_speed_weighted_simple_linear_regression(df, y_name, x_name, wind_dir_name, wind_dir_centre, wind_dir_span, wind_spd_centre, wind_spd_span):

    df = df[[x_name, wind_dir_name, y_name]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x_name])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    w = np.matrix(np.zeros(shape=(x0.shape[0], x0.shape[0])))

    for i in range(x0.shape[0]):
        for j in range(x0.shape[0]):
            if i == j:
                distance = degrees_distance(wind_dir_centre, df[wind_dir_name].iloc[i])
                if distance < wind_dir_span:
                    # Kernel cuadratic
                    w[i, j] = 70.0/81.0 * (1-math.fabs(distance/wind_dir_span)**3)**3
                else:
                    w[i, j] = 0.0
                if math.fabs(df[x_name].iloc[i]-wind_spd_centre) < wind_spd_span:
                    w[i, j] *= 70.0/81.0 * (1-math.fabs((df[x_name].iloc[i]-wind_spd_centre)/wind_spd_span)**3)**3

    b = y * w * x.T * np.linalg.inv(x * w * x.T)
    return b


#####################################################
############# Regression Test Functions #############
#####################################################

def me_no_regression(df, col1, col2):
    """

    :param file_path:
    """
    values = (df[col1] - df[col2])**2

    values = values.apply(np.sqrt)

    return values.sum()/df.shape[0]


def rmse_no_regression(df, col1, col2):
    """

    :param file_path:
    """
    se = ((df[col1] - df[col2])**2).sum()

    return math.sqrt(se/df.shape[0])


def me_simple_linear_regression(test_df, train_df, y_name, x_name):

    params = simple_linear_regression(train_df, y_name, x_name)

    e = 0

    for index, row in test_df.iterrows():
        #print(index)
        e += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))

    return e/test_df.shape[0]


def rmse_simple_linear_regression(test_df, train_df, y_name, x_name):

    params = simple_linear_regression(train_df, y_name, x_name)

    se = 0

    for index, row in test_df.iterrows():
        #print(index)
        se += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))**2

    return math.sqrt(se/test_df.shape[0])


def rmse_weigthed_simple_linear_regression(test_df, train_df, y_name, x_name, width):

    se = 0

    for index, row in test_df.iterrows():
        #print(index)
        params = weighted_simple_linear_regression(train_df, y_name, x_name, row[x_name], width)
        se += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))**2

    return math.sqrt(se/test_df.shape[0])


def me_direction_weighted_simple_linear_regression(test_df, train_df, y_name, x_name, wind_dir_name, wind_dir_span):

    e = 0

    #print('Test dataset is: {}'.format(test_df.shape[0]))
    #print('Train dataset is: {}'.format(train_df.shape[0]))

    for index, row in test_df.iterrows():
        params = direction_weighted_simple_linear_regression(train_df, y_name, x_name, wind_dir_name, row[wind_dir_name], wind_dir_span)
        e += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))

    return e/test_df.shape[0]


def rmse_direction_weighted_simple_linear_regression(test_df, train_df, y_name, x_name, wind_dir_name, wind_dir_span):

    se = 0

    for index, row in test_df.iterrows():
        params = direction_weighted_simple_linear_regression(train_df, y_name, x_name, wind_dir_name, row[wind_dir_name], wind_dir_span)
        se += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))**2

    return math.sqrt(se/test_df.shape[0])


def me_direction_speed_weighted_simple_linear_regression(test_df, train_df, y_name, x_name, wind_dir_name, wind_dir_span, wind_spd_span):

    e = 0

    for index, row in test_df.iterrows():
        params = direction_speed_weighted_simple_linear_regression(train_df, y_name, x_name, wind_dir_name, row[wind_dir_name], wind_dir_span, row[x_name], wind_spd_span)
        e += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))

    return e/test_df.shape[0]


def rmse_direction_speed_weighted_simple_linear_regression(test_df, train_df, y_name, x_name, wind_dir_name, wind_dir_span, wind_spd_span):

    se = 0

    for index, row in test_df.iterrows():
        params = direction_speed_weighted_simple_linear_regression(train_df, y_name, x_name, wind_dir_name, row[wind_dir_name], wind_dir_span, row[x_name], wind_spd_span)
        se += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))**2

    return math.sqrt(se/test_df.shape[0])


#####################################################
##################### Utils #########################
#####################################################

def degrees_distance(angle_a, angle_b):

    return min(math.fabs(angle_a-angle_b), 360-math.fabs(angle_a-angle_b))