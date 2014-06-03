__author__ = 'SmartWombat'

import pandas
import random
from regressor import *


datasets = ['../data/LEBBData.csv','../data/LEVTData.csv','../data/LESOData.csv']

for dataset in datasets:

    print(dataset[-12:])

    df = pandas.read_csv(dataset, na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    f = open('../data/results_{}'.format(dataset[-12:]), 'w')
    f.write('me_no_regr,me_simpl_regr,me_dir_w_simpl_regr,width\n')

    for i in range(10):

        print(i)

        rows = random.sample(df.index, int(df.shape[0]*.75))
        train_df = df.ix[rows]
        test_df = df.drop(rows)

        for width in [5, 10, 20, 30, 40, 60, 90, 120, 180]:
            f.write('{},'.format(rmse_no_regression(test_df, 'MetarwindSpeed', 'WindSpd')))
            f.write('{},,'.format(rmse_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd')))
            f.write('{},{}\n'.format(rmse_direction_weighted_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd', 'WindDir', width), width))

    f.close()
