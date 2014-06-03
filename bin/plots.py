__author__ = 'SmartWombat'

from ggplot import *
from regressor import weighted_simple_linear_regression
import math
import pandas as pd
import numpy as np
import datetime
from scipy import interpolate

def plot_fig_1():

    df = pd.read_csv('../data/LEBBData.csv', na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    df = df[['WindSpd', 'MetarwindSpeed', 'WindDir']]
    data = np.matrix(df)

    y = np.matrix(np.array(df['MetarwindSpeed']))
    x0 = np.array(df['WindSpd'])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    params = y * x.T * np.linalg.inv(x*x.T)

    x_values = np.arange(0, 36, 0.1)
    y_values = x_values * params[0, 0] + params[0, 1]

    df2 = pd.DataFrame(x_values, columns=['x_values'])
    df2['y_values'] = pd.Series(y_values, index=df2.index)

    x_values = np.arange(0, 36, 1.1)
    y_values = np.arange(0, 36, 1.1)
    
    for i, x_centre in np.ndenumerate(x_values):
        params = weighted_simple_linear_regression(df, 'MetarwindSpeed', 'WindSpd', x_centre, 10)
        y_values[i] = x_values[i] * params[0, 0] + params[0, 1]

    df3 = pd.DataFrame(x_values, columns=['x_values'])
    df3['y_values'] = pd.Series(y_values, index=df3.index)

    return ggplot(aes(x='WindSpd', y='MetarwindSpeed'), data=df) + \
        geom_point(color="orange") + \
        ggtitle("LEBB Wind Speed") + \
        xlim(-.9, 35) + \
        ylim(-.9, 27) + \
        xlab("GFS Wind Speed [knots]") + \
        ylab("METAR Wind Speed [knots]") + \
        geom_line(aes('x_values', 'y_values'), data=df2, color="darkblue") + \
        geom_line(aes('x_values', 'y_values'), data=df3, color="red")

def plot_fig_1p5():

    df = pd.read_csv('../data/LEBBData.csv', na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    df = df[['date', 'WindSpd', 'MetarwindSpeed', 'WindDir']]

    df.date = df.date.apply(lambda t: pd.to_datetime(t, format='%Y-%m-%d %H:%M'))

    date1 = datetime.datetime(2011, 10, 12)
    date2 = datetime.datetime(2011, 10, 14)

    print(type(df.date[3]))

    return ggplot(aes(x='date', y='MetarwindSpeed'), data=df[2000:2001]) + \
        geom_point(color="darkblue") + \
        scale_x_date(breaks=date_breaks('24 hours'), labels='%Y-%m-%dT%H') + \
        xlab("Time Stamp") + \
        ylab("Wind Speed [knots]") + \
        ggtitle("LEBB Sample Dataset")


def plot_fig_2(speed, width):

    df = pd.read_csv('../data/LESOData.csv', na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    df = df[['WindSpd', 'MetarwindSpeed', 'WindDir']]

    def tricube(x, speed, width):
        if math.fabs(x) < speed-width or math.fabs(x) > speed+width:
            return 0.0
        else:
            return 70.0/81.0 * (1-math.fabs((x-speed)/width)**3)**3

    df['Weigth'] = df['WindSpd'].map(lambda x: tricube(x, speed, width))

    df = df[(df['WindSpd'] > speed-width) & (df['WindSpd'] < speed+width)]

    return ggplot(aes(x='WindSpd', y='MetarwindSpeed', alpha='Weigth'), data=df) + \
        geom_point(color="orange") + \
        ggtitle("LESO Wind Speed") + \
        xlim(-.9, 35) + \
        ylim(-.9, 28) + \
        xlab("GFS Wind Speed [knots]") + \
        ylab("METAR Wind Speed [knots]") + \
        stat_function(fun=lambda x: 25*tricube(x, speed, width), color="darkblue", alpha=1.0)

def plot_fig_3_1():

    df = pd.read_csv('../data/LESOData.csv', na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    df = df[['WindSpd', 'MetarwindSpeed', 'WindDir']]
    data = np.matrix(df)

    y = np.matrix(np.array(df['MetarwindSpeed']))
    x0 = np.array(df['WindSpd'])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    params = y * x.T * np.linalg.inv(x*x.T)

    x_values = np.arange(0, 40, 0.1)
    y_values = x_values * params[0, 0] + params[0, 1]

    df2 = pd.DataFrame(x_values, columns=['x_values'])
    df2['y_values'] = pd.Series(y_values, index=df2.index)

    return ggplot(aes(x='WindSpd', y='MetarwindSpeed', color='WindDir'), data=df) + \
        geom_point() + \
        scale_colour_gradient(low="yellow", mid="blue", high="yellow") + \
        ggtitle("LESO GFS - Metar Wind Speed (GFS Wind Direction Colored)") + \
        xlim(-.9, 36) + \
        ylim(-.9, 26) + \
        xlab("GFS Wind Speed [knots]") + \
        ylab("METAR Wind Speed [knots]")
        #geom_line(aes('x_values', 'y_values'), data=df2, color='blue')

def plot_fig_3_2(dir, width):

    df = pd.read_csv('../data/LESOData.csv', na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    df = df[['WindSpd', 'MetarwindSpeed', 'WindDir']]

    def degrees_distance(angle_a, angle_b):
        return min(math.fabs(angle_a-angle_b), 360-math.fabs(angle_a-angle_b))

    def tricube(x, dir, width):
        distance = degrees_distance(x, dir)
        if distance > width:
            return 0.0
        else:
            # Kernel cuadratic
            return 70.0/81.0 * (1-(distance/width)**3)**3


    df['Weigth'] = df['WindDir'].map(lambda x: tricube(x, dir, width))

    """
    if dir>width:
        df = df[(df['WindDir'] > dir-width) & (df['WindDir'] < dir+width)]
        print(len(df.index))
    else:
        df1 = df[(df['WindDir'] > 0) & (df['WindDir'] < dir+width)]
        df2 = df[(df['WindDir'] > 360+(dir-width)) & (df['WindDir'] < 360)]
        print(dir)
        print(width)
        print(len(df2.index)+len(df1.index))
        df = pd.concat([df1, df2])
    """

    return ggplot(aes(x='WindSpd', y='MetarwindSpeed', alpha='Weigth', color='WindDir'), data=df) + \
        geom_point() + \
        scale_colour_gradient(low="yellow", mid="blue", high="yellow") + \
        ggtitle("LESO GFS - Metar Wind Speed (GFS Wind Direction Filtered)") + \
        xlim(-.9, 36) + \
        ylim(-.9, 26) + \
        xlab("GFS Wind Speed [knots]") + \
        ylab("METAR Wind Speed [knots]")

def plot_wind_data_kernel(speed, width):

    df = pd.read_csv('../data/LEBBData.csv', na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    df = df[['WindSpd', 'MetarwindSpeed', 'WindDir']]

    def tricube(x, speed, width):
        if math.fabs(x) < speed-width or math.fabs(x) > speed+width:
            return 0.0
        else:
            return 70.0/81.0 * (1-math.fabs((x-speed)/width)**3)**3

    df['Weigth'] = df['WindSpd'].map(lambda x: tricube(x, speed, width))

    df = df[(df['WindSpd'] > speed-width) & (df['WindSpd'] < speed+width)]

    return ggplot(aes(x='WindSpd', y='MetarwindSpeed', alpha='Weigth', color='WindDir'), data=df) + \
        geom_point() + \
        scale_colour_gradient(low="yellow", mid="red", high="blue") + \
        ggtitle("GFS - Metar Wind Speed") + \
        xlim(-.9, 36) + \
        ylim(-.9, 26)


def plot_fig_4():

    dfLEBB = pd.read_csv('../data/results1_LEBBData.csv')
    dfLESO = pd.read_csv('../data/results1_LESOData.csv')
    dfLEVT = pd.read_csv('../data/results1_LEVTData.csv')

    df2LEBB = dfLEBB.groupby('width', as_index=False).mean()
    df2LESO = dfLESO.groupby('width', as_index=False).mean()
    df2LEVT = dfLEVT.groupby('width', as_index=False).mean()

    tckLEBB = interpolate.splrep(df2LEBB.width.values, df2LEBB.me_dir_w_simpl_regr, s=0)
    tckLESO = interpolate.splrep(df2LESO.width.values, df2LESO.me_dir_w_simpl_regr, s=0)
    tckLEVT = interpolate.splrep(df2LEVT.width.values, df2LEVT.me_dir_w_simpl_regr, s=0)

    x_new = np.arange(5,180,1)

    yLEBB_new = interpolate.splev(x_new, tckLEBB, der=0)
    yLESO_new = interpolate.splev(x_new, tckLESO, der=0)
    yLEVT_new = interpolate.splev(x_new, tckLEVT, der=0)

    daLEBB = np.vstack([x_new, yLEBB_new]).transpose()
    daLESO = np.vstack([x_new, yLESO_new]).transpose()
    daLEVT = np.vstack([x_new, yLEVT_new]).transpose()


    df4LEBB = pd.DataFrame(daLEBB, columns=['xs', 'ys'])
    df4LESO = pd.DataFrame(daLESO, columns=['xs', 'ys'])
    df4LEVT = pd.DataFrame(daLEVT, columns=['xs', 'ys'])

    return ggplot(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LEBB) + \
           geom_point(color='blue') + geom_line(aes(x='xs', y='ys'), data=df4LEBB, color='darkblue') +\
           geom_point(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LESO, color='maroon') + \
           geom_line(aes(x='xs', y='ys'), data=df4LESO, color='purple') + \
           geom_point(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LEVT, color='orange') + \
           geom_line(aes(x='xs', y='ys'), data=df4LEVT, color='orange') + \
           xlim(-5, 185) + \
           ylim(2.5, 4) + \
           xlab("GFS Wind Direction Kernel dmax [degrees]") + \
           ylab("METAR Wind Speed mean 10 hold-out RMSE")

def plot_fig_5():

    dfLEBB = pd.read_csv('../data/results2_LEBBData.csv')
    dfLESO = pd.read_csv('../data/results2_LESOData.csv')
    dfLEVT = pd.read_csv('../data/results2_LEVTData.csv')

    df2LEBB = dfLEBB.groupby('width', as_index=False).mean()
    df2LESO = dfLESO.groupby('width', as_index=False).mean()
    df2LEVT = dfLEVT.groupby('width', as_index=False).mean()

    df3LEBB = dfLEBB.groupby('width').std()
    df3LESO = dfLESO.groupby('width').std()
    df3LEVT = dfLEVT.groupby('width').std()



    tckLEBB = interpolate.splrep(df2LEBB.width.values, df2LEBB.me_dir_w_simpl_regr, s=0)
    tckLESO = interpolate.splrep(df2LESO.width.values, df2LESO.me_dir_w_simpl_regr, s=0)
    tckLEVT = interpolate.splrep(df2LEVT.width.values, df2LEVT.me_dir_w_simpl_regr, s=0)

    x_new = np.arange(.5,10,.5)

    yLEBB_new = interpolate.splev(x_new, tckLEBB, der=0)
    yLESO_new = interpolate.splev(x_new, tckLESO, der=0)
    yLEVT_new = interpolate.splev(x_new, tckLEVT, der=0)

    daLEBB = np.vstack([x_new, yLEBB_new]).transpose()
    daLESO = np.vstack([x_new, yLESO_new]).transpose()
    daLEVT = np.vstack([x_new, yLEVT_new]).transpose()


    df4LEBB = pd.DataFrame(daLEBB, columns=['xs', 'ys'])
    df4LESO = pd.DataFrame(daLESO, columns=['xs', 'ys'])
    df4LEVT = pd.DataFrame(daLEVT, columns=['xs', 'ys'])

    return ggplot(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LEBB) + \
           geom_point(color='blue') + geom_line(aes(x='xs', y='ys'), data=df4LEBB, color='darkblue') +\
           geom_point(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LESO, color='maroon') + \
           geom_line(aes(x='xs', y='ys'), data=df4LESO, color='purple') + \
           geom_point(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LEVT, color='orange') + \
           geom_line(aes(x='xs', y='ys'), data=df4LEVT, color='orange') + \
           xlim(-.9, 10) + \
           ylim(2, 3)


if __name__ == "__main__":
    #print plot_fig_2(20, 10)
    #print plot_fig_3_1()
    #print plot_fig_3_2(0, 35)
    #print plot_fig_3_2(180, 35)
    print plot_fig_1p5()
