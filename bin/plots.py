__author__ = 'SmartWombat'


from ggplot import *
import math
import pandas as pd
import numpy as np

#print type(diamonds)
#print diamonds

def plot_test_scale():
#### Scale testing #####
    x_values = np.arange(0, 41, 1)
    df = pd.DataFrame(x_values, columns=['x_values'])
    df['y_values'] = pd.Series(x_values, index=df.index)
    df['color'] = pd.Series(x_values, index=df.index)

    return ggplot(aes(x='x_values', y='y_values', color='color'), data=df) + \
        geom_point(alpha=0) + \
        scale_colour_gradient2(low="yellow", mid="red", high="blue") + \
        ggtitle("scale_colour_gradient test")

"""
print ggplot(aes(x='drat', y='mpg', color='wt'), data=mtcars) + \
    geom_point() + \
    scale_colour_gradient(low="white", mid="blue", high="black")


print ggplot(aes(x='drat', y='mpg', color='wt'), data=mtcars) + \
    geom_point() + \
    scale_colour_gradient2(low="blue", mid="red", high="yellow")
"""



def plot_wind_data():

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

    x_values = np.arange(0, 40, 0.1)
    y_values = x_values * params[0, 0] + params[0, 1]

    df2 = pd.DataFrame(x_values, columns=['x_values'])
    df2['y_values'] = pd.Series(y_values, index=df2.index)

    return ggplot(aes(x='WindSpd', y='MetarwindSpeed', color='WindDir'), data=df) + \
        geom_point() + \
        scale_colour_gradient(low="yellow", mid="red", high="blue") + \
        ggtitle("GFS - Metar Wind Speed") + \
        xlim(-.9, 40) + \
        ylim(-.9, 30)
        #geom_line(aes('x_values', 'y_values'), data=df2, color='blue')



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
        xlim(-.9, 40) + \
        ylim(-.9, 30)


def plot_kernel_widths():

    dfLEBB = pd.read_csv('../data/results1_LEBBData.csv')
    dfLESO = pd.read_csv('../data/results1_LESOData.csv')
    dfLEVT = pd.read_csv('../data/results1_LEVTData.csv')

    df2LEBB = dfLEBB.groupby('width', as_index=False).mean()
    df2LESO = dfLESO.groupby('width', as_index=False).mean()
    df2LEVT = dfLEVT.groupby('width', as_index=False).mean()

    df3LEBB = dfLEBB.groupby('width').std()
    df3LESO = dfLESO.groupby('width').std()
    df3LEVT = dfLEVT.groupby('width').std()



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

           #
    return ggplot(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LEBB) + \
           geom_point(color='blue') + geom_line(aes(x='xs', y='ys'), data=df4LEBB, color='blue') +\
           geom_point(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LESO, color='purple') + \
           geom_line(aes(x='xs', y='ys'), data=df4LESO, color='purple') + \
           geom_point(aes(x='width', y='me_dir_w_simpl_regr'), data=df2LEVT, color='orange') + \
           geom_line(aes(x='xs', y='ys'), data=df4LEVT, color='orange') + \
           xlim(-5, 185) + \
           ylim(1.5, 3.5)

def test_alpha():
    #### Alpha testing #####
    x_values = np.arange(0, 41, 1)
    df = pd.DataFrame(x_values, columns=['x_values'])
    df['y_values'] = pd.Series(x_values, index=df.index)
    df['alpha'] = 1 - pd.Series(x_values, index=df.index)
    df['color'] = pd.Series(x_values, index=df.index)

    return ggplot(aes(x='x_values', y='y_values', alpha='alpha', color='color'), data=df) + \
        geom_point() + \
        scale_colour_gradient2(low="yellow", mid="red", high="blue") + \
        ggtitle("alpha test")

def plot_function():
    def tricube(x):
        if math.fabs(x) > 2:
            return 0
        else:
            return 70.0/81.0 * (1-math.fabs(x/2)**3)**3
    return ggplot(pd.DataFrame({'x':np.arange(-3,4)}), aes(x='x')) + \
    stat_function(fun=lambda x: tricube(x), color="green")
    #stat_function(fun=np.sin,color="red") + \
    #stat_function(fun=np.cos,color="blue") + \
    #stat_function(fun= lambda x:x**2)



"""
print(ggplot(aes('carat', 'price'), data=diamonds) + \
    geom_point(alpha=1/20.) + \
    ylim(0, 20000))

print ggplot(aes('date','beef * 2000'), data=meat) + \
    geom_line(color='coral') + \
    scale_x_date(breaks=date_breaks('36 months'), labels='%Y') + \
    scale_y_continuous(labels='millions')

print ggplot(aes(x='date', y='beef'), data=meat) + \
    geom_point(color='lightblue') + \
    stat_smooth(span=.15, color='black', se=True) + \
    ggtitle("Beef: It's What's for Dinner") + \
    xlab("Date") + \
    ylab("Head of Cattle Slaughtered")
"""

if __name__ == "__main__":
    #print plot_wind_data()
    print plot_wind_data_kernel(20, 10)
    #print test_alpha()