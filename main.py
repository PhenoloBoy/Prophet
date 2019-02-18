from fbprophet import Prophet
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


def main():
    data = xr.open_rasterio(r'C:\Users\Pier\PycharmProjects\Prophet\data\chianti.img')
    ndvi = data[:, 0, 0]

    dates = pd.date_range('1/1/1998', '31/12/2013')
    dek = dates[dates.day.isin([1, 11, 21])]
    dek = dek[:ndvi.shape[0]]

    ts = pd.Series(ndvi, index=dek)

    ts[ts.isin(range(250, 256))] = pd.NaT
    # tsd = ts.resample('D').interpolate(method='linear').to_frame(name='y')

    tsd = ts.to_frame(name='y')

    tsd.reset_index(inplace=True)
    tsd.rename(columns={'index': 'ds'}, inplace=True)

    m = Prophet(growth='linear', yearly_seasonality=True)
    m.fit(tsd)

    change = tsd[tsd['ds'].isin(m.changepoints)]
    register_matplotlib_converters()

    future = m.make_future_dataframe(periods=600)
    future.tail()
    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    #

    tsd.set_index(pd.to_datetime(tsd['ds'], unit='s'), inplace=True)

    points = tsd[tsd['ds'].isin(m.changepoints)]

    plt.figure(3)
    tsd['y'].plot()
    points['y'].plot()

    # plt.show(fig1)
    # plt.plot(fig2)

    plt.show()




if __name__ == '__main__':
    main()