from fbprophet import Prophet
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import outlier


def main():

    data = xr.open_rasterio(r'C:\Users\Pier\PycharmProjects\Prophet\data\Ispra_2x4.img')
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

    perval = 600
    future = m.make_future_dataframe(periods=perval)
    forecast = m.predict(future)
    without = outlier.dbl_mad_clnr((tsd['y'] - forecast['yhat'])[:-perval])
    without += forecast['yhat'][:-perval]

    without = pd.Series(without.values, index=dek).to_frame(name='y')
    without.reset_index(inplace=True)
    without.rename(columns={'index': 'ds'}, inplace=True)

    m2 = Prophet(growth='linear', yearly_seasonality=True)
    m2.fit(without)
    forecast_out = m2.make_future_dataframe(periods=perval).predict(without)

    fig1 = m2.plot(forecast_out)
    fig2 = m2.plot_components(forecast_out)


    #
    #
    #
    # tsd.set_index(pd.to_datetime(tsd['ds'], unit='s'), inplace=True)
    #
    # points = tsd[tsd['ds'].isin(m.changepoints)]
    #
    # tsd.set_index(tsd['ds'], inplace=True)
    # (tsd['y'] - forecast['yhat']).plot()
    # m.plot(forecast)
    #
    # fig1 = m.plot(forecast)
    # fig2 = m.plot_components(forecast)
    #
    # plt.figure(3)
    # tsd['y'].plot()
    # points['y'].plot()
    #
    # # plt.show(fig1)
    # # plt.plot(fig2)

    plt.show()




if __name__ == '__main__':
    main()