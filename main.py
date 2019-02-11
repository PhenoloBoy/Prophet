# import fbprophet
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = xr.open_rasterio(r'c:\temp\chianti.img')
    ndvi = data[:, 0, 0]

    dates = pd.date_range('1/1/1998', '31/12/2013')
    dek = dates[dates.day.isin([1, 11, 21])]
    dek = dek[:ndvi.shape[0]]

    d = {'time': dek, 'ndvi': ndvi}
    df = pd.DataFrame(data=d)

    print(df)


if __name__ == '__main__':
    main()