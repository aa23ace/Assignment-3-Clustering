# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:06:00 2024

@author: anish
"""
# Import the standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt

# Import user defined packages
import cluster_tools as ct
import errors as err


def read_data(filename):
    '''
    This is a function takes the filename (dataset) as the input and
    return the tuple having two dataframes one with data and other with
    transpose of the data
    '''
    # Reading the dataframe as CSV and skipping the first 4 rows
    df = pd.read_csv(filename, skiprows=4)

    # Drop the Unnecessary colunns from the data
    df.drop(["Unnamed: 67", "Country Code", "Indicator Code"], axis=1,
            inplace=True)

    # Return the tuple of data and its transpose
    return df, df.transpose()


def cluster_num(df):
    '''
    This is a function takes a dataframe as arguments and returns the optimal
    cluster number suitable for the data.
    '''

    # Define the list of clusters and scores
    clusters = []
    scores = []

    # loop over number of clusters
    for ncluster in range(2, 10):

        # Setting up clusters over number of clusters
        kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)

        # Cluster fitting
        kmeans.fit(df)
        labels = kmeans.labels_

        clusters.append(ncluster)
        scores.append(skmet.silhouette_score(df, labels))

    clusters = np.array(clusters)
    scores = np.array(scores)

    # Get the best cluster
    best_ncluster = clusters[scores == np.max(scores)]

    # Return the optimal cluster number
    return best_ncluster[0]


def cluster_data(df, ncluster, year):
    '''
    This is a function which takes dataframe, cluster number and year as
    arguments and returns the clustering of data with the given number of
    cluster numbers and plot it.
    '''

    # Normalising data and storing minimum and maximum
    df_norm, df_min, df_max = ct.scaler(df)

    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)

    # Fit the data
    kmeans.fit(df_norm)

    # Extract cluster labels
    labels = kmeans.labels_

    # Extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_

    # Get the original centers
    cen = ct.backscale(cen, df_min, df_max)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    # extract x and y values of data points
    x = df["CO2 emissions (metric tons per capita)"]
    y = df["Electric power consumption (kWh per capita)"]

    # Intialization of the figure
    plt.figure(figsize=(8, 6))

    # plot data with kmeans cluster number
    scatter_data = plt.scatter(x, y, 10, labels, marker="o", cmap="Paired")

    # show cluster centres
    plt.scatter(xkmeans, ykmeans, 50, c='k', marker="X")

    # Define the plot title
    plt.title('Cluster Analysis of Countries CO2 emissions vs Electric power '
              f'consumption in {year}')

    # Axes labelling
    plt.xlabel("CO2 emissions (metric tons per capita)")
    plt.ylabel("Electric power consumption (kWh per capita)")

    # Define the legend
    legend_labels = [f'Cluster {i}' for i in range(kmeans.n_clusters)]

    # Display the legend
    plt.legend(handles=scatter_data.legend_elements()[0], title="Clusters",
               labels=legend_labels)

    # Display plot
    plt.show()


def get_cluster_data(df, year):
    '''
    This is a function which takes the dataframe and year. It returns the
    data with the required year.
    '''

    # List of the selected indicator
    indicator_list = [
        "Electric power consumption (kWh per capita)",
        "CO2 emissions (metric tons per capita)"]

    # Get the selected fields
    df_cluster = df[df["Indicator Name"].isin(indicator_list)]

    # Select the required columns
    df_year = df_cluster[["Country Name", "Indicator Name", str(year)]]

    # Pivot the data for dersired dataset
    df_year = df_year.pivot(index='Country Name', columns='Indicator Name',
                            values=str(year))

    # Drop NaN values using dropna along row
    df_year.dropna(axis=0, inplace=True)

    # Return the data
    return df_year


def logistic(t, n0, g, t0):
    '''
    This is a function which defines the logistic function with scale factor
    n0 and growth rate g
    '''

    # Define the logistic function
    f = n0 / (1 + np.exp(-g*(t - t0)))

    # Return the function
    return f


def data_fit_forecast(df, country_name, fit_fun, start_year, end_year):
    '''
    This is a function takes dataframe, country name, fit function, start year
    and end year as parameters. Here the data is fit using the fit function and
    forecasting is done for future prediction.
    '''

    # Define the Dataset
    df_sub = df[df["Indicator Name"] ==
                "Electric power consumption (kWh per capita)"]
    df_sub = df_sub[df_sub["Country Name"] == country_name]

    # Drop Indicator Namw
    df_sub.drop(["Indicator Name"], axis=1, inplace=True)

    # Drop NaN values
    df_sub.dropna(axis=1, inplace=True)

    # Set Index as Country Name
    df_sub.set_index("Country Name", inplace=True)

    # Transpose the data
    country_col_df = df_sub.stack().unstack(level=0)
    country_col_df.index = country_col_df.index.astype(int)

    # Fit the Curve
    param, covar = opt.curve_fit(logistic, country_col_df.index,
                                 country_col_df[country_name],
                                 p0=(3e12, 0.10, 1990))

    # Define the range
    year_range = np.arange(start_year, end_year)

    # Get the sigma value
    sigma = err.error_prop(year_range, logistic, param, covar)
    forecast = logistic(year_range, *param)

    # Get the Lower and Upper limits
    low, up = err.err_ranges(year_range, logistic, param, sigma)

    # Intialization of the figure
    plt.figure()

    # Plot Graph
    plt.plot(country_col_df.index, country_col_df[country_name],
             label="Energy Use")
    plt.plot(year_range, forecast, label="Forecast", color='k')
    plt.fill_between(year_range, low, up, color="yellow", alpha=0.7,
                     label='Confidence Margin')
    
    # Define the plot title
    plt.title(f"Electric power consumption forecast for {country_name}")
    
    # Axes labelling
    plt.xlabel("Year")
    plt.ylabel("Electric power consumption (kWh per capita)")

    # Display the legend
    plt.legend()

    # Display plot
    plt.show()


def main():
    # Get the dataframe by calling the read_data method
    df, df_t = read_data("API_19_DS2_en_csv_v2_5998250.csv")

    # Calling the get cluster data function
    df_1990 = get_cluster_data(df, 1990)
    cluster_data(df_1990, cluster_num(df_1990), 1990)

    df_2010 = get_cluster_data(df, 2010)
    cluster_data(df_2010, cluster_num(df_2010), 2010)

    # Calling the forecast function
    data_fit_forecast(df, "United States", logistic, 1960, 2030)

    data_fit_forecast(df, "Japan", logistic, 1960, 2030)


if __name__ == "__main__":
    # Start of the program from here by calling main()
    main()
