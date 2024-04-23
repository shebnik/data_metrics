import pandas as pd
import numpy as np
from scipy.stats import f, chi2, norm


def retrieve_data():
    df = pd.read_csv("results.csv")
    x = df["Relative RFC"].values.astype(float)
    y = df["Relative CBO"].values.astype(float)
    return x, y


def normalize_data(x, y):
    x = np.log10(x)
    y = np.log10(y)

    print("Нормалізовані дані (log10)")
    print(f"x: {x}")
    print()
    print(f"y: {y}")
    print()
    
    return x, y

def calculate_cov_inv(x, y):
    n = len(x)
    rfc_mean = np.mean(x)
    cbo_mean = np.mean(y)
    cov_matrix = np.zeros((2, 2))
    for i in range(n):
        cov_matrix[0, 0] += (x[i] - rfc_mean) * (x[i] - rfc_mean)
        cov_matrix[0, 1] += (x[i] - rfc_mean) * (y[i] - cbo_mean)
        cov_matrix[1, 0] += (y[i] - cbo_mean) * (x[i] - rfc_mean)
        cov_matrix[1, 1] += (y[i] - cbo_mean) * (y[i] - cbo_mean)
    cov_matrix /= n
    cov_inv = np.linalg.inv(cov_matrix)
    return cov_inv


def calculate_mahalanobis_distances(x, y, cov_inv):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    diff_matrix = np.column_stack((x - x_mean, y - y_mean))
    mahalanobis_distances = np.sum(diff_matrix @ cov_inv * diff_matrix, axis=1)
    return mahalanobis_distances


def calculate_test_statistic(n, mahalanobis_distances):
    test_statistic = np.zeros(n)
    for i in range(n):
        test_statistic[i] = ((n - 2) * n / ((n**2 - 1) * 2)) * mahalanobis_distances[i]
    return test_statistic


def determine_outliers(x, y, alpha = 0.005):    
    n = len(y)
    cov_inv = calculate_cov_inv(x, y)

    mahalanobis_distances = calculate_mahalanobis_distances(x, y, cov_inv)
    test_statistic = calculate_test_statistic(n, mahalanobis_distances)
    fisher_f = f.ppf(1 - alpha, 2, n - 2)

    indexes = []
    for i in range(n):
        if test_statistic[i] > fisher_f:
            print(f"Видалено викид: x={x[i]}, y={y[i]}")
            indexes.append(i)
    return indexes


if __name__ == "__main__":
    x, y = retrieve_data()
    # x, y = normalize_data(x, y)

    outliers = determine_outliers(x, y)
    while len(outliers) > 0:
        x = np.delete(x, outliers)
        y = np.delete(y, outliers)
        outliers = determine_outliers(x, y)

    print("Викидів не виявлено")
    print()