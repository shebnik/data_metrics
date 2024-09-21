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
    Z = np.column_stack((x, y))
    N, d = Z.shape
    Z_mean = np.mean(Z, axis=0)

    S_N = np.zeros((d, d))

    for z_i in Z:
        z_i_centered = z_i - Z_mean
        S_N += np.outer(z_i_centered, z_i_centered)

    S_N /= N

    return np.linalg.inv(S_N), Z_mean


def calculate_mahalanobis_distances(x, y, cov_inv, Z_mean):
    diff_matrix = np.column_stack((x, y)) - Z_mean
    mahalanobis_distances = np.sum(diff_matrix @ cov_inv * diff_matrix, axis=1)
    return mahalanobis_distances


def calculate_test_statistic(n, mahalanobis_distances):
    return ((n - 2) * n / ((n**2 - 1) * 2)) * mahalanobis_distances


def determine_outliers(x, y, alpha=0.05):
    n = len(y)
    cov_inv, Z_mean = calculate_cov_inv(x, y)
    # print("Обернена коваріаційна матриця:")
    # print(cov_inv)
    # print()

    mahalanobis_distances = calculate_mahalanobis_distances(x, y, cov_inv, Z_mean)
    # print("D^2:")
    # print(mahalanobis_distances)
    # print()

    test_statistic = calculate_test_statistic(n, mahalanobis_distances)
    # print("Тестова статистика:")
    # print(test_statistic)
    # print()

    fisher_f = f.ppf(1 - alpha, 2, n - 2)
    # print("F-розподіл Фішера із 2 ступнями вільності для alpha=0.005: ", fisher_f)
    # print()

    indexes = []
    for i in range(n):
        if test_statistic[i] > fisher_f:
            print(f"Видалено викид: Zx={x[i]:.4f}, Zy={y[i]:.4f}")
            indexes.append(i)
    return indexes


if __name__ == "__main__":
    x, y = retrieve_data()

    outliers = determine_outliers(Zx, Zy)
    while len(outliers) > 0:
        x = np.delete(x, outliers)
        y = np.delete(y, outliers)
        Zx = np.delete(Zx, outliers)
        Zy = np.delete(Zy, outliers)
        outliers = determine_outliers(Zx, Zy)

    print("Викидів не виявлено")
    print()
