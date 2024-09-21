import pandas as pd
import numpy as np
from scipy.stats import f, chi2, norm, kurtosis, skew


def retrieve_data():
    df = pd.read_csv("results3.csv")
    y = df["WMC"].values.astype(float)
    x1 = df["DIT"].values.astype(float)
    x2 = df["NOC"].values.astype(float)
    return y, x1, x2


def normalize_data(y, x1, x2):
    y = np.log10(y)
    x1 = np.log10(x1)
    x2 = np.log10(x2)

    print("Нормалізовані дані (log10)")
    print(f"y: {y}")
    print()
    print(f"x1: {x1}")
    print()
    print(f"x2: {x2}")
    print()

    return y, x1, x2


def calculate_cov_inv(y, x1, x2):
    cov_inv = np.linalg.inv(np.cov(np.column_stack((y, x1, x2)).T))
    return cov_inv


def calculate_mahalanobis_distances(y, x1, x2, cov_inv):
    data = np.column_stack((y, x1, x2))
    means = np.mean(data, axis=0)
    diff_matrix = data - means
    mahalanobis_distances = np.sum(diff_matrix @ cov_inv * diff_matrix, axis=1)
    return mahalanobis_distances


def calculate_test_statistic(n, mahalanobis_distances):
    test_statistic = np.zeros(n)
    for i in range(n):
        test_statistic[i] = ((n - 3) * n / ((n**2 - 1) * 3)) * mahalanobis_distances[i]
    return test_statistic


def determine_outliers(y, x1, x2, alpha=0.005):
    n = len(y)
    cov_inv = calculate_cov_inv(y, x1, x2)

    mahalanobis_distances = calculate_mahalanobis_distances(y, x1, x2, cov_inv)
    test_statistic = calculate_test_statistic(n, mahalanobis_distances)
    fisher_f = f.ppf(1 - alpha, 3, n - 3)

    indexes = []
    for i in range(n):
        if test_statistic[i] > fisher_f:
            print(
                "Видалено викид: x1={:.4f}, x2={:.4f}, y={:.4f}".format(
                    x1[i], x2[i], y[i]
                )
            )
            indexes.append(i)
    return indexes


def mardia_test(data, alpha=0.005):
    N, p = data.shape
    mean_vec = np.mean(data, axis=0)

    inv_cov_mat = calculate_cov_inv(data[:, 0], data[:, 1], data[:, 2])

    centered_data = data - mean_vec

    skewn = 0
    for i in range(N):
        for j in range(N):
            delta_i = centered_data[i]
            delta_j = centered_data[j]
            term = (delta_i @ inv_cov_mat @ delta_j) ** 3
            skewn += term
    skewn /= N**2

    kurt = 0
    for i in range(N):
        delta_i = centered_data[i]
        term = (delta_i @ inv_cov_mat @ delta_i) ** 2
        kurt += term
    kurt /= N

    skewn_stat = N * skewn / 6
    kurt_stat = (kurt - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / N)

    skewn_crit = chi2.ppf(1 - alpha, p * (p + 1) * (p + 2) / 6)
    kurt_crit = norm.ppf(1 - alpha)

    is_normal = (skewn_stat < skewn_crit) and (abs(kurt_stat) < kurt_crit)

    print(
        f"Асиметрія Мардія: {skewn:.6f}\nТестова Статистика: {skewn_stat:.6f}\nКритичне значення: {skewn_crit:.6f}\n"
    )
    print(
        f"Ексцес Мардія: {kurt:.6f}\nТестова Статистика: {abs(kurt_stat):.6f}\nКритичне значення: {kurt_crit:.6f}\n"
    )
    print(f"Дані нормально розподілені: {is_normal}\n")


if __name__ == "__main__":
    y, x1, x2 = retrieve_data()
    # y, x1, x2 = normalize_data(y, x1, x2)
    mardia_test(np.column_stack((y, x1, x2)))

    outliers = determine_outliers(y, x1, x2)
    while len(outliers) > 0:
        y = np.delete(y, outliers)
        x1 = np.delete(x1, outliers)
        x2 = np.delete(x2, outliers)
        outliers = determine_outliers(y, x1, x2)
    print("Викидів не виявлено\n")

    mardia_test(np.column_stack((y, x1, x2)))