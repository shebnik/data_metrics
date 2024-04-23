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
    cov_matrix /= n - 1
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


def determine_outliers(x, y):
    n = len(y)
    cov_inv = calculate_cov_inv(x, y)

    mahalanobis_distances = calculate_mahalanobis_distances(x, y, cov_inv)

    test_statistic = calculate_test_statistic(n, mahalanobis_distances)

    a = 0.005
    fisher_f = f.ppf(1 - a, 2, n - 2)

    indexes = []
    for i in range(n):
        if test_statistic[i] > fisher_f:
            print(f"Видалено викид: x={x[i]}, y={y[i]}")
            indexes.append(i)
    return indexes


def mardia_multivariate_skewness(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov_matrix = np.cov(x, y, bias=True)
    cov_inv = np.linalg.inv(cov_matrix)

    skewness = 0
    for i in range(n):
        diff_vector = np.array([x[i] - x_mean, y[i] - y_mean])
        skewness += (diff_vector.T @ cov_inv @ diff_vector) ** 3

    skewness = skewness / n**2
    test_statistic = n / 6 * skewness
    p_value = 1 - chi2.cdf(test_statistic, 2 * (2 + 1) * (2 + 2) / 6)

    print(f"Багатовимірна асиметрія Мардіа: {skewness:.6f}")
    print(f"Тестова статистика для асиметрії: {test_statistic:.6f}")
    print(f"p-значення для асиметрії: {p_value:.6f}")
    print()

    return test_statistic, p_value


def mardia_multivariate_kurtosis(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov_matrix = np.cov(x, y, bias=True)
    cov_inv = np.linalg.inv(cov_matrix)

    kurtosis = 0
    for i in range(n):
        diff_vector = np.array([x[i] - x_mean, y[i] - y_mean])
        kurtosis += (diff_vector.T @ cov_inv @ diff_vector) ** 2

    kurtosis = kurtosis / n
    expected_kurtosis = 2 * (2 + 2)
    test_statistic = kurtosis - expected_kurtosis
    p_value = 2 * norm.cdf(-abs(test_statistic) / np.sqrt(8 * expected_kurtosis / n))

    print(f"Багатовимірний ексцес Мардіа: {kurtosis:.6f}")
    print(f"Тестова статистика для ексцесу: {test_statistic:.6f}")
    print(f"p-значення для ексцесу: {p_value:.6f}")
    print()

    return test_statistic, p_value


if __name__ == "__main__":
    x, y = retrieve_data()
    x, y = normalize_data(x, y)

    outliers = determine_outliers(x, y)
    while len(outliers) > 0:
        x = np.delete(x, outliers)
        y = np.delete(y, outliers)
        outliers = determine_outliers(x, y)

    print("Викидів не виявлено")
    print()

    skewness_stat, skewness_p = mardia_multivariate_skewness(x, y)
    kurtosis_stat, kurtosis_p = mardia_multivariate_kurtosis(x, y)

    alpha = 0.005
    if skewness_p < alpha or kurtosis_p < alpha:
        print("Двовимірні дані не є нормальними з рівнем значущості 0.005")
    else:
        print("Двовимірні дані є нормальними з рівнем значущості 0.005")
