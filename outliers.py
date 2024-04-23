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


def determine_outliers(x, y, alpha=0.005):
    n = len(y)
    cov_inv = calculate_cov_inv(x, y)

    mahalanobis_distances = calculate_mahalanobis_distances(x, y, cov_inv)
    test_statistic = calculate_test_statistic(n, mahalanobis_distances)
    fisher_f = f.ppf(1 - alpha, 2, n - 2)

    indexes = []
    for i in range(n):
        if test_statistic[i] > fisher_f:
            print("Видалено викид: x={:.4f}".format(x[i]) + ", y={:.4f}".format(y[i]))
            indexes.append(i)
    return indexes


def mardia_test(x, y, alpha=0.005):
    def mardia_skewness(X):
        n = X.shape[0]
        X_mean = np.mean(X, axis=0)
        S_inv = np.linalg.inv(np.cov(X.T))
        skewness = 0
        for i in range(n):
            for j in range(n):
                skewness += ((X[i] - X_mean).T @ S_inv @ (X[j] - X_mean))**3
        skewness /= n**2
        print(f"Багатовимірна асиметрія Мардіа: {skewness:.4f}")
        test_stat = n / 6 * skewness
        p_value = 1 - chi2.cdf(test_stat, df=6)
        return test_stat, p_value

    def mardia_kurtosis(X):
        n = X.shape[0]
        X_mean = np.mean(X, axis=0)
        S_inv = np.linalg.inv(np.cov(X.T))
        kurtosis = 0
        for i in range(n):
            kurtosis += ((X[i] - X_mean).T @ S_inv @ (X[i] - X_mean))**2
        kurtosis /= n
        print(f"Багатовимірний ексцес Мардіа: {kurtosis:.4f}")
        expected_kurtosis = 8
        test_stat = (kurtosis - expected_kurtosis) / np.sqrt(64 / n)
        p_value = 2 * norm.cdf(-np.abs(test_stat))
        return test_stat, p_value

    X = np.column_stack((x, y))

    skewness_stat, skewness_p_value = mardia_skewness(X)
    print(f"Тестова статистика для асиметрії: {skewness_stat:.4f}, p-значення: {skewness_p_value:.8f}\n")

    kurtosis_stat, kurtosis_p_value = mardia_kurtosis(X)
    print(f"Тестова статистика для ексцесу: {kurtosis_stat:.4f}, p-значення: {kurtosis_p_value:.8f}\n")

    if skewness_p_value < alpha or kurtosis_p_value < alpha:
        print(f"Двовимірні дані не є нормальними з рівнем значущості {alpha}\n")
    else:
        print(f"Двовимірні дані є нормальними з рівнем значущості {alpha}\n")



if __name__ == "__main__":
    x, y = retrieve_data()
    mardia_test(x, y)
    # x, y = normalize_data(x, y)

    outliers = determine_outliers(x, y)
    while len(outliers) > 0:
        x = np.delete(x, outliers)
        y = np.delete(y, outliers)
        outliers = determine_outliers(x, y)

    print("Викидів не виявлено")
    print()

    mardia_test(x, y)
