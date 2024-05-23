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


def determine_outliers(x, y, alpha=0.005):
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


def mardia_test(data, alpha=0.005):
    N, p = data.shape
    mean_vec = np.mean(data, axis=0)

    inv_cov_mat, _ = calculate_cov_inv(data[:, 0], data[:, 1])

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


def calculate_regression_coefficients(Zx, Zy):
    ZxAvg = np.mean(Zx)
    ZyAvg = np.mean(Zy)

    b1_numerator = np.sum((Zx - ZxAvg) * (Zy - ZyAvg))
    b1_denominator = np.sum((Zx - ZxAvg) ** 2)

    b1 = b1_numerator / b1_denominator
    b0 = ZyAvg - b1 * ZxAvg

    return b0, b1


def calculate_regression_metrics(Zy_hat, y):
    n = len(y)
    y_hat = 10**Zy_hat
    y_y_hat_diff_squared = (y - y_hat) ** 2
    y_avg = np.mean(y)
    y_y_avg_diff_squared = (y - y_avg) ** 2
    y_y_hat_diff_y_divided = (y - y_hat) / y

    sy = np.sum(y_y_hat_diff_squared)
    r_squared = 1 - (sy / np.sum(y_y_avg_diff_squared))
    mmre = 1 / n * np.sum(np.abs(y_y_hat_diff_y_divided))
    pred = np.sum(np.abs(y_y_hat_diff_y_divided) < 0.25) / n

    return r_squared, sy, mmre, pred


if __name__ == "__main__":
    x, y = retrieve_data()
    Zx, Zy = normalize_data(x, y)

    outliers = determine_outliers(Zx, Zy)
    while len(outliers) > 0:
        x = np.delete(x, outliers)
        y = np.delete(y, outliers)
        Zx = np.delete(Zx, outliers)
        Zy = np.delete(Zy, outliers)
        outliers = determine_outliers(Zx, Zy)

    print("Викидів не виявлено\n")

    b0, b1 = calculate_regression_coefficients(Zx, Zy)
    print("\nКоефіцієнти регресії:")
    print("b0:", b0)
    print("b1:", b1)

    Zy_hat = b0 + b1 * Zx
    print("\nПредбачені значення:")
    print("Zy_hat:", Zy_hat)

    r_squared, sy, mmre, pred = calculate_regression_metrics(Zy_hat, y)
    print("\nЯкість моделі:")
    print("R^2:", r_squared)
    print("Sy:", sy)
    print("MMRE:", mmre)
    print("PRED:", pred)

    while True:
        value = input("\nВведіть значення RFC для передбачення CBO: ")
        if value == "exit":
            break
        try:
            value = float(value)
            y_hat = 10**b0 * value**b1
            print(f"Значення CBO для RFC={value}: {y_hat:.4f}")
        except ValueError:
            print("Некоректне значення")
            continue