"""Решения задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    p, n = matrices.shape[0], matrices.shape[1]
    vectors = np.asarray(vectors).reshape(p, n, 1)
    # сумма по p от A_i @ x_i
    return np.einsum("pij,pjk->ik", matrices, vectors)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_matrix_histograms(stats: MatrixStatistics) -> None:
    """Гистограммы значений по строкам и по столбцам."""
    import matplotlib.pyplot as plt

    matrix = stats.matrix
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in matrix:
        axes[0].hist(row, alpha=0.4)
    axes[0].set_title("Строки")
    for col in matrix.T:
        axes[1].hist(col, alpha=0.4)
    axes[1].set_title("Столбцы")
    plt.tight_layout()
    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i, j = np.indices((rows, columns))
    return np.where((i + j) % 2 == 0, first, second)


def _canvas(image_height, image_width, shape_color, background_color):
    dtype = np.asarray([*background_color, *shape_color]).dtype
    image = np.empty((image_height, image_width, 3), dtype=dtype)
    image[:] = background_color
    return image


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = _canvas(image_height, image_width, shape_color, background_color)
    top = max((image_height - height) // 2, 0)
    left = max((image_width - width) // 2, 0)
    image[top:top + height, left:left + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = _canvas(image_height, image_width, shape_color, background_color)
    y0 = (image_height - 1) / 2
    x0 = (image_width - 1) / 2
    y, x = np.ogrid[:image_height, :image_width]
    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    v = np.asarray(values, dtype=float)
    mid = v[1:-1]
    maxima = np.where((mid > v[:-2]) & (mid > v[2:]))[0] + 1
    minima = np.where((mid < v[:-2]) & (mid < v[2:]))[0] + 1
    moving = np.convolve(v, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=v.mean(),
        variance=v.var(),
        std=v.std(),
        local_maxima_indices=maxima.tolist(),
        local_minima_indices=minima.tolist(),
        moving_average=moving,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
