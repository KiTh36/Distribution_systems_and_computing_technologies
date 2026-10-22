


import numpy as np
from time import perf_counter

WIDTH = 800
HEIGHT = 600
MAX_ITER = 300

X_MIN, X_MAX = -2.0, 0.5
Y_MIN, Y_MAX = -1.25, 1.25


def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    z = 0j

    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c

    return max_iter


def poslidovno():
    pixels = []
    dx = (X_MAX - X_MIN) / WIDTH
    dy = (Y_MAX - Y_MIN) / HEIGHT

    for row in range(HEIGHT):
        y = Y_MIN + row * dy
        row_pixels = []

        for col in range(WIDTH):
            x = X_MIN + col * dx
            row_pixels.append(
                mandelbrot_pixel(complex(x, y), MAX_ITER)
            )

        pixels.append(row_pixels)

    return pixels


def mandelbrot_numpy(endpoint=False):
    # Повторюємо формулу координат із poslidovno()
    x = X_MIN + np.arange(WIDTH) * ((X_MAX - X_MIN) / WIDTH)
    y = Y_MIN + np.arange(HEIGHT) * ((Y_MAX - Y_MIN) / HEIGHT)

    if endpoint:
        x = np.linspace(X_MIN, X_MAX, WIDTH)
        y = np.linspace(Y_MIN, Y_MAX, HEIGHT)

    cx, cy = np.meshgrid(x, y)

    zx = np.zeros((HEIGHT, WIDTH))
    zy = np.zeros((HEIGHT, WIDTH))
    out = np.full((HEIGHT, WIDTH), MAX_ITER, dtype=np.int32)
    active = np.ones((HEIGHT, WIDTH), dtype=bool)

    for n in range(MAX_ITER):
        # Значення до оновлення z
        escaped_before = active & (zx * zx + zy * zy > 4.0)
        out[escaped_before] = n
        active &= ~escaped_before

        if not active.any():
            break

        zx_new = zx * zx - zy * zy + cx
        zy_new = 2.0 * zx * zy + cy

        zx = np.where(active, zx_new, zx)
        zy = np.where(active, zy_new, zy)

    return out


if __name__ == "__main__":
    base = np.asarray(poslidovno(), dtype=np.int32)

    for endpoint in (True, False):
        t0 = perf_counter()
        result = mandelbrot_numpy(endpoint=endpoint)
        elapsed = perf_counter() - t0

        print(f"endpoint={endpoint}")
        print("NumPy:", round(elapsed, 4), "с")
        print(
            "розбіжних точок:",
            int(np.count_nonzero(result != base)),
            "із",
            base.size
        )
