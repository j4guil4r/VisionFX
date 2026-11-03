import math
import numpy as np


class LowPassFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.s = None

    def filter(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        x_arr = np.asarray(x, dtype=np.float64)
        if self.s is None:
            self.s = x_arr.copy()
        else:
            self.s = self.alpha * x_arr + (1.0 - self.alpha) * self.s
        return self.s

    def last(self):
        return self.s

    def reset(self):
        self.s = None


class OneEuroFilter:
    def __init__(self, min_cutoff=0.5, beta=0.05, d_cutoff=1.0):
        """
        :param min_cutoff: Frecuencia de corte mínima (Hz) a velocidad 0. 
                           Menor valor = mayor estabilidad cuando estás quieto.
        :param beta: Coeficiente de velocidad. 
                     Mayor valor = respuesta más instantánea ante movimientos rápidos.
        :param d_cutoff: Frecuencia de corte para el filtro de la derivada.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    def _alpha(self, rate, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, timestamp):
        x_arr = np.asarray(x, dtype=np.float64)

        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter(x_arr)

        dt = timestamp - self.last_time
        self.last_time = timestamp

        # Evitar divisiones por cero o timestamps duplicados
        if dt <= 1e-5:
            return self.x_filter.last()

        rate = 1.0 / dt
        prev_x = self.x_filter.last()

        # 1. Estimar velocidad (derivada discreta) y suavizarla
        dx = (x_arr - prev_x) * rate
        edx = self.dx_filter.filter(dx, self._alpha(rate, self.d_cutoff))

        # 2. Calcular corte adaptativo componente a componente
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(rate, cutoff)

        return self.x_filter.filter(x_arr, alpha)

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None