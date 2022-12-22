import numpy as np

class Lotka_Volterra: #微分方程式を定義するクラス
    def __init__(self, a, b, c, d, time_inv, max_day):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.day = np.arange(0, max_day, time_inv)

    def f(self, xy, t):
        vx = self.a*xy[0] - self.b*xy[0]*xy[1]
        vy = -self.d*xy[1] + self.c*xy[0]*xy[1]
        return np.array([vx, vy])
    
    def conservation(self, x, y):
        return self.c*x + self.b*y - self.d*np.log(x) - self.a*np.log(y)
    
    class F_:
        def __init__(self, g, p, q, d):
            self.g = g
            self.p = p
            self.q = q
            self.d = d

        def f_(self, xy, t=None):
            self.xy = xy
            vx = self.g(xy)*xy[0] - self.p(xy)*xy[1]
            vy = -self.d*xy[1] + self.q(xy)*xy[1]
            return np.array([vx, vy])

    class Logistic:
        def __init__(self, rx, Kx, ry, Ky,):
            self.rx, self.Kx, self.ry, self.Ky = rx, Kx, ry, Ky           

        def logisticx(self,xy):
            return self.rx*(1 - xy[0]/self.Kx)

        def logisticy(self,xy):
            return self.ry*(xy[0] - xy[1]/self.Ky)
    
    class Holling:
        def __init__(self, aa, h):
            self.aa = aa
            self.h = h

        def holling(self, xy):
            return self.aa*xy[0]/(1+self.aa*self.h*xy[0])