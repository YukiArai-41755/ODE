import numpy as np
from scipy.integrate import odeint

class Solvers: #常微分方程式を数値的に解くクラスSolvers
    
    def __init__(self, f, xv0, t):
        self.f = f
        self.xv0 = xv0
        self.t = t
        self.internal = self.Internal_processes(self.xv0, self.t)
    
    def using_scipy(self, xv0=None, t=None):
        """
        scipy.integrate.odeintによる常微分方程式の解。returnはndarray
        初期値やtを代入しなければ、それらにはインスタンス変数が採用されます。
        """
        xv0 = self.internal.set_initial_values(xv0)
        t = self.internal.set_t(t)
        xv = odeint(self.f, xv0, t)
        return xv

    def rk4(self, xv0=None, t=None):
        """
        4次のルンゲクッタ法による常微分方程式の解。returnはndarray
        初期値やtを代入しなければ、それらにはインスタンス変数が採用されます。
        """
        xv0 = self.internal.set_initial_values(xv0)
        t = self.internal.set_t(t)
        xv = self.internal.set_initial_array(xv0, t)
        for k in range(t.size-1):
            tau = t[k+1] - t[k]
            F1 = self.f(xv[k,:], t[k])
            F2 = self.f(xv[k,:]+tau/2*F1, t[k]+tau/2)
            F3 = self.f(xv[k,:]+tau/2*F2, t[k]+tau/2)  
            F4 = self.f(xv[k,:]+tau*F3, t[k]+tau)
            xv[k+1,:] = xv[k,:] \
                + tau/6 * (F1 +2*F2 +2*F3 +F4)
        return xv

    def euler(self, xv0=None, t=None):
        """
        オイラー法による状微分方程式の解。returnはndarray
        初期値やtを代入しなければ、それらにはインスタンス変数が採用されます。
        """
        xv0 = self.internal.set_initial_values(xv0)
        t = self.internal.set_t(t)
        xv = self.internal.set_initial_array(xv0, t)
        for k in range(t.size-1):
            tau = t[k+1] - t[k]
            xv[k+1,:] = xv[k,:] + self.f(xv[k,:], t)*tau
        return xv

    class Internal_processes:
        def __init__(self, xv0, t):
            self.xv0 = xv0
            self.t = t

        def set_initial_values(self, xv0):
            if xv0==None:
                xv0 = self.xv0
            return xv0

        def set_t(self, t):
            if t==None:
                t = self.t
            return t
        
        def set_initial_array(self, xv0, t):
            m = t.size
            n = xv0.size
            xv = np.zeros([m, n])
            xv[0,:] = xv0
            return xv