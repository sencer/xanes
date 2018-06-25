import numpy as np
import warnings
from inspect import getargspec
from scipy.interpolate import interp1d

class ClassBuilder():

    def __new__(cls, name, xmin=None, xmax=None, dx=None, xgrid=None,
                broadening_function=None):

        if xgrid is not None:
            xmin = xgrid[0]
            xmax = xgrid[-1]
            dx = xgrid[1] - xmin
        elif xmin is not None:
            xgrid = np.arange(xmin, xmax, dx)
        else:
            raise Exception("You need to supply xgrid or (xmin, xmax, dx)")

        if broadening_function is not None:
            M = ClassBuilder.broadening_matrix(dx, xgrid, *broadening_function)
        else:
            M = None

        new_class = type(name, (Generic, ), dict(
            xmin=xmin, xmax=xmax, dx=dx,
            x=xgrid,
            M=M
        ))

        new_class.x.setflags(write=False)

        return new_class


    @staticmethod
    def broadening_matrix(dx, xgrid, func, args):

        assert len(getargspec(func).args) == len(args) + 1

        L = len(xgrid)

        args_ = []

        var_args = False

        for arg in args:

            try:
                args_.append(np.array([float(arg)] * L))
            except:
                var_args = True
                args_.append(np.array(arg))
                assert len(arg) == L

        if not var_args:
            args_ = [arg[0] for arg in args_]
            x = dx * np.arange(L, dtype=int)
            B = func(x, *args_)

        M = np.zeros((L, L))
        for i, row in enumerate(M):

            if var_args:
                args_i = [arg[i] for arg in args_]
                Lmax = max(i, L-i)
                x = dx * np.arange(L, dtype=int)
                B = func(x, *args_i)

            row[i:] = B[:L-i]
            row[:i] = B[:i][::-1]

        M /= np.sum(M, axis=1, keepdims=True)

        return M


class Generic(np.ndarray):

    def __new__(cls, E, I, xshift=0, scale=1, broaden=True):


        obj = np.zeros_like(cls.x).view(cls)

        # make sure these are numpy arrays
        obj._x = np.array(E)
        obj._y = np.array(I)

        # and are immutable!
        obj._x.setflags(write=False)
        obj._y.setflags(write=False)

        # not interpolated nor broadened yet
        obj._interpolated = False
        obj._broadened = False

        # set the values
        obj.scale = scale
        obj.xshift = xshift

        obj.interpolate()

        if broaden and cls.M is not None:
            obj.broaden()

        return obj

    @property
    def xshift(self):
        return self._xshift


    @xshift.setter
    def xshift(self, val):

        self._xshift = val

        if self._interpolated:
            self.interpolate()


    @property
    def scale(self):
        return self._scale


    @scale.setter
    def scale(self, val):
        if self._interpolated:
            self *= val / self._scale

        self._scale = val

    @property
    def broadened(self):
        return self._broadened


    @broadened.setter
    def broadened(self, val):

        if val and not self._broadened:
            self.broaden()

        elif not val and self._broadened:
            self._broadened = val
            self.interpolate()


    def interpolate(self):

        self._interpolated = True

        with np.errstate(divide='ignore', invalid='ignore'):
            i = interp1d(self.xshift + self._x, self._y,
                         fill_value="extrapolate",
                         bounds_error=False)(self.__class__.x)

        # # calculate the area under curve for normalization purposes
        # mask = (((self._x + val) >  self.__class__.xmin) &
        #         ((self._x + val) <= self.__class__.xmax))
        self._integral = np.trapz(i, x=self.x)

        if self._integral < 1E-15:
            warnings.warn("No peaks in the region of interest " +
                         ("(%.1f - %.1f). " % (self.__class__.xmin,
                                               self.__class__.xmax))+
                          "Check your x-shift value.")

        self[:] = self._scale * i / self._integral


    def broaden(self):
        self._broadened = True
        b =  self.__class__.M @ self
        self[:] = self._scale * b / np.trapz(b, x=self.x)


    def plot(self, ax=None, *args, **kwargs):

        # TODO check how pandas do that?

        if ax is None:
            # check if plt available
            import matplotlib.pyplot as plt
            plt.plot(self.x, self, *args, **kwargs)
            ax = plt.gca()
        else:
            ax.plot(self.x, self, *args, **kwargs)

        ax.set_xlabel("Energy, eV")
        ax.set_yticks(())

        return ax


    @classmethod
    def fromfile(cls, fname, xshift=0, scale=1, broaden=True, *args, **kwargs):
        data = np.genfromtxt(fname, *args, **kwargs)

        x = data[:, 0]
        y = data[:, 1:].sum(axis=1)

        return cls(x, y, xshift, scale, broaden)


    @classmethod
    def fromdirectory(cls):
        pass
