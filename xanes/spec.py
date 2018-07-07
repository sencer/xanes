""" Data container for spectral data

This module provides a `Generic` class as container for spectrum data,
and a `ClassBuilder` class which returns a subclass of `Generic` for a
specific edge of a specific element.

Example:
    # ClassBuilder will create a new class xanes.Spec.TiKEdge dynamically,
    # which is aliased to TiKEdge by `TiKEdge = ...`
    TiKEdge = ClassBuilder("TiKEdge", xmin=4950, xmax=5050, dx=0.1)
    anatase = TiKEdge(x, y)

TODO:
    * implement fromfolder, dumpdb, readdb methods for Generic class.
"""


import numpy as np
import warnings
from inspect import getargspec
from scipy.interpolate import interp1d

class ClassBuilder():
    """ Subclass `Generic` for given parameters, requires either xgrid, or
        (xmin, xmax, dx) provided as arguments.

        Args:
            name (str): name for the dynamically created class
            xmin (float): minimum of the energy range of the spectra
            xmax (float): maximum of the energy range of the spectra
            dx   (float): grid width for the spectral data
            xgrid (np.array): a numpy array in form np.arange(xmin, xmax, dx)
            broadening_function ((callable, [params)]), optional):
                A tuple of a callable, and its parameters as an iterable, that
                will be used for broadening the spectra.

        Returns:
            Dynamically created class xanes.spec.name where name is the input
            parameter


        Example:
            # use a lorentzian for broadening, custom functions can also be used
            from xanes.broadening import lorentzian
            # create a grid for the energy axis
            xgrid = np.arange(4960, 5010, 0.1)
            # gamma of the lorentzian is energy dependent
            gamma = ((xgrid - 4960)/50)**2 + 0.89
            # build the class
            ClassBuilder('TiKEdge', xgrid=xgrid,
                         broadening_function=(lorentzian, (gamma,)))
    """

    def __new__(cls, name, xmin=None, xmax=None, dx=None, xgrid=None,
                broadening_function=None):

        if not ((xmin is None and xmax is None and dx is None) ^
                (xgrid is None)):
            raise Exception(
                      "Either (xmin, xmax, dx) or xgrid should be provided")

        if xgrid is None:
            xgrid = np.arange(xmin, xmax, dx)
        elif xmin is None or xmax is None or dx is None:
            xmin = xgrid[0]
            xmax = xgrid[-1]
            dx = xgrid[1] - xmin

        if broadening_function is not None:
            M = ClassBuilder.broadening_matrix(xgrid, *broadening_function)
        else:
            M = None

        new_class = type(name, (Generic, ), dict(M=M,
                                                 name=name,
                                                 xmin=xmin,
                                                 xmax=xmax,
                                                 dx=dx,
                                                 xgrid=xgrid))
        new_class.x.setflags(write=False)

        return new_class


    @staticmethod
    def broadening_matrix(xgrid, func, args):
        """ Create a matrix M, that when multiplied with spectrum, broadens it

            Args:
                xgrid (np.array): energy grid of the class
                func (callable): broadening function, like a gaussian or
                    a lorentzian. Its signature should be f(x, *args)
                args (iterable): Arguments of the broadening function
        """

        assert len(getargspec(func).args) == len(args) + 1

        L = len(xgrid)
        dx = xgrid[1] - xgrid[0]

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


    def plot(self, ax=None, scale=1, *args, **kwargs):

        if ax is None:
            # check if plt available
            import matplotlib.pyplot as plt
            plt.plot(self.x, self * scale, *args, **kwargs)
            ax = plt.gca()
        else:
            ax.plot(self.x, self * scale, *args, **kwargs)

        ax.set_xlabel("Energy, eV")
        ax.set_yticks(())

        return ax


    @staticmethod
    def _getkwarg(key, default, kwargs):
        try:
            return kwargs.pop(key)
        except:
            return default


    @classmethod
    def fromfile(cls, fname, *args, **kwargs):

        xshift = cls._getkwarg('xshift', 0, kwargs)
        scale = cls._getkwarg('scale', 1, kwargs)
        broaden = cls._getkwarg('broaden', True, kwargs)

        data = np.genfromtxt(fname, *args, **kwargs)

        x = data[:, 0]
        y = data[:, 1:].sum(axis=1)

        return cls(x, y, xshift, scale, broaden)


    @classmethod
    def fromstructure(cls, file, species, idmap=None, warnonly=False,
                      *args, **kwargs):

        if idmap is None:
            idmap = lambda x: "%d.dat" % x

        xshift = cls._getkwarg('xshift', 0, kwargs)
        scale = cls._getkwarg('scale', 1, kwargs)
        broaden = cls._getkwarg('broaden', True, kwargs)
        symprec = cls._getkwarg('symprec', 0.1, kwargs)
        angle_tolerance = cls._getkwarg('angle_tolerance', 15.0, kwargs)


        from ase.io import read
        from os.path import dirname
        from spglib import get_symmetry
        from collections import Counter

        folder = dirname(file)

        atoms = read(file)
        equiv = get_symmetry((atoms.get_cell(),
                              atoms.get_scaled_positions(),
                              atoms.get_atomic_numbers()),
                             symprec=symprec,
                             angle_tolerance=angle_tolerance)['equivalent_atoms']

        numequiv = Counter(equiv)
        mask = (atoms.numbers == species)

        specs = []
        weights = []
        notfound = 0
        for i, atom in enumerate(atoms):

            if mask[i] and i == equiv[i]:
                try:
                    specs.append(cls.fromfile("%s/%s" % (folder, idmap(i)),
                                xshift=xshift, scale=scale, broaden=broaden))
                    weights.append(numequiv[i])
                except IOError:

                    if not warnonly:
                        raise Exception("Some files are not found, please " +
                                        "check idmap, symprec and " +
                                        "angle_tolerance.")
                    specs.append(None)
                    weights.append(-1)
                    notfound += 1

        if warnonly and notfound > 0:
            if notfound == len(weights):
                raise Exception("No spectrum found.")

            warnings.warn("Spectra not found for %d sites" % notfound)

        weights = np.array(weights)
        mask = weights > -1
        raw = np.array([spec._y for i, spec in enumerate(specs) if mask[i]])
        weights = weights[mask]

        return cls(specs[np.argmax(mask)]._x, weights @ raw, xshift=xshift,
                   scale=scale, broaden=broaden), specs, weights / weights.sum()
