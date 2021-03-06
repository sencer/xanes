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
                                                 x=xgrid))
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

        # number of arguments provided in args should be 1
        # less than needed for calling func
        assert len(getargspec(func).args) == len(args) + 1

        L = len(xgrid)
        dx = xgrid[1] - xgrid[0]

        args_ = []
        e_dependent = False
        for arg in args:
            try:
                # try casting arg into a float, if not castable it is
                # supposed to be an array - do E-dependent broadening
                args_.append(float(arg))
            except:
                e_dependent = True
                assert len(arg) == L  # make sure we have correct size for grid
                args_.append(np.array(arg))

        if not e_dependent:
            x = dx * np.arange(L, dtype=int)
            B = func(x, *args_)

        M = np.zeros((L, L))
        nn = int(5/dx)  # average over this many units to interpolate the tail
        for i, row in enumerate(M):

            if e_dependent:
                args_i = [arg if isinstance(arg, float) else arg[i]
                          for arg in args_]

                Lmax = max(i+1, L-i+nn)
                x = dx * np.arange(Lmax, dtype=int)
                B = func(x, *args_i)

            row[i:] = B[:L-i]
            row[:i] = B[1:i+1][::-1]

            # add an average broadening to the high-energy tail
            # computed as if the highest 5 eV region of the spectrum
            # represents the uncalculated part
            # integral(tail of lorentzian) * 
            row[-nn:] += B[L-i:].sum()/nn

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

        if broaden and cls.M is not None:
            obj.broaden()

        return obj


    def __array_finalize__(self, obj):

        if obj is None:
            return

        if len(self) < len(obj):
            pass

        elif not np.allclose(np.asarray(self),
                             np.asarray(obj)):
            # TODO
            # test for edge cases; currently this sets
            # missing parameters for new spectra generated
            # by algebraic operations
            self._x = self.x
            self._x.setflags(write=False)
            self._y = np.asarray(self)
            self._y.setflags(write=False)
            self._interpolated = True
            self._broadened = False
            self._xshift = 0
            self._scale = 1


    @property
    def xshift(self):
        return self._xshift


    @xshift.setter
    def xshift(self, val):

        if hasattr(self, '_xshift') and val == self._xshift:
            return

        self._xshift = val
        self.interpolate()

    @property
    def scale(self):
        return self._scale


    @scale.setter
    def scale(self, val):
        if hasattr(self, '_scale') and val == self._scale:
            return

        if self._interpolated:
            self *= val / self._scale

        self._scale = val


    @property
    def broadened(self):
        return self._broadened


    @broadened.setter
    def broadened(self, val):

        if val and not self._broadened:
            # Not already broadened, do broaden
            self.broaden()

        elif not val and self._broadened:
            # Broadened, undo broaden
            self._broadened = val
            self.interpolate()


    def interpolate(self):

        self._interpolated = True

        # do not interpolate if grids are same
        if len(self.__class__.x) == len(self._x) and \
                np.allclose(self.__class__.x, self._x) and \
                self._xshift == 0:
            self[:] = self._y
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                i = interp1d(self.xshift + self._x, self._y,
                            fill_value="extrapolate",
                            bounds_error=False)(self.__class__.x)

            self[:] = i

        if self._broadened:
            self.broaden()
        else:
            self.normalize()


    def normalize(self):

        self.integrate()
        self *= self._scale / self._integral


    def integrate(self):
        # # calculate the area under curve for normalization purposes
        # mask = (((self._x + val) >  self.__class__.xmin) &
        #         ((self._x + val) <= self.__class__.xmax))
        self._integral = np.trapz(np.asarray(self), x=self.x)

        if self._integral < 1E-15:
            warnings.warn("No peaks in the region of interest " +
                          ("(%.1f - %.1f). " % (self.__class__.xmin,
                                                self.__class__.xmax))+
                          "Check your x-shift value.")


    def broaden(self):
        self._broadened = True
        self[:] =  self.__class__.M @ self
        self.normalize()


    def plot(self, ax=None, scale=1, rank=False, *args, **kwargs):

        if rank:
            from scipy.stats import rankdata
            data = rankdata(self)
        else:
            data = np.asarray(self) * scale

        if ax is None:
            # check if plt available
            import matplotlib.pyplot as plt
            plt.plot(self.x, data, *args, **kwargs)
            ax = plt.gca()
        else:
            ax.plot(self.x, data, *args, **kwargs)

        ax.set_xlabel("Energy, eV")
        ax.set_yticks(())

        return ax


    def align(self, other, xmin=None, xmax=None, dx=5,
              metric='pearson', inplace=True):

        if metric == 'pearson':
            from scipy.stats import pearsonr
            dist = lambda x, y: pearsonr(x, y)[0]
        elif metric == 'spearman':
            from scipy.stats import spearmanr
            dist = lambda x, y: spearmanr(x, y)[0]
        elif metric == 'rmse':
            dist = lambda x, y: -np.mean(np.sqrt((x-y)**2)/x)

        dx = int(dx/self.dx)

        if xmin is None:
            xmin = dx
        else:
            xmin = int((xmin - self.xmin) / self.dx)


        if xmax is None:
            xmax = len(self) - 1 - dx
        else:
            xmax = int((xmax - self.xmin) / self.dx + 1)

        l = xmax - xmin

        assert dx <= xmin
        assert xmax + dx < len(self)

        this = np.asarray(self)
        that = np.asarray(other)

        distances = [dist(that[xmin:xmax], this[s:s+l])
                     for s in range(xmin-dx, xmin+dx)]

        imaxdist = np.argmax(distances)

        shift = dx - imaxdist

        shift *= self.dx

        if inplace:
            self.xshift += shift

        return shift, distances[imaxdist]


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


    @classmethod
    def dump(cls, file, specs):
        assert isinstance(specs, list)

        specs = [(i._x, i._y, i.xshift, i.scale, i._broadened)
                 for i in specs]

        np.savez(file, original=False, specs=specs, name=cls.name,
                 x=(cls.xmin, cls.xmax, cls.dx), M=cls.M)




def load(file):

    with np.load(file) as dump:
        specs = dump['specs']
        name = str(dump['name'])
        xmin, xmax, dx = dump['x']
        M = dump['M']
        original = bool(dump['original'])

    new_class = type(name, (Generic, ), dict(name=name,
                                             xmin=xmin,
                                             xmax=xmax,
                                             dx=dx,
                                             x=np.arange(xmin, xmax, dx)))

    new_class.M = M

    specs = [new_class(i[0], i[1], xshift=i[2], scale=i[3], broaden=i[4])
                for i in specs]

    return new_class, specs
