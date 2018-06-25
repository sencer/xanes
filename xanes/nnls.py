import deap
import numpy as np

from deap import base, algorithms, tools, creator
from scipy.optimize import nnls
from scipy.stats import rankdata, spearmanr

import warnings

def weighted_spearman(x, y, w=None):
    if w is None:
        w = 1/(y + np.percentile(y, 10))
    rx = rankdata(x)
    ry = rankdata(y)
    cov = np.cov(rx, ry, aweights=w)[0, 1]

    return cov / np.std(rx) / np.std(ry)

class BestNNLS():

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray,
                                 fitness=creator.FitnessMin)


    def __init__(self, unknown, basis, max_shift=7, fit_start=None,
                 fit_end=None, maxiter=100, mask=None, popsize=500,
                 score=0, verbose=False, seed=0):

        # convert max_shift, fit_start and fit_end from eV to array units
        self.dx = unknown.x[1] - unknown.x[0]  # eV difference between array elem.
        self.max_shift = int(max_shift / self.dx) + 1

        if fit_start is None:
            self.fit_start = self.max_shift
        else:
            self.fit_start = int((fit_start - unknown.x[0]) / self.dx)
            assert self.fit_start >= self.max_shift

        len_unknown = len(unknown)
        if fit_end is None:
            self.fit_end = len_unknown - self.max_shift
        else:
            self.fit_end = int((fit_end - unknown.x[0]) / self.dx) + 1
            assert self.fit_end + max_shift < len_unknown

        self.min_shift = self.fit_start - self.max_shift
        self.max_shift = self.fit_start + self.max_shift

        # region of the spectrum to be fitted, and its length
        self.unknown = unknown[self.fit_start:self.fit_end]
        self.len = self.fit_end - self.fit_start

        self.maxiter = maxiter
        self.verbose = verbose

        # seeded random number generator
        self._gen = np.random.RandomState(seed)

        # how the scoring will be done
        #     0: MSE
        #     1: weighted MSE, w = 1/unknown
        #     2: (1 - Spearman r)
        #     3: (1 - weighted Spearman r)
        self.score = score

        self.N = len(basis)
        if mask is None:
            self.mask = np.ones(self.N, dtype=bool)
        else:
            self.mask = mask

        # the basis set, and its length (which will be the length
        # of the "individual" to be evolved)
        self.basis = np.empty(len(basis), dtype=object)
        self.basis[:] = basis
        self.basis[~self.mask][:] = 0


        # create the evolutionary toolbox
        self._tb = base.Toolbox()
        self._tb.register("attr_shift", self._gen.randint, self.min_shift,
                                                           self.max_shift)
        self._tb.register("individual", tools.initRepeat, creator.Individual,
                                        self._tb.attr_shift, n=self.N)

        self._tb.register("population", tools.initRepeat, list,
                                        self._tb.individual)
        self._tb.register("evaluate", self.fitness)
        self._tb.register("mate", self.crossover)
        self._tb.register("mutate", self.mutate, rate=0.1)
        self._tb.register("select", tools.selTournament, tournsize=3)

        self.pop = self._tb.population(popsize)
        self.hof = tools.HallOfFame(1, similar=np.array_equal)

        # statistics to log
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        self.cache = {}


    def mutate(self, ind, rate=1E-1):

        # make sure the mutation generates a new individual
        # be content with and old one if it cannot in 100 trials
        counter = 0
        while counter < 100 and np.array_str(ind) in self.cache :
            ind[self._gen.randint(self.N)] = self._gen.randint(self.min_shift,
                                                               self.max_shift)
            counter += 1

        return ind,


    def crossover(self, ind1, ind2):

        counter = 0

        if self.N < 3:

            perms = [(0, 0), (0, 1), (1, 0), (1, 1)]

            for cx1, cx2 in self._gen.shuffle(perms):
                tmp1 = np.array(ind1)
                tmp2 = np.array(ind2)
                tmp1[cx1], tmp2[cx2] = tmp2[cx2], tmp1[cx1]

                if not (np.array_str(tmp1) in self.cache and
                        np.array_str(tmp2) in self.cache):
                    ind1 = tmp1
                    ind2 = tmp2
        else:
            counter = 0
            tmp1 = np.array(ind1)
            tmp2 = np.array(ind2)

            while counter < 100 and (np.array_str(tmp1) in self.cache and
                                     np.array_str(tmp2) in self.cache):
                cx1 = self._gen.randint(0, self.N - 2)
                cx2 = self._gen.randint(cx1, self.N - 1)

                tmp1[cx1:cx2] = ind2[cx1:cx2]
                tmp2[cx1:cx2] = ind1[cx1:cx2]

                counter += 1

            ind1 = tmp1
            ind2 = tmp2

        return creator.Individual(ind1), creator.Individual(ind2)


    def spectrum(self, ind, fractions=None):
        if fractions is None:
            A = np.column_stack((b[s:s+self.len] for b, s in
                                zip(self.basis, ind)))

            # ordinary MSE score
            fractions, fitness = nnls(A, self.unknown, maxiter=self.maxiter)
            fractions[~self.mask] = 0

        spec = A @ fractions

        return spec, fractions, fitness


    def fitness(self, ind):

        s = np.array_str(ind)
        if s not in self.cache:

            fitted, fractions, fitness = self.spectrum(ind)

            if self.score > 0:
                if self.score == 1:
                    # weighted MSE score

                    # scale the errors with the intensity of the spectrum,
                    # so th
                    fitness = np.sum((fitted - self.unknown)**2 /
                                     (self.unknown +
                                      np.percentile(self.unknown, 10)))
                elif self.score == 2:
                    # spearman score
                    fitness = 1 - spearmanr(fitted, self.unknown)[0]
                    # weighted spearman score
                elif self.score == 3:
                    fitness = 1 - weighted_spearman(fitted, self.unknown)

            self.cache[s] = fitness
        else:
            fitness = self.cache[s]

        return fitness,


    def run(self, ngen=100, mutation_rate=0.7, crossover_rate=0.7):

        pop, log = algorithms.eaSimple(self.pop, self._tb, ngen=ngen,
                                       cxpb=crossover_rate,
                                       mutpb=mutation_rate,
                                       stats=self.stats,
                                       halloffame=self.hof,
                                       verbose=self.verbose)

        spec, fractions, fitness = self.spectrum(self.hof[0])

        mask = fractions != 0
        shifts = (self.fit_start - self.hof[0]) * self.dx
        shifts[~mask] = 0

        for b, s in zip(self.basis, shifts):
            b.xshift += s

        fractions, fitness = nnls(np.column_stack([b[self.fit_start:self.fit_end]
                                                   for b in self.basis]),
                                  self.unknown, maxiter=self.maxiter)

        spec = np.array(list(self.basis[:]), dtype=float).T @ fractions

        return spec.view(self.unknown.__class__), fractions, fitness, shifts
