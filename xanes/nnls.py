import numpy as np
import warnings

from scipy.optimize import nnls


def simplescore(unk, basis, fitstart, fitend, shifts, maxiter):

    assert len(shifts) == len(basis)

    basis = np.column_stack([b[fitstart-i:fitend-i]
                             for i, b in zip(shifts, basis)])


    frac, score = nnls(basis, unk, maxiter)

    return score


class BestNNLS():


    def __init__(self, unknown, basis, score=simplescore, optdim=None,
                 maxshift=5, fitstart=None, fitend=None, maxiter=100,
                 adapter=None, verbose=False, seed=0, popsize=500):


        # get the width of eV grid
        self.dx = unknown.dx

        # convert eV parameters to integer indices
        try:
            maxshift = int(round(maxshift / self.dx))
            self.shifts = None
        except:
            self.shifts = np.round(np.array(maxshift) / self.dx).astype(int)
            maxshift = self.shifts.max()


        if fitstart is None:
            self.fitstart = maxshift
        else:
            self.fitstart = int(round((fitstart - unknown.x[0])/self.dx)) + 1
            assert self.fitstart >= maxshift

        if fitend is None:
            self.fitend = len(unknown) - maxshift
        else:
            self.fitend = int(round((fitend - unknown.x[0]) / self.dx)) + 1
            assert self.fitend + maxshift < len(unknown)

        if optdim is None:
            self.N = len(basis)
        else:
            self.N = optdim

        if self.shifts is None:
            self.shifts = np.array([maxshift] * self.N)
        else:
            assert len(self.shifts) == self.N

        # convert the spectra to regular numpy arrays,
        # but keep the class information
        self.unknown = np.asarray(unknown)[self.fitstart:self.fitend]
        self.basis = np.array(basis)
        self.basis_ = basis
        self.cls = unknown.__class__

        # random number generator with seed
        self.rng = np.random.RandomState(seed)

        # scoring function
        self.score = score

        # store other arguments
        self.maxiter = maxiter
        self.verbose = verbose
        self.npop = popsize

        # a cache to avoid re-evaluating same individuals
        self.cache = {}


    def isnew(self, ind):
        return np.array_str(ind) not in self.cache


    def addtocache(self, ind):
        if self.isnew(ind):
            score = self.score(self.unknown, self.basis, self.fitstart,
                               self.fitend, ind, self.maxiter)
            self.cache[np.array_str(ind)] = score


    def getscore(self, ind):
        return self.cache[np.array_str(ind)]


    def newindividual(self):
        while True:
            ind = np.array([self.rng.randint(-i, i+1) for i in self.shifts])

            if self.isnew(ind):
                self.addtocache(ind)
                break

        return ind


    def mutate(self, ind, rate=0.1):
        counter = 0
        tmp = np.copy(ind)

        while counter < 100 and not self.isnew(tmp):
            index = self.rng.randint(self.N)
            shift = self.shifts[index]
            tmp[index] = self.rng.randint(-shift, shift+1)
            counter += 1

        if self.isnew(tmp):
            self.addtocache(tmp)

        return tmp


    def mate(self, ind1, ind2):

        tmp1 = np.copy(ind1)
        tmp2 = np.copy(ind2)

        if self.N < 3:

            points = [0, 1]
            for cx in self.rng.shuffle(perms):
                tmp1 = np.copy(ind1)
                tmp2 = np.copy(ind2)
                tmp1[cx], tmp2[cx] = tmp2[cx], tmp1[cx]

                if self.isnew(tmp1) or self.isnew(tmp2):
                    break

        else:
            counter = 0

            while counter < 100 and not (self.isnew(tmp1) or self.isnew(tmp2)):

                tmp1 = np.copy(ind1)
                tmp2 = np.copy(ind2)

                cx1 = self.rng.randint(0, self.N - 2)
                cx2 = self.rng.randint(cx1, self.N)

                tmp1[cx1:cx2], tmp2[cx1:cx2] = ind2[cx1:cx2], ind1[cx1:cx2]

                counter += 1

        if self.isnew(tmp1):
            self.addtocache(tmp1)
        if self.isnew(tmp2):
            self.addtocache(tmp2)
        return ind1, ind2


    def run(self, ngen=100, mutation_rate=0.9, mate_rate=0.1,
                  nnew=100, stop=30, tournsize=3):

        assert mutation_rate + mate_rate <= 1.0

        mate_rate += mutation_rate

        pop = [self.newindividual() for _ in range(self.npop)]
        fitnesses = [self.getscore(ind) for ind in pop]

        best = 1e6
        oldbest = 1e6

        def selrandom(k):
            indices = self.rng.choice(np.arange(len(pop)), k, replace=False)
            return [pop[i] for i in indices]

        def seltourn(k):
            return [min(selrandom(tournsize),
                        key=lambda ind: self.getscore(ind)) for _ in range(k)]


        nold = self.npop - nnew
        nconv = 0

        for gen in range(ngen):

            offspring = seltourn(nold)
            lst = np.arange(nold)

            var = []
            for i, ind in enumerate(offspring):

                epsilon = self.rng.rand()

                if epsilon < mutation_rate:
                    var.append(self.mutate(ind))
                elif epsilon < mate_rate:
                    j = (self.rng.choice(lst)%(nold-1)+(i+1))%nold
                    var += self.mate(offspring[i], offspring[j])
                    i, j = max(i, j), min(i, j)

            pop = var + [self.newindividual() for _ in range(nnew)]

            best = min([self.cache[i] for i in self.cache])

            if self.verbose:
                print("%3d %3d %12.6e " % (1+gen, len(pop), best), end="")

            if  oldbest - best < 1E-7:
                nconv += 1
                if self.verbose:
                    print("~ %d" % nconv)
                if nconv >= stop:
                    break
            else:
                if self.verbose:
                    print("")
                nconv = 0

            oldbest = best

        ind = np.array([int(i)
                        for i in min(self.cache,
                                     key=lambda x:
                                     self.cache[x])[1:-1].split()]) * self.dx

        for b, shift in zip(self.basis_, ind):
            b.xshift += shift

        frac, score = nnls(np.column_stack([np.asarray(b)[self.fitstart:self.fitend]
                                            for b in self.basis_]),
                           self.unknown)

        # xshifts where fracs == 0 can drift arbitrarily; fix it
        for i, f in enumerate(frac):
            if f < 1E-8:
                ind[i] = 0

        return (np.array(self.basis_).T @ frac).view(self.cls), frac, score, ind
