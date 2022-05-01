
import paddle
from lettuce.utils import pdtensordot

__all__ = ["Equilibrium", "QuadraticEquilibrium", "IncompressibleQuadraticEquilibrium",
           "QuadraticEquilibrium_LessMemory"]


class Equilibrium:
    pass


class QuadraticEquilibrium(Equilibrium):
    def __init__(self, lattice):
        self.lattice = lattice

    def __call__(self, rho, u, *args):
        exu = pdtensordot(self.lattice.e, u, dims=1)
        uxu = self.lattice.einsum("d,d->", [u, u])
        feq = self.lattice.einsum(
            "q,q->q",
            [self.lattice.w,
             rho * ((2 * exu - uxu) / (2 * self.lattice.cs ** 2) + 0.5 * (exu / (self.lattice.cs ** 2)) ** 2 + 1)]
        )
        return feq


class QuadraticEquilibrium_LessMemory(QuadraticEquilibrium):
    """does the same as the normal equilibrium, how ever it uses somewhere around 20% less RAM,
    but runs about 2% slower on GPU and 11% on CPU

    Use this by setting
    lattice.equilibrium = QuadraticEquilibrium_LessMemory(lattice)
    before starting your simulation
    """

    def __call__(self, rho, u, *args):
        return self.lattice.einsum(
            "q,q->q",
            [self.lattice.w,
             rho * ((2 * pdtensordot(self.lattice.e, u, dims=1) - self.lattice.einsum("d,d->", [u, u]))
                    / (2 * self.lattice.cs ** 2)
                    + 0.5 * (pdtensordot(self.lattice.e, u, dims=1) / (self.lattice.cs ** 2)) ** 2 + 1)]
        )


class IncompressibleQuadraticEquilibrium(Equilibrium):
    def __init__(self, lattice, rho0=1.0):
        self.lattice = lattice
        self.rho0 = rho0

    def __call__(self, rho, u, *args):
        exu = self.lattice.einsum("qd,d->q", [self.lattice.e, u])
        uxu = self.lattice.einsum("d,d->", [u, u])
        feq = self.lattice.einsum(
            "q,q->q",
            [self.lattice.w,
             rho
             + self.rho0 * ((2 * exu - uxu) / (2 * self.lattice.cs ** 2) + 0.5 * (exu / (self.lattice.cs ** 2)) ** 2)]
        )
        return feq

