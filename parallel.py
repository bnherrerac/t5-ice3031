import numpy as np
import matplotlib.pyplot as plt
import pyvista
import ufl
import time
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, mesh, plot, nls, log, io

from hyperelasticity import HyperelasticModel

ksp_types = ['richardson', 'cg', 'bicg', 'bcgs', 'gmres'] # ibcgs, tfqmr
pc_types = ['jacobi', 'sor', 'ilu', 'gamg']

times = {}

model = HyperelasticModel(nn = [20,10,10], ksp_type=ksp_types[4], pc_type=pc_types[3], show=False)
times[ksp_types[0]] = model.run_simulation(gif=True)



