import numpy as np
import matplotlib.pyplot as plt
import pyvista
import ufl
import time
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, mesh, plot, nls, log, io


class HyperelasticModel():
    
    def __init__(self, nn, ksp_type='cg', pc_type='gamg', pc_factor_mat_solver_type='superlu', show=False) -> None:
        self.nx = nn[0]
        self.ny = nn[1]
        self.nz = nn[2]
        self.ksp_type = ksp_type
        self.pc_type = pc_type
        self.pc_factor_mat_solver_type = pc_factor_mat_solver_type
        self.show = show
        
    def create_prism(self):

        self.xlim = 4
        self.ylim = 2
        self.zlim = 2
        vertex = [self.xlim, self.ylim, self.zlim]

        self.domain = mesh.create_box(MPI.COMM_WORLD, 
                                [[0.0, 0.0, 0.0], vertex], 
                                [self.nx, self.ny, self.nz], 
                                mesh.CellType.tetrahedron) 
        
        self.V = fem.VectorFunctionSpace(self.domain, ('Lagrange', 2))

    def import_mesh(self):
        with io.XDMFFile(MPI.COMM_WORLD, 'rve_small.xdmf', 'r') as xdmf:
            self.domain = xdmf.read_mesh(name='Grid')
            self.V = fem.VectorFunctionSpace(self.domain, ('Lagrange', 2))
            

    def mark(self):
        
        # self.atol = 1E-10
        self.atol = 0.5

        _, _, domain_geometry = plot.create_vtk_mesh(self.domain, self.domain.topology.dim)

        # max_dims = np.around([max(domain_geometry[:,0]), max(domain_geometry[:,1]), max(domain_geometry[:,2])], 5)
        # min_dims = np.around([min(domain_geometry[:,0]), min(domain_geometry[:,1]), min(domain_geometry[:,2])], 5)

        max_dims = [24.76658, 24.74611, 24.77482]
        min_dims = [0.28539, 0.2329,  0.2496 ]

        print(f"max_dims = {max_dims}")
        print(f"min_dims = {min_dims}")

        # def left(x): # D0
        #     return np.isclose(x[0], 0, atol=self.atol)
        # def right(x): # D1
        #     return np.isclose(x[0], self.xlim, atol=self.atol)
        # def front(x): # N
        #     return np.isclose(x[1], 0, atol=self.atol)
        # def back(x):
        #     return np.isclose(x[1], self.ylim, atol=self.atol)
        # def bottom(x):
        #     return np.isclose(x[2], 0, atol=self.atol)
        # def top( x):
        #     return np.isclose(x[2], self.zlim, atol=self.atol)

        def left(x): # D0
            return np.isclose(x[0], min_dims[0], atol=self.atol)
        def right(x): # D1
            return np.isclose(x[0], max_dims[0], atol=self.atol)
        def front(x): # N
            return np.isclose(x[1], min_dims[1], atol=self.atol)
        def back(x):
            return np.isclose(x[1], max_dims[1], atol=self.atol)
        def bottom(x):
            return np.isclose(x[2], min_dims[2], atol=self.atol)
        def top(x):
            return np.isclose(x[2], max_dims[2], atol=self.atol)

        fdim = self.domain.topology.dim -1

        left_facets = mesh.locate_entities_boundary(self.domain, fdim, left)
        right_facets = mesh.locate_entities_boundary(self.domain, fdim, right)
        front_facets = mesh.locate_entities_boundary(self.domain, fdim, front)
        back_facets = mesh.locate_entities_boundary(self.domain, fdim, back)
        bottom_facets = mesh.locate_entities_boundary(self.domain, fdim, bottom)
        top_facets = mesh.locate_entities_boundary(self.domain, fdim, top)

        print()

        marked_facets = np.hstack([left_facets, right_facets, front_facets, back_facets, bottom_facets, top_facets])
        marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2),
                                    np.full_like(front_facets, 3), np.full_like(back_facets, 4),
                                    np.full_like(bottom_facets, 5), np.full_like(top_facets, 6)])
        sorted_facets = np.argsort(marked_facets)

        self.facet_tag = mesh.meshtags(self.domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

    def run_simulation(self, gif=False):
        
        # self.create_prism()
        self.import_mesh()
        self.mark()

        alpha = 1
        delta_max = 4

        # Dirichlet boundary condition
        u_d0 = fem.Constant(self.domain, PETSc.ScalarType((0, 0, 0)))
        u_d1 = fem.Constant(self.domain, PETSc.ScalarType((0, 0, 0)))
        T = fem.Constant(self.domain, PETSc.ScalarType((0, 0, 0)))
        # T_up = fem.Constant(self.domain, PETSc.ScalarType((0, 0, 0.5)))

        left_dofs = fem.locate_dofs_topological(self.V, self.facet_tag.dim, self.facet_tag.find(1))
        right_dofs = fem.locate_dofs_topological(self.V, self.facet_tag.dim, self.facet_tag.find(2))
        front_dofs = fem.locate_dofs_topological(self.V, self.facet_tag.dim, self.facet_tag.find(3))
        back_dofs = fem.locate_dofs_topological(self.V, self.facet_tag.dim, self.facet_tag.find(4))
        bottom_dofs = fem.locate_dofs_topological(self.V, self.facet_tag.dim, self.facet_tag.find(5))
        top_dofs = fem.locate_dofs_topological(self.V, self.facet_tag.dim, self.facet_tag.find(6))

        bcs = [fem.dirichletbc(u_d0, left_dofs, self.V), fem.dirichletbc(u_d1, right_dofs, self.V)]

        eta = ufl.TestFunction(self.V)   # variación admisible (test function)

        self.u = fem.Function(self.V)         # desplazamiento
        self.u.name = "Desplazamiento"

        # Spatial dimension
        d = len(self.u)

        # Identity tensor
        I = ufl.variable(ufl.Identity(d))

        # Deformation gradient
        F = ufl.variable(I + ufl.grad(self.u)) 

        # Right Cauchy-Green tensor
        C = ufl.variable(F.T * F)

        # Invariants of deformation tensors
        I1 = ufl.variable(ufl.tr(C)) # = tr(F^T F)
        I2 = ufl.variable(0.5*(ufl.tr(C)*ufl.tr(C) - ufl.tr(C*C))) 
        J  = ufl.variable(ufl.det(F))

        # Mooney-Rivlin parameters
        c10 = 0.8
        c01 = 0.5
        D1 = 0.34

        # Stored strain energy density (Birzle et al. model)
        psi = c10 * (J**(-2/3)*I1 - 3) + c01 * (J**(-4/3)*I2) + (1/D1)*(J-1)**2

        # Stress
        # Hyper-elasticity
        P = ufl.diff(psi, F)

        sigma = ufl.variable(J**(-1)*P*F.T)

        metadata = {"quadrature_degree": 3}

        ds = ufl.Measure('ds', domain=self.domain, subdomain_data=self.facet_tag, metadata=metadata)
        dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)

        F = ufl.inner(P, ufl.grad(eta))*dx
        F += -ufl.inner(T, eta)*ds(3) -ufl.inner(T, eta)*ds(4)
        F += -ufl.inner(T, eta)*ds(5) -ufl.inner(T, eta)*ds(6)

        problem = fem.petsc.NonlinearProblem(F, self.u, bcs)
        solver = nls.petsc.NewtonSolver(self.domain.comm, problem)

        # Set Newton solver options
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "incremental"

        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        try:
            opts[f"{option_prefix}ksp_type"] = self.ksp_type
            opts[f"{option_prefix}pc_type"] = self.pc_type
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = self.pc_factor_mat_solver_type
        except:
            raise ValueError('Option not allowed.')
        ksp.setFromOptions()

        log.set_log_level(log.LogLevel.INFO)

        t = 0
        nmax = 5
        delta_max = 8

        if gif:
            pyvista.start_xvfb()
            plotter = pyvista.Plotter()
            plotter.open_gif("alv_deformation.gif", fps=5)

            gif_topology, gif_cells, gif_geometry = plot.create_vtk_mesh(self.u.function_space)
            function_grid = pyvista.UnstructuredGrid(gif_topology, gif_cells, gif_geometry)

            values = np.zeros((gif_geometry.shape[0], 3))
            values[:, :len(self.u)] = self.u.x.array.reshape(gif_geometry.shape[0], len(self.u))
            function_grid["u"] = values
            function_grid.set_active_vectors("u")

            # Warp mesh by deformation
            warped = function_grid.warp_by_vector("u", factor=1)
            warped.set_active_vectors("u")

            # Add mesh to plotter and visualize
            actor = plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, 10])

            # Compute magnitude of displacement to visualize in GIF
            Vs = fem.FunctionSpace(self.domain, ("Lagrange", 2))
            magnitude = fem.Function(Vs)
            us = fem.Expression(ufl.sqrt(sum([self.u[i]**2 for i in range(len(self.u))])), Vs.element.interpolation_points())
            magnitude.interpolate(us)
            warped["mag"] = magnitude.x.array
            plotter.update_scalar_bar_range([-delta_max, delta_max])

        residuals = np.zeros((nmax,1))

        for n in range(nmax):
            u_d1.value[0] = alpha*delta_max*n/(nmax-1)
            t1 = time.time()
            num_its, converged = solver.solve(self.u)
            t2 = time.time() - t1
            assert(converged)
            self.u.x.scatter_forward()
            
            if gif:
                function_grid["u"][:, :len(self.u)] = self.u.x.array.reshape(gif_geometry.shape[0], len(self.u))
                magnitude.interpolate(us)
                warped.set_active_scalars("mag")
                warped_n = function_grid.warp_by_vector(factor=1)
                plotter.update_coordinates(warped_n.points.copy(), render=False)
                # plotter.update_scalar_bar_range([0, np.max(self.u.x.array)])
                plotter.update_scalars(magnitude.x.array)
                plotter.write_frame()

            # residuals[n] = solver.b.norm()
            # print(f"b type = {type(solver.b)}")

            t += t2
        
        plotter.close()
        print(f"Time elapsed = {t} s.")

        if self.show:
            top, cel, geom = plot.create_vtk_mesh(self.u.function_space)
            grid = pyvista.UnstructuredGrid(top, cel, geom)
            pyvista.set_jupyter_backend('panel')
            pyvista.start_xvfb()
            plotter = pyvista.Plotter()
                
            plotter.add_mesh(grid, show_edges=True)

            grid["u"] = self.u.x.array.reshape(geom.shape[0], len(self.u))
            Vs = fem.FunctionSpace(self.domain, ("Lagrange", 2))
            magnitude = fem.Function(Vs)
            us = fem.Expression(ufl.sqrt(sum([self.u[i]**2 for i in range(len(self.u))])), Vs.element.interpolation_points())
            magnitude.interpolate(us)
            warped = grid.warp_by_vector("u", factor=1)
            warped.set_active_vectors("u")
            warped['mag'] = magnitude.x.array
            warped.set_active_scalars("mag")
            # warped_n = grid.warp_by_vector(factor=1)
            plotter.update_coordinates(warped.points.copy(), render=False) 
            # plotter.update_scalar_bar_range([0, np.max(magnitude.x.array)])
            # plotter.update_scalars(magnitude.x.array)
            # plotter.write_frame()
            plotter.view_xy()
            plotter.show()
            plotter.close()


        return t
