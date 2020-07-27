# FEM code - Strain relaxation simulation on thin lamella InP/InAsP/InP - Sample #1895
#
# Start : 12 June 2019
# End : xxx
#
# Alexandre Pofelski
# Geometry - Viraj Whabi

# ------------------
# 1) Simulation code written in FEniCS (Python)
# 2) Geometry and mesh imported from .xml file
# 3) Data exported in .vtk file
# ------------------

from dolfin import *
import ufl
import numpy as np
import time

start_time = time.time()

print("Simulation starting time : ", start_time)

# Import geometry, mesh and materials properties

# Complex mesh .xmdf
geometry_folder = "geometry/2D_105nm/"
geometry_filename ="2D_anubis105_140K"
geometry_fileformat = ".xdmf"
mesh = Mesh()
with XDMFFile(geometry_folder + geometry_filename + geometry_fileformat) as file:
    file.read(mesh)

# C_ijkl for each part of the  (simulated data)
filename_bulk = 'InP_stiffness_tensor_simu_exp.npy'
filename_epi = 'InAs35P65_stiffness_tensor_simu_exp.npy'
filename_cap = 'InP_stiffness_tensor_simu_exp.npy'
materials_bulk = np.load(filename_bulk)
materials_epi = np.load(filename_epi)
materials_cap = np.load(filename_cap)

# Atomic misfit at equilibrium (simulated data)
m_bulk = 0
m_epi = 0.01118101586
m_cap = 0

# Define domains, boudaries from the mesh domain (Geometry specific)

tol = 1E-12

class CapLayer(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] >= 0.98 * 1E-6 + tol)

class EpiLayer(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] <= 0.98 * 1E-6 - tol and x[1] >= 0.94 * 1E-6 + tol)

class BulkLayer(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] <= 0.94 * 1E-6 - tol)

class TopInterface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1E-6,  tol)

class BottomInterface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0,  tol)

class LeftInterface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0, tol)

class RightInterface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0, tol)

subdomain_cap = CapLayer()
subdomain_epi = EpiLayer()
subdomain_bulk = BulkLayer()

interface_top = TopInterface()
interface_bottom = BottomInterface()
interface_left = LeftInterface()
interface_right = RightInterface()

domains = MeshFunction('size_t', mesh, mesh.topology().dim())
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
domains.set_all(0)
boundaries.set_all(0)

subdomain_bulk.mark(domains, 0)
subdomain_epi.mark(domains, 1)
subdomain_cap.mark(domains, 2)

interface_bottom.mark(boundaries, 0)
interface_left.mark(boundaries, 1)
interface_right.mark(boundaries, 2)
interface_top.mark(boundaries, 3)

# Function space to apply on the mesh (vector and/tensor on each node)
# 'P' and 1 refer to the most primitive Lagrangian node

V_vector = VectorFunctionSpace(mesh, 'P', 1)
W_tensor = TensorFunctionSpace(mesh, 'P', 1)

# Apply boundary conditions and materials properties in respective sub-domains

class MaterialsLayers(UserExpression):


    def __init__(self, subdomains, tensor_bulk, tensor_epi, tensor_cap, **kwargs):
        self.subdomains = subdomains
        self.tensor_0 = tensor_bulk
        self.tensor_1 = tensor_epi
        self.tensor_2 = tensor_cap
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == 0:
            values[:] = self.tensor_0.flatten()
        if self.subdomains[cell.index] == 1:
            values[:] = self.tensor_1.flatten()
        if self.subdomains[cell.index] == 2:
            values[:] = self.tensor_2.flatten()

    def value_shape(self):
        return(3,3,3,3)

class BiaxialStressInit(UserExpression):


    def __init__(self, subdomains, f_bulk, f_epi, f_cap, **kwargs):
        self.subdomains = subdomains
        self.f_bulk = f_bulk
        self.f_epi = f_epi
        self.f_cap = f_cap
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == 0:
            values[:] = self.f_bulk
        if self.subdomains[cell.index] == 1:
            values[:] = self.f_epi
        if self.subdomains[cell.index] == 2:
            values[:] = self.f_cap

    def value_shape(self):
        return(1,)

C_ij_tensor = MaterialsLayers(domains, materials_bulk, materials_epi, materials_cap)
m = BiaxialStressInit(domains, m_bulk, m_epi, m_cap)
DC_bottom = DirichletBC(V_vector, Constant((0, 0, 0)), interface_bottom)
DCx_right = DirichletBC(V_vector.sub(0), Constant(0), interface_right)
DCx_left = DirichletBC(V_vector.sub(0), Constant(0), interface_left)
DCs = [DC_bottom, DCx_left, DCx_right]

# System of differential equation - Stress/Strain

dx = Measure('dx', domain=mesh, subdomain_data=domains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Strain
def epsilon(u):
    i,j = ufl.indices(2)
    strain = 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    return strain

# Stress
def sigma(u):
    i,j,k,l = ufl.indices(4)
    stress = as_tensor(C_ij_tensor[i,j,k,l] * epsilon(u)[k,l], (i,j))
    return stress

# Initial stress because of the epitaxy - See Laurent Clement thesis
def sigma_id(m):
    i,j,k,l = ufl.indices(4)
    stress_init = as_tensor(C_ij_tensor[i,j,k,l] * m[0] * Identity(3)[k,l], (i,j))
    return stress_init

def epsilon_m(u, m):
    epsilonm = epsilon(u) - m[0] * Identity(3)
    return epsilonm

def sigma_m(u, m):
    i, j, k, l = ufl.indices(4)
    stress = as_tensor(C_ij_tensor[i, j, k, l] * epsilon_m(u, m)[k, l], (i, j))
    return stress

# Define variational problem
u = TrialFunction(V_vector)
v = TestFunction(V_vector)
# Generic ==> T = Constant((0, 0, 0))
# Generic ==> f = Constant((0, 0, 0))
a = inner((sigma(u)), epsilon(v)) * dx
# Generic ==> L = dot(f, v)*dx + dot(T, v) * ds + inner(sigma_id(m), epsilon(v)) * dx
L = inner(sigma_id(m), epsilon(v)) * dx

# Solve PDE

# print(list_linear_solver_methods(), list_krylov_solver_preconditioners())

U = Function(V_vector)

parameters["form_compiler"]["representation"] = 'uflacs'

print("Geometry and FEM parameters loaded")
print(time.time() - start_time)

start_time_calc = time.time()

# Recommended FEniCS - small geometries
#solve(a == L, U, DC_bottom, solver_parameters={'linear_solver':'gmres', 'preconditioner': 'ilu'})

#solve(a == L, U, DC_bottom, solver_parameters={'linear_solver':'mumps', 'preconditioner': 'hypre_euclid'})

# Fastest option #

solve(a == L, U, DCs, solver_parameters={'linear_solver':'cg', 'preconditioner': 'hypre_amg'})

#solve(a ==L, U, DC_bottom, solver_parameters={'linear_solver': 'gmres','preconditioner': 'hypre_euclid'})

print("Calculation done")
print(time.time() - start_time_calc)

#Export parameters

export_folder = 'results/'
filename_param = geometry_filename

#Export displacement
start_time_displacement = time.time()

vtkfile = File(export_folder + filename_param + '_displacement' '.pvd')
vtkfile << U

print('Displacement field saved')
print(time.time() - start_time_displacement)

#Export strain
start_time_strain = time.time()

project_w = project(epsilon(U), W_tensor, solver_type='cg', preconditioner_type='hypre_amg')

vtkfile = File(export_folder + filename_param + '_strain' + '.pvd')
vtkfile << project_w

project_w = project(epsilon_m(U,m), W_tensor, solver_type='cg', preconditioner_type='hypre_amg')

vtkfile = File(export_folder + filename_param + '_strain_m' + '.pvd')
vtkfile << project_w

print('Strain tensor saved')
print(time.time() - start_time_strain)

# Export stress
start_time_stress = time.time()

project_w = project(sigma_m(U,m), W_tensor, solver_type='cg', preconditioner_type='hypre_amg')

vtkfile = File(export_folder + filename_param + '_stress_m' + '.pvd')
vtkfile << project_w

print('Stress tensor saved')
print(time.time() - start_time_stress)

print("Total FEniCS simulation time")
print(time.time() - start_time)
