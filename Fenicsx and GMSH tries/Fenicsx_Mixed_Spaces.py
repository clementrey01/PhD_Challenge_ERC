import dolfinx.mesh as mesh
import dolfinx.fem as fem
import dolfinx
from dolfinx.fem.petsc import LinearProblem
import ufl
import dolfinx.io as io
from mpi4py import MPI
import basix
import numpy as np
import scifem

comm = MPI.COMM_WORLD

# ------------------------ Physical Parameters ---------------------------
M = 1.0e10         # Biot modulus (Pa)
md = 6.0e-10       # Specific storage (1/Pa)
kappa = 1.0e-6     # Hydraulic diffusivity (m^2/s)
kappa_T = 1.3e-6   # Thermal diffusivity (m^2/s)
beta_e = 4.25e-5   # Thermal expansion (1/K)
kTp = 1.0e-9       # Mechano-caloric (m^2/s/K)
kappaTp = 4.0e-11  # Thermo-osmosis (m^2*K/(Pa*s))

# Boundary values
T_top = 13.0       # °C (atmospheric temperature)
T_bot = 200.0      # °C (bottom temperature)
T_inj = 20.0       # °C (injection temperature)
p_top = 1.013e5    # Pa (atmospheric pressure)
p_inj = 22e6       # Pa (injection pressure)
p_ext = 18e6       # Pa (extraction pressure)
length = 3000
depth = 6000
x_inj = [250, -5000]  # Injection well coordinates
x_ext = [-250, -5000]  # Extraction well coordinates
nx, nz = 300, 600

domain = mesh.create_rectangle(comm, [[-length/2, -depth], [length/2, 0]], [nx, nz], mesh.CellType.quadrilateral)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim - 1, tdim)

# ------------------------ Simulation Parameters ---------------------------
timestep = float(86400) # 1 day
num_steps = 10
output_interval = 1
dt = fem.Constant(domain, dolfinx.default_scalar_type(timestep))


# ---------------------- Function Spaces and DOFs ------------------------
el_pT = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
el_mixed = basix.ufl.mixed_element([el_pT, el_pT])

W = fem.functionspace(domain, el_mixed)
p, T = ufl.TrialFunctions(W) # Delta of Pressure and of Temperature
q, v = ufl.TestFunctions(W)

# Initialize functions
wh = fem.Function(W)
w0 = fem.Function(W)  # solution from previous converged step
ph, Th = ufl.split(wh)
p0, T0 = ufl.split(w0)

# Set initial conditions, Delta of 0 everywhere
w0.x.array[:] = 0.0


# ------------------------- Variational Formulation ------------------------
# Bilinear form
a = (
    # Pressure equation
    (1/dt) * ufl.inner(q, p) * ufl.dx
    - (1/dt) * beta_e * M * ufl.inner(q, T) * ufl.dx
    + kappa * M * ufl.inner(ufl.grad(q), ufl.grad(p)) * ufl.dx
    + kTp * M * ufl.inner(ufl.grad(q), ufl.grad(T)) * ufl.dx

    # Temperature equation
    + (1/dt) * ufl.inner(v, T) * ufl.dx
    - (1/dt) * (beta_e / md) * ufl.inner(v, p) * ufl.dx
    + kappa_T * ufl.inner(ufl.grad(v), ufl.grad(T)) * ufl.dx
    + kappaTp * ufl.inner(ufl.grad(v), ufl.grad(p)) * ufl.dx
)

# Linear form
L = (
    # Pressure equation
    (1/dt) * ufl.inner(q, p0) * ufl.dx
    - (1/dt) * beta_e * M * ufl.inner(q, T0) * ufl.dx

    # Temperature equation
    + (1/dt) * ufl.inner(v, T0) * ufl.dx
    - (1/dt) * (beta_e / md) * ufl.inner(v, p0) * ufl.dx
)


# ---------------------- Boundary Conditions -----------------------------

# left/right facets: Delta p = 0, Delta T = 0
def left_right_marker(x):
    return np.isclose(x[0], -length/2) | np.isclose(x[0], length/2)

# top/bottom facets : Delta T = 0, Neumann on Delta p
def top_bottom_marker(x):
    return np.isclose(x[1], 0) | np.isclose(x[1], -depth)

# Locate boundary facets
lr_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left_right_marker)
tb_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, top_bottom_marker)
# all_boundary_facets = mesh.exterior_facet_indices(domain.topology)
# remaining_facets = np.setdiff1d(all_boundary_facets, np.union1d(tb_facets, lr_facets))

# Apply Dirichlet boundary conditions

Wp = W.sub(0)  # Pressure subspace
WT = W.sub(1)  # Temperature subspace

# Get collapsed spaces
Vp, p_map = Wp.collapse()
VT, T_map = WT.collapse()

# Locate DOFs these return tuples of (space_dofs, subspace_dofs)
p_lr_dofs = fem.locate_dofs_topological((Wp, Vp), domain.topology.dim - 1, lr_facets)
T_lr_dofs = fem.locate_dofs_topological((WT, VT), domain.topology.dim - 1, lr_facets)

p_tb_dofs = fem.locate_dofs_topological((Wp, Vp), domain.topology.dim - 1, tb_facets)
T_tb_dofs = fem.locate_dofs_topological((WT, VT), domain.topology.dim - 1, tb_facets)

bcs = []
# new_bc = dolfinx.fem.dirichletbc(u_D, combined_dofs, W0)
# new_wh = dolfinx.fem.Function(W)
# new_bc.set(new_wh.x.array)
# new_wh.x.scatter_forward()

zero_value = fem.Constant(domain, dolfinx.default_scalar_type(0.0))
T_tb_bc = fem.dirichletbc(zero_value, T_tb_dofs[0], WT)  # Use parent DOFs
T_lr_bc = fem.dirichletbc(zero_value, T_lr_dofs[0], WT)  # Use parent DOFs
p_lr_bc = fem.dirichletbc(zero_value, p_lr_dofs[0], Wp)  # Use parent DOFs

bcs.extend([p_lr_bc, T_lr_bc, T_tb_bc])


geom_dtype = domain.geometry.x.dtype

if domain.comm.rank == 0:
    inj_point = np.array(x_inj, dtype=geom_dtype)
    ext_point = np.array(x_ext, dtype=geom_dtype)
else:
    inj_point = np.zeros((0, 2), dtype=geom_dtype)
    ext_point = np.zeros((0, 2), dtype=geom_dtype)


# Next, we create the point source object and apply it to the right hand side vector.

def temperature_value(depth):
    """Calculate temperature based on depth."""
    if depth < 0:
        return T_top  # Top boundary condition
    else:
        return T_top + (T_bot - T_top)*depth/6000  # 200°C in Kelvin

mag_Tinj = 20 - temperature_value(-x_inj[1])
mag_pinj = p_inj - p_top
mag_pext = p_ext - p_top

T_inj_ps = scifem.PointSource(WT, inj_point, magnitude=mag_Tinj)
p_inj_ps = scifem.PointSource(Wp, inj_point, magnitude=mag_pinj)
p_ext_ps = scifem.PointSource(Wp, ext_point, magnitude=mag_pext)

T_inj_ps.apply_to_vector(w0)
p_inj_ps.apply_to_vector(w0)
p_ext_ps.apply_to_vector(w0)


# Create the linear problem
problem = LinearProblem(
    a, L, bcs=bcs, 
    petsc_options={
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 1000,
        "ksp_monitor": None
    }
)

# ---------------------- Time Stepping -----------------------------------
num_steps = 10
output_interval = 1

# Create output file
with io.XDMFFile(MPI.COMM_WORLD, "fenicsx_48.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    
    # Initial state
    p_out, T_out = w0.sub(0).collapse(), w0.sub(1).collapse()
    p_out.name = "Pressure"
    T_out.name = "Temperature"
    xdmf.write_function(p_out, 0.0)
    xdmf.write_function(T_out, 0.0)
    
    for step in range(num_steps):
        print(f"Step {step+1}/{num_steps}")
        
        try:
            # Solve the system
            w = problem.solve()
            
            # Update for next time step
            w0.x.array[:] = w.x.array[:]
            
            # Output results at specified intervals
            if (step + 1) % output_interval == 0:
                p_out, T_out = w0.sub(0).collapse(), w0.sub(1).collapse()
                p_out.name = "Pressure"
                T_out.name = "Temperature"
                time = (step + 1) * timestep
                xdmf.write_function(p_out, time)
                xdmf.write_function(T_out, time)
                
                # Print some statistics
                p_max = np.max(p_out.x.array)
                p_min = np.min(p_out.x.array)
                T_max = np.max(T_out.x.array)
                T_min = np.min(T_out.x.array)
                print(f"  Pressure range: [{p_min/1e6:.2f}, {p_max/1e6:.2f}] MPa")
                print(f"  Temperature range: [{T_min:.2f}, {T_max:.2f}] °C")
                
        except Exception as e:
            print(f"Error at step {step+1}: {e}")
            break

print("Simulation completed!")