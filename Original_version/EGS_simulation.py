"""
Coupled Thermo-Hydro-Mechanical Diffusion Equations

This solves the coupled system:
∂T/∂t - κ_T ∇²T = (m_d/β_e) ∂p/∂t + κ_Tp ∇²p + S_T
∂p/∂t - κ ∇²p = β_e M ∂T/∂t + k_Tp ∇²T + S_p

with the theta method (Crank-Nicolson).

The code is inspired from 
https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex19.py
"""

from math import ceil
from typing import Iterator, Tuple

import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix

from skfem import MeshTri, Basis, ElementTriP2, asm, penalize
from skfem.models.poisson import laplace, mass

# ----------- FLOATING POINT PRECISION -------------

# Python uses float64 by default, which corresponds to FEM simulations


# ---------- NUMERICAL PARAMETERS -------------------

# Due to diffusion length L = √(κt) ~ 70m after one year
# I implement a buffer zone of 100m away from the wells
mesh_width = 700.0    # x-direction [m]
mesh_height = 200.0   # y-direction [m]

# Wells are centered in the domain
inj_well = np.array([250.0, 0.0])   # injection well (x, y)
ext_well = np.array([-250.0, 0.0])  # extraction well (x, y)

# Timestep and theta method
dt = 3600*24 # 1 day
theta = 0.5  # Crank-Nicolson
max_time = 3600*24*365 # 1 year

# ------- PHYSICAL PARAMETERS (SI units) -------------

# Material properties
M = 1.0             # Biot modulus [Pa]
m_d = 6.0e-10       # Specific constrained storage [Pa^-1]
kappa = 1.0e-6      # Hydraulic diffusivity [m²/s]
kappa_T = 1.3e-6    # Thermal diffusivity [m²/s]
beta_e = 4.25e-5    # Undrained volumetric thermal expansion [K^-1]
k_Tp = 1.0e-9       # Mechano-caloric coefficient [m² s^-1 Pa K^-1]
kappa_Tp = 4.0e-11  # Thermo-osmosis coefficient [m² s^-1 K Pa^-1]

# System properties
p_inj = 22e6         # Injection pressure [Pa]
p_ext = -18e6        # Extraction pressure [Pa]
T_inj = 20           # Injection temperature [C]
T_surf = 13          # Surface temperature [C]
T_bot = 200          # Bottom temperature at 6km depth [C]
T0 = 170             # Temperature at 5.1km depth [C]
well_depth = 5000.0  # Depth of the wells [m]

# Characteristic scales
Lc = 500.0                      # Distance between wells [m]
tc = Lc**2 / kappa              # Characteristic time scale of pressure diffusion
Deltapc = np.abs(p_inj - p_ext) # Pressure difference between wells [Pa]
DeltaTc = np.abs(T0 - T_inj)    # Temperature difference between injection well and bottom [C]


# ------- DIMENSIONLESS PARAMETERS (SI units) -------------

epsilon = kappa_T / kappa                   # Relative diffusivity coefficient
chi = beta_e * M * DeltaTc / Deltapc        # Scaled thermal pressurization coefficient
alpha = k_Tp / kappa * DeltaTc/ Deltapc     # Scaled thermo-osmotic coupling ratio
gamma = m_d / beta_e * Deltapc / DeltaTc    # Scaled poro-thermal coupling ratio
beta = kappa_Tp / kappa * Deltapc / DeltaTc # Scaled thermo-osmotic cross-diffusion ratio

# Dimensionless source values
p_inj_scaled = p_inj / Deltapc
p_ext_scaled = p_ext / Deltapc
T_inj_scaled = (T_inj - T0*5/6) / DeltaTc

# Dimensionless domain size
mesh_width_scaled = mesh_width / Lc
mesh_height_scaled = mesh_height / Lc
inj_well_scaled = inj_well / Lc
ext_well_scaled = ext_well / Lc

# Dimensionless time
dt_scaled = dt / tc
max_time_scaled = max_time / tc

# ---------- MESH DEFINITION -------------------

# I have a GMSH code refined around the wells
# But the mesh is too big for my computer for some reason

# Number of elements on the x axis
nb_elements = 2**5 # Maximum on my computer

# Create a grid
x_coords = np.linspace(-mesh_width_scaled/2, mesh_width_scaled/2, nb_elements)
y_coords = np.linspace(-mesh_height_scaled/2, mesh_height_scaled/2,
                       nb_elements * ceil(mesh_height_scaled / mesh_width_scaled))

# Create a triangular mesh
mesh = MeshTri.init_tensor(x_coords, y_coords)

# Create basis functions
element = ElementTriP2() # 2nd order for increased stability
basisT = Basis(mesh, element)  # Temperature basis
basisp = Basis(mesh, element)  # Pressure basis

print(f"Number of DOFs per field: {basisT.N}")
print(f"Total DOFs: {2 * basisT.N}")


# -------------- BOUNDARY CONDITIONS ---------------------

# I follow Gutiérrez and Stefanou : Drained boundary conditions for pressure
# And homogeneous Dirichlet for temperature on top and bottom to stay physically consistent

# Find nodes closest to wells
# to set up the Dirichlet conditions
def find_closest_node(mesh, point):
    distances = np.linalg.norm(mesh.p.T - point, axis=1)
    return np.argmin(distances)

inj_node = find_closest_node(mesh, inj_well_scaled)
ext_node = find_closest_node(mesh, ext_well_scaled)

# Temperature: homogeneous Dirichlet on top and bottom
boundary_dofs_T_bottom = basisT.get_dofs(lambda x: np.isclose(x[1], -mesh_height_scaled/2))
boundary_dofs_T_top= basisT.get_dofs(lambda x: np.isclose(x[1], mesh_height_scaled/2))
# Combine the dofs
bc_dofs_T = np.unique(np.hstack((boundary_dofs_T_bottom, boundary_dofs_T_top)))

# Pressure: homogeneous Neumann everywhere
# Directly handled by the variational formulation so no need to set them

# POINT SOURCE VECTORS
# CAUSE SOME SORT OF INSTABILITY
# # Create point source vectors
# b_T_inj = basisT.point_source(mesh.p[:,inj_node]) * T_inj_scaled
# b_p_inj = basisp.point_source(mesh.p[:,inj_node]) * p_inj_scaled
# b_p_ext = basisp.point_source(mesh.p[:,ext_node]) * p_ext_scaled

# # Total source vectors
# b_T_total = b_T_inj
# b_p_total = b_p_inj + b_p_ext
# # Create combined source vector
# b_combined = np.concatenate([b_T_total, b_p_total])


# --------- GRADIENT CAPPING TO LIMIT INSTABILITY ----------------

def cap_gradient_simple(u, basisT, basisp, max_grad_T=1.0, max_grad_p=1.0):
    """
    Simple gradient capping for temperature and pressure fields
    """
    u_T = u[:basisT.N].copy()
    u_p = u[basisT.N:].copy()
    
    # Cap temperature gradients
    mesh = basisT.mesh
    
    # Loop through elements
    for elem_idx in range(mesh.t.shape[1]):
        nodes = mesh.t[:, elem_idx]
        coords = mesh.p[:, nodes]
        T_vals = u_T[nodes]
        
        # Simple gradient estimate: max difference / min edge length
        max_diff = np.max(T_vals) - np.min(T_vals)
        edges = np.diff(coords, axis=1)
        min_edge = np.min([np.linalg.norm(edge) for edge in edges.T])
        
        if min_edge > 1e-12:
            grad_est = max_diff / min_edge
            
            if grad_est > max_grad_T:
                # Scale values around mean
                T_mean = np.mean(T_vals)
                scale = max_grad_T / grad_est
                u_T[nodes] = T_mean + scale * (T_vals - T_mean)
    
    # Cap pressure gradients
    mesh_p = basisp.mesh
    for elem_idx in range(mesh_p.t.shape[1]):
        nodes = mesh_p.t[:, elem_idx]
        coords = mesh_p.p[:, nodes]
        p_vals = u_p[nodes]
        
        # Simple gradient estimate
        max_diff = np.max(p_vals) - np.min(p_vals)
        edges = np.diff(coords, axis=1)
        min_edge = np.min([np.linalg.norm(edge) for edge in edges.T])
        
        if min_edge > 1e-12:
            grad_est = max_diff / min_edge
            
            if grad_est > max_grad_p:
                # Scale values around mean
                p_mean = np.mean(p_vals)
                scale = max_grad_p / grad_est
                u_p[nodes] = p_mean + scale * (p_vals - p_mean)
    
    # Return modified solution
    u_new = u.copy()
    u_new[:basisT.N] = u_T
    u_new[basisT.N:] = u_p
    return u_new


# --------------------- INITIAL CONDITIONS -------------------

# Since we work with ΔT and Δp => zero everywhere (but wells)

# Initial conditions - zero everywhere
u_T_init = np.zeros(basisT.N) 
u_p_init = np.zeros(basisp.N)

# Set initial values at wells
u_T_init[inj_node] = T_inj_scaled
u_p_init[inj_node] = p_inj_scaled  
u_p_init[ext_node] = p_ext_scaled

# Combined initial state vector
u_init = np.concatenate([u_T_init, u_p_init])

u_init = cap_gradient_simple(u_init, basisT, basisp, 10, 1)

print(f"Initial state vector size: {len(u_init)}")
print(f"Initial T range: [{np.min(u_T_init):.3f}, {np.max(u_T_init):.3f}]")
print(f"Initial p range: [{np.min(u_p_init):.3f}, {np.max(u_p_init):.3f}]")
print(f"Values at wells: T_inj={u_T_init[inj_node]:.3f}, p_inj={u_p_init[inj_node]:.3f}, T_inj={u_T_init[ext_node]:.3f}, p_ext={u_p_init[ext_node]:.3f}")


# -------------- VARIATIONAL FORMULATION ---------------------

# I use the bilinear forms Mass and Laplace
# Mass : M(v) = ∫ u v dΩ with v test function
# Laplace : L(v) = ∫ ∇u ∇v dΩ with v test function

# Structure of the bilinear forms :
# asm(form, trial basis, test basis). Since the bilinear forms are SYMMETRIC,
# asm(form, test basis, trial basis) works too.
# If the test and trial bases are the same, one can write :
# asm(form, basis)

# Assemble matrices for individual fields

# Temperature matrices
L_T = epsilon * asm(laplace, basisT)
M_T = asm(mass, basisT)

# Pressure matrices  
L_p = asm(laplace, basisp)
M_p = asm(mass, basisp)

# Cross-coupling matrices
# T equation
L_Tp = beta * asm(laplace, basisT, basisp)  # T test, p trial
M_Tp = gamma * asm(mass, basisT, basisp)  # T test, p trial

# p equation
L_pT = alpha * asm(laplace, basisp, basisT)  # p test, T trial
M_pT = chi * asm(mass, basisp, basisT)   # p test, T trial


# Apply boundary conditions using penalization
L0_T, M0_T = penalize(L_T, M_T, D=bc_dofs_T)
# L0_Tp, M0_Tp = penalize(L_Tp, M_Tp, D=bc_dofs_T)
L0_Tp = L_Tp.copy()
M0_Tp = M_Tp.copy()
L0_Tp[bc_dofs_T, :] = 0
M0_Tp[bc_dofs_T, :] = 0

# Pressure : no boundary
L0_p, M0_p = L_p, M_p #penalize(L_p, M_p, D=boundary_dofs_p)
L0_pT, M0_pT = L_pT, M_pT #penalize(L_pT, M_pT, D=boundary_dofs_p)


# -------------- SYSTEM MATRICES ---------------------

# Block matrices method
# I could use a conditioner

# Left-hand side matrix A = [[A_TT, A_Tp], [A_pT, A_pp]]
A_TT = M0_T + dt_scaled * theta * L0_T
A_Tp = -M0_Tp + dt_scaled * theta * L0_Tp
A_pT = -M0_pT + dt_scaled * theta * L0_pT  
A_pp = M0_p + dt_scaled * theta * L0_p

# Right-hand side matrix B = [[B_TT, B_Tp], [B_pT, B_pp]]
B_TT = M0_T - dt_scaled * (1 - theta) * L0_T
B_Tp = -M0_Tp - dt_scaled * (1 - theta) * L0_Tp
B_pT = -M0_pT - dt_scaled * (1 - theta) * L0_pT
B_pp = M0_p - dt_scaled * (1 - theta) * L0_p


# Convert to dense for block assembly
def to_dense(mat):
    if hasattr(mat, 'toarray'):
        return mat.toarray()
    return mat

A_TT_dense = to_dense(A_TT)
A_Tp_dense = to_dense(A_Tp) 
A_pT_dense = to_dense(A_pT)
A_pp_dense = to_dense(A_pp)

B_TT_dense = to_dense(B_TT)
B_Tp_dense = to_dense(B_Tp)
B_pT_dense = to_dense(B_pT) 
B_pp_dense = to_dense(B_pp)


# Assemble block matrices
# Coupling matrices

A_block = np.block([[A_TT_dense, A_Tp_dense],
                    [A_pT_dense, A_pp_dense]])

B_block = np.block([[B_TT_dense, B_Tp_dense],
                    [B_pT_dense, B_pp_dense]])

# Convert A to sparse and factor once for all
A_sparse = csc_matrix(A_block)
backsolve = splu(A_sparse).solve

# Time evolution iterator
def evolve(t: float, u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:

    step_count = 0
    
    while t < max_time_scaled and step_count < 10000:  # Limit steps to avoid infinite loop

        # # Right-hand side with point sources
        # rhs = B_block @ u + dt_scaled * b_combined

        rhs = B_block @ u 

        # Compute new solution 
        u_new = backsolve(rhs)
        t = t + dt_scaled
        step_count += 1
        
        # Enforce boundary conditions
        u_new[:basisT.N][bc_dofs_T] = 0.0  # Temperature BC
        
        # Maintain well source values
        u_new[inj_node] = T_inj_scaled  # Temperature at injection well
        u_new[basisT.N + inj_node] = p_inj_scaled  # Pressure at injection well
        u_new[basisT.N + ext_node] = p_ext_scaled  # Pressure at extraction well
    
        u_new = cap_gradient_simple(u_new, basisT, basisp, 10, 1)

        # Enforce boundary conditions
        u_new[:basisT.N][bc_dofs_T] = 0.0  # Temperature BC
        
        # Maintain well source values
        u_new[inj_node] = T_inj_scaled  # Temperature at injection well
        u_new[basisT.N + inj_node] = p_inj_scaled  # Pressure at injection well
        u_new[basisT.N + ext_node] = p_ext_scaled  # Pressure at extraction well

        yield t, u_new
        u = u_new


# ------------------ UTILITY FUNCTIONS ------------------------

def dimensionalize_and_add_background(u_T_nd, u_p_nd, mesh):
    """
    Convert nondimensional fields to physical fields with background
    """
    # Dimensionalize the perturbation fields (these are already at nodal DOFs)
    u_T_physical = u_T_nd[basisT.nodal_dofs.flatten()] * DeltaTc
    u_p_physical = u_p_nd[basisp.nodal_dofs.flatten()] * Deltapc / 1e6
    
    # Get y-coordinates at temperature nodal DOFs only
    nodal_indices = basisT.nodal_dofs.flatten()
    y_coords_nodal = mesh.p[1, nodal_indices]
    
    # Add geothermal background to temperature
    G = (T_bot - T_surf)/6000
    T_background = T_surf - G * (y_coords_nodal - well_depth)
    u_T_total = u_T_physical + T_background
    
    # Pressure background is zero
    u_p_total = u_p_physical
    
    return u_T_total, u_p_total

def extract_fields(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract temperature and pressure from combined solution vector"""
    n_dofs = len(u) // 2
    u_T = u[:n_dofs]
    u_p = u[n_dofs:]
    return u_T, u_p

# Probe points for monitoring
probe_center_T = basisT.probes(np.zeros((mesh.dim(), 1)))
probe_center_p = basisp.probes(np.zeros((mesh.dim(), 1)))
probe_inj_T = basisT.probes(mesh.p[:,inj_node].reshape(-1, 1))
probe_inj_p = basisp.probes(mesh.p[:,inj_node].reshape(-1, 1))
probe_ext_p = basisp.probes(mesh.p[:,ext_node].reshape(-1, 1))

def probe_locations(u: np.ndarray) -> dict:
    """Evaluate both fields at key locations"""
    u_T, u_p = extract_fields(u)
    
    return {
        'T_center': (probe_center_T @ u_T)[0],
        'p_center': (probe_center_p @ u_p)[0],
        'T_inj': (probe_inj_T @ u_T)[0],
        'p_inj': (probe_inj_p @ u_p)[0],
        'p_ext': (probe_ext_p @ u_p)[0],
        'T_max': np.max(np.abs(u_T)),
        'p_max': np.max(np.abs(u_p))
    }

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from skfem.visuals.matplotlib import plot
    import numpy as np

    # Run initial analysis
    print("\n" + "="*50)
    print("INITIAL ANALYSIS")
    print("="*50)
    
    probes_init = probe_locations(u_init)
    
    print("Initial state (nondimensional):")
    print(f"  Center: T={probes_init['T_center']:.3f}, p={probes_init['p_center']:.3f}")
    print(f"  Wells: T_inj={probes_init['T_inj']:.3f}, p_inj={probes_init['p_inj']:.3f}, p_ext={probes_init['p_ext']:.3f}")
    print(f"  Max values: T={probes_init['T_max']:.3f}, p={probes_init['p_max']:.3f}")

    # Collect all simulation data first
    print("Collecting simulation data...")
    evolution_data = []
    probe_evolution = {
        'time': [],
        'T_center': [],
        'T_inj': [],
        'p_center': [],
        'p_inj': [],
        'p_ext': [],
        'T_max': [],
        'p_max': []
    }
    
    for i, (t, u) in enumerate(evolve(0., u_init)):
        evolution_data.append((t, u.copy()))
        
        # Store probe data
        probes = probe_locations(u)
        probe_evolution['time'].append(t * tc / (3600 * 24))  # Convert to days
        probe_evolution['T_center'].append(probes['T_center'])
        probe_evolution['T_inj'].append(probes['T_inj'])
        probe_evolution['p_center'].append(probes['p_center'])
        probe_evolution['p_inj'].append(probes['p_inj'])
        probe_evolution['p_ext'].append(probes['p_ext'])
        probe_evolution['T_max'].append(probes['T_max'])
        probe_evolution['p_max'].append(probes['p_max'])
        
        if (i + 1) % 50 == 0:
            print(f"  Collected {i + 1} timesteps (t={t*tc/(3600*24):.2f} days)")
    
    print(f"Generated {len(evolution_data)} frames for visualization")

    # Interactive visualization class
    class InteractiveSimulationViewer:
        def __init__(self, evolution_data):
            self.evolution_data = evolution_data
            self.current_frame = 0
            self.playing = False
            self.auto_step_interval = 0
            
            # Setup the plot with correct aspect ratio
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
            self.setup_dual_plot()
            
            # Add control buttons
            self.setup_controls()
            
            # Initial display
            self.update_display()
            
        def setup_dual_plot(self):
            """Setup dual field visualization"""
            u_T_init_viz_, u_p_init_viz_ = extract_fields(self.evolution_data[0][1])
            
            u_T_init_viz, u_p_init_viz = dimensionalize_and_add_background(u_T_init_viz_, 
                                                                           u_p_init_viz_, mesh)

            # Temperature subplot
            self.temp_plot = plot(mesh, u_T_init_viz[basisT.nodal_dofs.flatten()], 
                        shading='gouraud', cmap='viridis', ax=self.ax1)
            self.ax1.set_title('Temperature at t = 0.00')
            self.ax1.set_aspect('equal')
            self.temp_cbar = self.fig.colorbar(self.ax1.get_children()[0], ax=self.ax1, label='Temperature (C)')
            
            # Pressure subplot  
            self.pres_plot = plot(mesh, u_p_init_viz[basisp.nodal_dofs.flatten()],
                        shading='gouraud', cmap='viridis', ax=self.ax2)
            self.ax2.set_title('Pressure at t = 0.00')
            self.ax2.set_aspect('equal')
            self.pres_cbar = self.fig.colorbar(self.ax2.get_children()[0], ax=self.ax2, label='Pressure (MPa)')

        def setup_controls(self):
            """Setup control buttons"""
            plt.subplots_adjust(bottom=0.15)
            
            # Button positions
            button_width = 0.08
            button_height = 0.04
            button_y = 0.02
            
            # Previous button
            ax_prev = plt.axes([0.1, button_y, button_width, button_height])
            self.btn_prev = Button(ax_prev, '← Prev')
            self.btn_prev.on_clicked(self.prev_frame)
            
            # Play/Pause button
            ax_play = plt.axes([0.2, button_y, button_width, button_height])
            self.btn_play = Button(ax_play, 'Play')
            self.btn_play.on_clicked(self.toggle_play)
            
            # Next button
            ax_next = plt.axes([0.3, button_y, button_width, button_height])
            self.btn_next = Button(ax_next, 'Next →')
            self.btn_next.on_clicked(self.next_frame)
            
            # Save button
            ax_save = plt.axes([0.4, button_y, button_width, button_height])
            self.btn_save = Button(ax_save, 'Save')
            self.btn_save.on_clicked(self.save_current_frame)
            
            # Jump to frame input
            ax_jump = plt.axes([0.5, button_y, 0.15, button_height])
            from matplotlib.widgets import TextBox
            self.txt_jump = TextBox(ax_jump, 'Frame: ', initial='0')
            self.txt_jump.on_submit(self.jump_to_frame)
            
            # Frame info
            ax_info = plt.axes([0.7, button_y, 0.25, button_height])
            ax_info.set_xlim(0, 1)
            ax_info.set_ylim(0, 1)
            ax_info.axis('off')
            self.info_text = ax_info.text(0, 0.5, '', fontsize=10, va='center')
            
        def update_display(self):
            """Update the visualization for current frame"""
            t, u = self.evolution_data[self.current_frame]
            u_T_nd, u_p_nd = extract_fields(u)
            u_T_d, u_p_d = dimensionalize_and_add_background(u_T_nd, u_p_nd, mesh)
            
            time_days = t * tc / (3600 * 24)

            # Update temperature plot
            u_T_viz = u_T_d[basisT.nodal_dofs.flatten()]
            self.ax1.get_children()[0].set_array(u_T_viz)
            self.ax1.set_title(f'Temperature (C) at t = {time_days:.1f} days')
            
            # Update pressure plot
            u_p_viz = u_p_d[basisp.nodal_dofs.flatten()]  
            self.ax2.get_children()[0].set_array(u_p_viz)
            self.ax2.set_title(f'Pressure (MPa) at t = {time_days:.1f} days')
            
            # Update info text
            self.info_text.set_text(f'Frame {self.current_frame}/{len(self.evolution_data)-1}\n'
                                   f't = {time_days:.2f} days')
            
            # Print probe analysis
            probes = probe_locations(u)
            print(f'Frame {self.current_frame}, t={time_days:.1f} days: '
                  f'T_max={probes["T_max"]:.3f}, p_max={probes["p_max"]:.3f}, '
                  f'T_center={probes["T_center"]:.3f}, p_center={probes["p_center"]:.3f}')
            
            self.fig.canvas.draw()
            
        def prev_frame(self, event):
            """Go to previous frame"""
            if self.current_frame > 0:
                self.current_frame -= 1
                self.update_display()
                
        def next_frame(self, event):
            """Go to next frame"""
            if self.current_frame < len(self.evolution_data) - 1:
                self.current_frame += 1
                self.update_display()
                
        def toggle_play(self, event):
            """Toggle play/pause"""
            self.playing = not self.playing
            self.btn_play.label.set_text('Pause' if self.playing else 'Play')
            
            if self.playing:
                self.play_animation()
                
        def play_animation(self):
            """Play animation automatically"""
            while self.playing and self.current_frame < len(self.evolution_data) - 1:
                self.current_frame += 1
                self.update_display()
                
                if self.auto_step_interval > 0:
                    plt.pause(self.auto_step_interval)
                else:
                    plt.pause(0.1)  # Default pause
                    
                # Check if window is still open
                if not plt.get_fignums():
                    break
                    
            self.playing = False
            self.btn_play.label.set_text('Play')
            
        def jump_to_frame(self, text):
            """Jump to specific frame"""
            try:
                frame_num = int(text)
                if 0 <= frame_num < len(self.evolution_data):
                    self.current_frame = frame_num
                    self.update_display()
                else:
                    print(f"Frame {frame_num} out of range [0, {len(self.evolution_data)-1}]")
            except ValueError:
                print("Invalid frame number")
                
        def save_current_frame(self, event):
            """Save current frame as image"""
            t, _= self.evolution_data[self.current_frame]
            time_days = t * tc / (3600 * 24)
            
            filename = f'coupled_diffusion_dual_frame_{self.current_frame:04d}_t_{time_days:.1f}days.png'
            
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")

    # Create interactive viewer
    viewer = InteractiveSimulationViewer(evolution_data)
    
    print("\n" + "="*50)
    print("INTERACTIVE CONTROLS:")
    print("="*50)
    print("← Prev  : Previous frame")
    print("Play    : Play/Pause animation") 
    print("Next →  : Next frame")
    print("Save    : Save current frame as PNG")
    print("Frame   : Jump to specific frame number")
    print("Close window to exit")
    print("="*50)
    
    # Show interactive plot
    plt.show()
    
    # After interactive plot is closed, show probe evolution
    print("\nCreating probe evolution plots...")
    
    fig_probes, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature evolution
    ax1.plot(probe_evolution['time'], probe_evolution['T_center'], 'b-', label='Center', linewidth=2)
    ax1.plot(probe_evolution['time'], probe_evolution['T_inj'], 'r-', label='Injection well', linewidth=2)
    ax1.plot(probe_evolution['time'], probe_evolution['T_max'], 'g--', label='Max |T|', linewidth=1)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Temperature perturbation (nondim)')
    ax1.set_title('Temperature Evolution at Probe Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pressure evolution
    ax2.plot(probe_evolution['time'], probe_evolution['p_center'], 'b-', label='Center', linewidth=2)
    ax2.plot(probe_evolution['time'], probe_evolution['p_inj'], 'r-', label='Injection well', linewidth=2)
    ax2.plot(probe_evolution['time'], probe_evolution['p_ext'], 'm-', label='Extraction well', linewidth=2)
    ax2.plot(probe_evolution['time'], probe_evolution['p_max'], 'g--', label='Max |p|', linewidth=1)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Pressure perturbation (nondim)')
    ax2.set_title('Pressure Evolution at Probe Points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Temperature evolution (log scale)
    ax3.semilogy(probe_evolution['time'], np.abs(probe_evolution['T_center']), 'b-', label='|T| Center', linewidth=2)
    ax3.semilogy(probe_evolution['time'], np.abs(probe_evolution['T_inj']), 'r-', label='|T| Injection', linewidth=2)
    ax3.semilogy(probe_evolution['time'], probe_evolution['T_max'], 'g--', label='Max |T|', linewidth=1)
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('|Temperature perturbation| (nondim)')
    ax3.set_title('Temperature Evolution (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Pressure evolution (log scale)
    ax4.semilogy(probe_evolution['time'], np.abs(probe_evolution['p_center']), 'b-', label='|p| Center', linewidth=2)
    ax4.semilogy(probe_evolution['time'], np.abs(probe_evolution['p_inj']), 'r-', label='|p| Injection', linewidth=2)
    ax4.semilogy(probe_evolution['time'], np.abs(probe_evolution['p_ext']), 'm-', label='|p| Extraction', linewidth=2)
    ax4.semilogy(probe_evolution['time'], probe_evolution['p_max'], 'g--', label='Max |p|', linewidth=1)
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('|Pressure perturbation| (nondim)')
    ax4.set_title('Pressure Evolution (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save probe evolution plot
    probe_filename = 'probe_evolution.png'
    fig_probes.savefig(probe_filename, dpi=300, bbox_inches='tight')
    print(f"Probe evolution plot saved as {probe_filename}")
    
    plt.show()
    
    print("\nFinal probe values:")
    print(f"T_center: {probe_evolution['T_center'][-1]:.6f}")
    print(f"T_inj: {probe_evolution['T_inj'][-1]:.6f}")
    print(f"p_center: {probe_evolution['p_center'][-1]:.6f}")
    print(f"p_inj: {probe_evolution['p_inj'][-1]:.6f}")
    print(f"p_ext: {probe_evolution['p_ext'][-1]:.6f}")
