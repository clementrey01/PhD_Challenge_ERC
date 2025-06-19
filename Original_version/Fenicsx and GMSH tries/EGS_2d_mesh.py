import gmsh
import sys # To save the mesh

def EGS_2D(model_name="EGS_2D"):

    # Initialization
    model = gmsh.model()
    # Define the name of the model
    model.add(model_name)


    ## ---------- GEOMETRY PARAMETERS (in km) -----------

    # Simulation box
    L = 3000.0  # Total width
    H = 6000.0  # Total height
    z_top = 0.0 # Top boundary
    z_bottom = z_top - H # Bottom boundary

    # Wells
    d = 5000.0 # Depth of the wells
    x_inj = 250.0 # Center of the injection well
    x_ext = -250.0 # Center of the extraction well
    z_well = z_top - d  # Bottom of wells


    ## ---------- MESH PARAMETERS -----------

    # Mesh size is controlled at the points
    lc_box = 200.0 # Mesh size on domain boundary, the coarser
    lc_well_bottom = 10.0 # Mesh size at the bottom of the wells, the finer


    ## ---------- CASCADE - GEOMETRY CONSTRUCTION -----------

    # POINTS (starting from lower left corner to stay consistent with gmsh)

    # Domain corners
    p1 = model.occ.addPoint(-L/2, z_bottom, 0, meshSize=lc_box)
    p2 = model.occ.addPoint(-L/2, z_top, 0, meshSize=lc_box)
    p3 = model.occ.addPoint(L/2, z_top, 0, meshSize=lc_box)
    p4 = model.occ.addPoint(L/2, z_bottom, 0, meshSize=lc_box)

    p5 = model.occ.addPoint(x_inj, z_well, 0, meshSize=lc_well_bottom)
    p6 = model.occ.addPoint(x_ext, z_well, 0, meshSize=lc_well_bottom)



    # LINES (starting from lower left corner)

    l1 = model.occ.addLine(p1, p2)
    l2 = model.occ.addLine(p2, p3)
    l3 = model.occ.addLine(p3, p4)
    l4 = model.occ.addLine(p4, p1)


    # SURFACE OF THE DOMAIN

    domain_loop = model.occ.addCurveLoop([l1, l2, l3, l4])

    domain_surf = model.occ.addPlaneSurface([domain_loop])

    ## --------- GMSH - PHYSICAL GROUPS -----------

    # Convert Cascade object into gmsh object
    model.occ.synchronize()

    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "NodesList", [p5, p6])

    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "SizeMin", lc_well_bottom)     # min element size near point
    model.mesh.field.setNumber(2, "SizeMax", lc_box)      # max size far away
    model.mesh.field.setNumber(2, "DistMin", 10)      # start refining within this distance
    model.mesh.field.setNumber(2, "DistMax", 2000)      # stop refining beyond this distance

    # Physical Surface
    model.addPhysicalGroup(2, [domain_surf], tag=1)
    model.setPhysicalName(2, 1, "Domain")

    # Physical Lines
    model.addPhysicalGroup(1, [l2], tag=2)
    model.setPhysicalName(1, 2, "TopLine")

    model.addPhysicalGroup(1, [l4], tag=3)
    model.setPhysicalName(1, 3, "BottomLine")

    # Both lateral lines are joined because
    # they have the same boundary conditions
    model.addPhysicalGroup(1, [l1, l3], tag=4)
    model.setPhysicalName(1, 4, "LateralLines")

    # Wells (point sources)
    # Points seem to not be exported as physical groups
    model.addPhysicalGroup(0, [p5], tag=5)
    model.setPhysicalName(0, 5, "InjectionWell")

    model.addPhysicalGroup(0, [p6], tag=6)
    model.setPhysicalName(0, 6, "ExtractionWell")


    ## ---------- MESH + EXPORT ----------
    
    # Activate mesh refinement field
    model.mesh.field.setAsBackgroundMesh(2)
    model.occ.synchronize()

    # Generate a 2D mesh
    model.mesh.generate(2)

    return model


model_name = "EGS_2D"
gmsh.initialize()
_ = EGS_2D()
gmsh.write(model_name + ".msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()