import numpy

# ------------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------------

# Input particle file to read
# input_file = "./snapshots/0.000halo511.dat"

# Simulation parameters (needed for unit conversions)
boxsize  = 500.0
ngrid    = 6144
Ncb      = 1536
Nnu      = 0
mnu_ev   = 0.0
hubble   = 0.6898
omega_m  = 0.2905
omega_nu = mnu_ev/93.14/(100.0*hubble)**2
omega_cb = omega_m - omega_nu
rho_crit = 2.77536627e11
mass_cb  = rho_crit*omega_cb*(boxsize/Ncb)**3
if Nnu > 0:
    mass_nu = rho_crit*omega_nu*(boxsize/Nnu)**3
else:
    mass_nu = 0.0
mass_p   = (ngrid/(1.0*Ncb))**3

# Flag to enable endian conversion of binary data. This is most likely needed since
# the simulations were performed on the BGQ which uses big-endian format while most
# other machines use little-endian. If the read returns garbage then toggle this flag
BYTESWAP = True

# ------------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------------

def Byteswap(x):
    """
    If the flag BYTESWAP is False then this does nothing. Otherwise, it performs an
    in-place endian swap of the numpy array x.
    """

    global BYTESWAP

    if BYTESWAP:
        return x.byteswap(inplace=True)
    else:
        return x

def PeriodicWrap(x):
    """
    Periodically wrap the positions around the global box.
    """

    global boxsize

    x[numpy.where(x<0.)] += boxsize
    x[numpy.where(x>=boxsize)] -= boxsize

    return x

def ConvertPositionUnits(xx, yy, zz):
    """
    Convert the position from code units to global comoving Mpc/h
    """

    global boxsize, ngrid

    # Convert to global comoving Mpc/h (rank coordinates are added in halo output)
    r_code2phys = boxsize/ngrid
    xx *= r_code2phys
    yy *= r_code2phys
    zz *= r_code2phys

    # Periodically wrap the positions around the global box
    xx = PeriodicWrap(xx)
    yy = PeriodicWrap(yy)
    zz = PeriodicWrap(zz)

    return xx, yy, zz

def ConvertVelocityUnits(vx, vy, vz, a):
    """
    Convert the velocity from code units to comoving km/s
    """

    global omega_m, boxsize, ngrid

    v_code2phys = 300.0*boxsize*numpy.sqrt(omega_m)/2.0/ngrid/a**2

    vx *= v_code2phys
    vy *= v_code2phys
    vz *= v_code2phys

    return vx, vy, vz

def ConvertLengthUnits(r):
    """
    Convert length value from code units to comoving Mpc/h
    """

    global boxsize, ngrid

    r_code2phys = boxsize/ngrid
    r *= r_code2phys

    return r



def ReadHaloFile_lcdm(df, a):
    """
    Read rank halo file
    """

    global mass_p, mass_cb, mass_nu, Nnu
    
    # Read depends on whether or not this is a simulation containing neutrino particles
    NEUTRINO_SIMULATION = True
    if Nnu == 0: NEUTRINO_SIMULATION = False

    if NEUTRINO_SIMULATION:
        hdtype = numpy.dtype([("xx", "float32"), ("yy", "float32"), ("zz", "float32"), \
                              ("mvir", "float32"), ("modc", "float32"), ("rvir", "float32"), ("rodc", "float32"), 
                              ("xxm", "float32"), ("yym", "float32"), ("zzm", "float32"), \
                              ("vxm", "float32"),  ("vym", "float32"), ("vzm", "float32"), \
                              ("lcmx", "float32"), ("lcmy", "float32"), ("lcmz", "float32"), \
                              ("v2x", "float32"), ("v2y", "float32"), ("v2z", "float32"), \
                              ("varx", "float32"), ("vary", "float32"), ("varz", "float32"), \
                              ("Ixx", "float32"), ("Ixy", "float32"), ("Ixz", "float32"), ("Iyy", "float32"), ("Iyz", "float32"), ("Izz", "float32"), \
                              ("xxm_nu", "float32"), ("yym_nu", "float32"), ("zzm_nu", "float32"), \
                              ("vxm_nu", "float32"), ("vym_nu", "float32"), ("vzm_nu", "float32"), \
                              ("n_nu", "int32")])
    else: # CDM-only simulation
        hdtype = numpy.dtype([("xx", "float32"), ("yy", "float32"), ("zz", "float32"), \
                              ("mvir", "float32"), ("modc", "float32"), ("rvir", "float32"), ("rodc", "float32"), 
                              ("xxm", "float32"), ("yym", "float32"), ("zzm", "float32"), \
                              ("vxm", "float32"),  ("vym", "float32"), ("vzm", "float32"), \
                              ("lcmx", "float32"), ("lcmy", "float32"), ("lcmz", "float32"), \
                              ("v2x", "float32"), ("v2y", "float32"), ("v2z", "float32"), \
                              ("varx", "float32"), ("vary", "float32"), ("varz", "float32")])

    fr = open(df, "rb")
    # Read the header
    nh = Byteswap(numpy.fromfile(fr, dtype="int32", count=1))[0] # Number of particles in this file
    halo_vir = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0] 
    halo_odc = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0]
    # Read the main body which contains various attributes for each halo
    data = Byteswap(numpy.fromfile(fr, dtype=hdtype, count=nh))
    fr.close()

    # Unpack important halo attributes
    xx = data["xx"]
    yy = data["yy"]
    zz = data["zz"]
    vx = data["vxm"]
    vy = data["vym"]
    vz = data["vzm"]
    mvir = data["mvir"]
    modc = data["modc"]
    # if NEUTRINO_SIMULATION: nnu = data["n_nu"]

    # Convert positions from code units to (global) comoving Mpc/h
    xx, yy, zz = ConvertPositionUnits(xx, yy, zz)
    vx, vy, vz = ConvertVelocityUnits(vx, vy, vz, a)

    # Convert radius from code units to comoving Mpc/h
    # rvir = ConvertLengthUnits(rvir)
    # rodc = ConvertLengthUnits(rodc)
    
    # Convert dm halo mass from code units to Msun/h
    mvir_dm = mvir/mass_p*mass_cb
    modc_dm = modc/mass_p*mass_cb

    # Compute mass of neutrinos within the search radius (set to 2 Mpc/h)
    # if NEUTRINO_SIMULATION: m2Mpc_nu = nnu*mass_nu 

    # Compute overdensity (for testing right now)
    # oden_vir = OverDensity(mvir_dm, rvir)
    # oden_odc = OverDensity(modc_dm, rodc)

    # print("xx: ", xx.min(), xx.max())
    # print("yy: ", yy.min(), yy.max())
    # print("zz: ", zz.min(), zz.max())
    # print("rvir: ", rvir.min(), rvir.max())
    # print("rodc: ", rodc.min(), rodc.max())
    # print("mvir_dm: ", mvir_dm.min(), mvir_dm.max())
    # print("modc_dm: ", modc_dm.min(), modc_dm.max())
    # if NEUTRINO_SIMULATION: print("m2Mpc_nu: ", m2Mpc_nu.min(), m2Mpc_nu.max())
    # print("halo_vir = ", halo_vir)
    # print("halo_odc = ", halo_odc)
    # print("oden_vir: ", oden_vir.min(), oden_vir.max())
    # print("oden_odc: ", oden_odc.min(), oden_odc.max())

    return [xx, yy, zz, vx, vy, vz, mvir_dm]


# def ReadHaloFile(df):
#     """
#     Read rank halo file
#     """

#     global mass_p, mass_cb, mass_nu, Nnu

#     # Read depends on whether or not this is a simulation containing neutrino particles
#     NEUTRINO_SIMULATION = True
#     if Nnu == 0: NEUTRINO_SIMULATION = False

#     if NEUTRINO_SIMULATION:
#         hdtype = numpy.dtype([("xx", "float32"), ("yy", "float32"), ("zz", "float32"), \
#                               ("mvir", "float32"), ("modc", "float32"), ("rvir", "float32"), ("rodc", "float32"), 
#                               ("xxm", "float32"), ("yym", "float32"), ("zzm", "float32"), \
#                               ("vxm", "float32"),  ("vym", "float32"), ("vzm", "float32"), \
#                               ("lcmx", "float32"), ("lcmy", "float32"), ("lcmz", "float32"), \
#                               ("v2x", "float32"), ("v2y", "float32"), ("v2z", "float32"), \
#                               ("varx", "float32"), ("vary", "float32"), ("varz", "float32"), \
#                               ("Ixx", "float32"), ("Ixy", "float32"), ("Ixz", "float32"), ("Iyy", "float32"), ("Iyz", "float32"), ("Izz", "float32"), \
#                               ("xxm_nu", "float32"), ("yym_nu", "float32"), ("zzm_nu", "float32"), \
#                               ("vxm_nu", "float32"), ("vym_nu", "float32"), ("vzm_nu", "float32"), \
#                               ("n_nu", "int32")])
#     else: # CDM-only simulation
#         hdtype = numpy.dtype([("xx", "float32"), ("yy", "float32"), ("zz", "float32"), \
#                               ("mvir", "float32"), ("modc", "float32"), ("rvir", "float32"), ("rodc", "float32"), 
#                               ("xxm", "float32"), ("yym", "float32"), ("zzm", "float32"), \
#                               ("vxm", "float32"),  ("vym", "float32"), ("vzm", "float32"), \
#                               ("lcmx", "float32"), ("lcmy", "float32"), ("lcmz", "float32"), \
#                               ("v2x", "float32"), ("v2y", "float32"), ("v2z", "float32"), \
#                               ("varx", "float32"), ("vary", "float32"), ("varz", "float32")])

#     fr = open(df, "rb")
#     # Read the header
#     nh = Byteswap(numpy.fromfile(fr, dtype="int32", count=1))[0] # Number of particles in this file
#     halo_vir = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0] 
#     halo_odc = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0]
#     # Read the main body which contains various attributes for each halo
#     data = Byteswap(numpy.fromfile(fr, dtype=hdtype, count=nh))
#     fr.close()

#     # Unpack important halo attributes
#     xx = data["xx"]
#     yy = data["yy"]
#     zz = data["zz"]
#     mvir = data["mvir"]
#     modc = data["modc"]
#     rvir = data["rvir"]
#     rodc = data["rodc"]
#     if NEUTRINO_SIMULATION: nnu = data["n_nu"]

#     # Convert positions from code units to (global) comoving Mpc/h
#     xx, yy, zz = ConvertPositionUnits(xx, yy, zz)

#     # Convert radius from code units to comoving Mpc/h
#     rvir = ConvertLengthUnits(rvir)
#     rodc = ConvertLengthUnits(rodc)

#     # Convert dm halo mass from code units to Msun/h
#     mvir_dm = mvir/mass_p*mass_cb
#     modc_dm = modc/mass_p*mass_cb

#     # Compute mass of neutrinos within the search radius (set to 2 Mpc/h)
#     if NEUTRINO_SIMULATION: m2Mpc_nu = nnu*mass_nu 

#     # Compute overdensity (for testing right now)
#     oden_vir = OverDensity(mvir_dm, rvir)
#     oden_odc = OverDensity(modc_dm, rodc)

#     print("xx: ", xx.min(), xx.max())
#     print("yy: ", yy.min(), yy.max())
#     print("zz: ", zz.min(), zz.max())
#     print("rvir: ", rvir.min(), rvir.max())
#     print("rodc: ", rodc.min(), rodc.max())
#     print("mvir_dm: ", mvir_dm.min(), mvir_dm.max())
#     print("modc_dm: ", modc_dm.min(), modc_dm.max())
#     if NEUTRINO_SIMULATION: print("m2Mpc_nu: ", m2Mpc_nu.min(), m2Mpc_nu.max())
#     print("halo_vir = ", halo_vir)
#     print("halo_odc = ", halo_odc)
#     print("oden_vir: ", oden_vir.min(), oden_vir.max())
#     print("oden_odc: ", oden_odc.min(), oden_odc.max())

#     exit()

def OverDensity(m, r):
    """ 
    Return the overdensity of the halo
    """
        
    global rho_crit, omega_m

    redshift = 0.0

    rho_crit = rho_crit*(omega_m*numpy.power(1.0+redshift,3) + (1.0-omega_m))
    ascal = 1.0/(1.0+redshift)
    oden = m/(4.*numpy.pi*numpy.power(r*ascal,3)/3.)/rho_crit
        
    return oden

# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------

# ReadHaloFile(input_file)

