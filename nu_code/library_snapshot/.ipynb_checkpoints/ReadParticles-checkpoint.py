import numpy
import os

# ------------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------------

# # Input particle file to read
# input_file = "./snapshots/0.000xv54_nu.dat"

# # Simulation parameters (needed for unit conversions)
omega_m = 0.2905
boxsize = 500.0
ngrid   = 6144
rankdim = 8

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

def DetermineRank(df):
    """
    Parse the filename to determine what rank this data corresponds to.
    """

    try:
        rank = int(df.split("xv")[1].split(".dat")[0].replace("_nu",""))
    except:
        print("ERROR: Could not parse rank from filename: ", df)
        exit()

    return rank

def PeriodicWrap(x):
    """
    Periodically wrap the positions around the global box.
    """

    global boxsize

    x[numpy.where(x<0.)] += boxsize
    x[numpy.where(x>=boxsize)] -= boxsize

    return x

def ConvertPositionUnits(xx0, yy0, zz0, rank, LENGTH_ONLY=False):
    """
    Convert the position from code units to global comoving Mpc/h
    """

    global boxsize, ngrid, rankdim

    # First convert to rank local comoving Mpc/h
    r_code2phys = boxsize/ngrid
    xx = xx0*r_code2phys
    yy = yy0*r_code2phys
    zz = zz0*r_code2phys

    if not LENGTH_ONLY:
        # Now add the rank offset to get global coordinates
        coord_z = rank//rankdim**2
        rank_xy = rank - coord_z*rankdim**2
        coord_y = rank_xy // rankdim
        coord_x = rank_xy - coord_y*rankdim
        xx += coord_x*(boxsize/rankdim)
        yy += coord_y*(boxsize/rankdim)
        zz += coord_z*(boxsize/rankdim)

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

def ReadParticleFile(df):
    """
    Read rank particle file (either dm or nu)
    """

    global boxsize, rankdim

    fr = open(df, "rb")
    # Read the header
    np = Byteswap(numpy.fromfile(fr, dtype="int32", count=1))[0] # Number of particles in this file
    a = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0] # Scale factor of this snapshot
    t = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0]
    tau = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0]
    nts = Byteswap(numpy.fromfile(fr, dtype="int32", count=1))[0]
    dt_f_acc = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0]
    dt_pp_acc = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0]
    dt_c_acc = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0]
    cur_checkpoint = Byteswap(numpy.fromfile(fr, dtype="int32", count=1))[0]
    cur_proj = Byteswap(numpy.fromfile(fr, dtype="int32", count=1))[0]
    cur_halo = Byteswap(numpy.fromfile(fr, dtype="int32", count=1))[0]
    mass_p = Byteswap(numpy.fromfile(fr, dtype="float32", count=1))[0] # Particle mass in code units
    # Check to see if the file has issues
    bsize = os.path.getsize(df)
    tsize = (4*12 + np*6*4)
    file_issue = False
    assert tsize >= bsize
    if bsize < tsize:
        np_file = np
        np = (bsize-4*12)//(4*6)
        file_issue = True
    # Read the body (which is x, y, z, vx, vy, vz for each particle at a time)
    data = Byteswap(numpy.fromfile(fr, dtype="float32", count=6*np))
    fr.close()

    # Unpack the data into position and velocity arrays
    data = data.reshape(np, 6).T
    xx = data[0]
    yy = data[1]
    zz = data[2]
    vx = data[3]
    vy = data[4]
    vz = data[5]

    # Print info if there as an issue
    if file_issue:
        xxl, yyl, zzl = ConvertPositionUnits(xx, yy, zz, 0, LENGTH_ONLY=True)
        print("WARNING: Had to manually reduce np_file = {:d} to np = {:d}".format(np_file, np))
        print(" ---> File name       : ", df)
        print(" ---> File size       : ", bsize)
        print(" ---> Fraction lost   : ", 1.0-np/(1.0*np_file))
        print(" ---> Check the following ranges which should be somewhere near [0.0, {:f}]".format(boxsize/rankdim))
        print(" ---> xx ranges : ", xxl.min(), xxl.max())
        print(" ---> yy ranges : ", yyl.min(), yyl.max())
        print(" ---> zz ranges : ", zzl.min(), zzl.max())
        exit()

    # Convert positions from code units to (global) comoving Mpc/h
    rank = DetermineRank(df)
    xx, yy, zz = ConvertPositionUnits(xx, yy, zz, rank)

    # Convert velocities from code units to comoving km/s
    vx, vy, vz = ConvertVelocityUnits(vx, vy, vz, a)

    return xx, yy, zz, vx, vy, vz

# # ------------------------------------------------------------------------------------
# # MAIN
# # ------------------------------------------------------------------------------------

# #
# # Read positions (in comoving Mpc/h) and velocities (in comoving km/s) from the file
# #

# xx, yy, zz, vx, vy, vz = ReadParticleFile(input_file)

# print("xx: ", xx.min(), xx.max())
# print("yy: ", yy.min(), yy.max())
# print("zz: ", zz.min(), zz.max())
# print("vx: ", vx.min(), vx.max())
# print("vy: ", vy.min(), vy.max())
# print("vz: ", vz.min(), vz.max())

