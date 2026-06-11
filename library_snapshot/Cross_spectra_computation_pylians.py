##### In this library we compute cross spectra for density, velocity, and momentum-divergence fields
import sys
import os
import time
import pickle
import copy

import numpy as np
import psutil
from mpi4py import MPI

sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/')
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/library_snapshot')

import Pk_library as PKL
from analysis_functions import load


def cross_power_spectra_pylians(
    boxsize,
    sim_type,
    spec,
    ngrid_min,
    ngrid_max,
    ngrid_step,
    bulk_species1,
    bulk_species2,
    file_path,
    string,
    mass_cut,
    mass_width,
    save_path,
):
    """
    Compute density-density, momentum-divergence, and velocity-velocity cross spectra.

    Notes
    -----
    The input pickle files are assumed to contain:
        sub_box_data["density"] : CIC-deposited count field.
        sub_box_data["P_x"]     : CIC-deposited number-weighted velocity field,
                                  P_x(cell)=sum_i W_CIC(x_cell-x_i) v_x,i.
                                  This is not a mean velocity.

    This function constructs:
        delta = density / <density> - 1
        V_i   = P_i / density

    Then PKL.XPk_vv(delta, V, ...) computes spectra of the momentum-divergence-like
    fields div[(1+delta) V]. Since (1+delta) V = P / <density>, this corresponds to
    the properly normalized number-current / momentum-density-like field.
    """
    boxsize = np.float64(boxsize)

    start_time_all = time.time()
    start_mem_all = psutil.Process().memory_info().rss

    comm, rank, size = get_mpi_info()

    check_simulation_errors(sim_type, bulk_species1, bulk_species2, rank, comm)
    validate_inputs(string, ngrid_step)

    loop_spectra_computation(
        rank,
        boxsize,
        sim_type,
        spec,
        ngrid_min,
        ngrid_max,
        ngrid_step,
        size,
        bulk_species1,
        bulk_species2,
        file_path,
        string,
        mass_cut,
        mass_width,
        save_path,
    )

    comm.Barrier()
    print_total_usage(rank, start_time_all, start_mem_all)

    comm.Barrier()
    if rank == 0:
        print("\n*********** All spectra computation finished! ***********\n", flush=True)


def loop_spectra_computation(
    rank,
    boxsize,
    sim_type,
    spec,
    ngrid_min,
    ngrid_max,
    ngrid_step,
    size,
    bulk_species1,
    bulk_species2,
    file_path,
    string,
    mass_cut,
    mass_width,
    save_path,
):
    """
    Split requested ngrid values across MPI ranks and compute spectra for each assigned ngrid.

    ngrid_max follows Python range convention and is exclusive.
    """
    boxsize = np.float64(boxsize)

    ngrid_list = list(range(ngrid_min, ngrid_max, ngrid_step))
    ngrid_list_split = np.array_split(ngrid_list, size)

    if len(ngrid_list_split[rank]) == 0:
        return

    for ngrid in ngrid_list_split[rank]:
        ngrid = int(ngrid)

        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss

        power_data = {}

        compute_cross_spectra(
            file_path,
            bulk_species1,
            bulk_species2,
            spec,
            ngrid,
            sim_type,
            boxsize,
            power_data,
            string,
            mass_cut,
            mass_width,
        )

        simulation = simulation_def(
            boxsize,
            spec,
            sim_type,
            bulk_species1,
            bulk_species2,
            string,
            mass_cut,
            mass_width,
        )

        save_and_print_usage(start_time, start_mem, simulation, ngrid, power_data, save_path)


def get_mpi_info():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def validate_inputs(string, ngrid_step):
    if string not in {"+", "-", "bin", "all"}:
        raise ValueError(f"string must be one of '+', '-', 'bin', or 'all'; got {string!r}")

    if ngrid_step <= 0:
        raise ValueError(f"ngrid_step must be positive; got {ngrid_step}")


def check_simulation_errors(sim_type, bulk_species1, bulk_species2, rank, comm):
    if sim_type == "0.0ev" and (bulk_species1 == "nu" or bulk_species2 == "nu"):
        if rank == 0:
            print("In the case of LCDM we don't have nu snapshots!", flush=True)
        comm.Abort(1)


def compute_cross_spectra(
    file_path,
    bulk_species1,
    bulk_species2,
    spec,
    ngrid,
    sim,
    boxsize,
    power_data,
    string,
    mass_cut=1.0,
    mass_width=0.5,
):
    """
    Load two precomputed gridded-field pickle files and compute cross spectra.

    Important:
    - No MPI collectives are used here. This avoids collective-call mismatches when
      different MPI ranks have different numbers of ngrid values.
    - Copies are passed to Pylians routines because some Pylians routines mutate their
      input arrays in-place.
    """
    data1 = load_files_pickles(file_path, bulk_species1, spec, ngrid, sim, string, mass_cut, mass_width)
    data2 = load_files_pickles(file_path, bulk_species2, spec, ngrid, sim, string, mass_cut, mass_width)

    power_data["metadata"] = build_spectra_metadata(
        data1,
        bulk_species1,
        bulk_species2,
        spec,
        sim,
        ngrid,
        string,
        mass_cut,
        mass_width,
        boxsize,
    )

    density1, delta1, Vx1, Vy1, Vz1, Jx1, Jy1, Jz1 = build_delta_and_velocity_fields(
        data1, field_label="field 1"
    )
    density2, delta2, Vx2, Vy2, Vz2, Jx2, Jy2, Jz2 = build_delta_and_velocity_fields(
        data2, field_label="field 2"
    )

    axis = 0       # no RSD
    threads = 1
    MAS = "CIC"

    # Density auto/cross spectra
    Pk = PKL.XPk(
        [delta1.copy(), delta2.copy()],
        boxsize,
        axis,
        ["CIC", "CIC"],
        threads,
    )

    k = Pk.k3D
    Pk0_1 = Pk.Pk[:, 0, 0]      # monopole auto spectrum of field 1
    Pk0_2 = Pk.Pk[:, 0, 1]      # monopole auto spectrum of field 2
    Pk0_X = Pk.XPk[:, 0, 0]     # monopole cross spectrum of fields 1 and 2
    Nmodes = Pk.Nmodes3D

    power_data["spectra_delta1_x_delta2"] = [k, Pk0_1, Pk0_2, Pk0_X, Nmodes]

    # Momentum-divergence auto/cross spectra.
    # XPk_vv internally constructs div[(1+delta) V]; pass copies because it may mutate fields.
    k, Pk_11, Pk_22, Pk_12, Nmodes = PKL.XPk_vv(
        delta1.copy(),
        Vx1.copy(),
        Vy1.copy(),
        Vz1.copy(),
        delta2.copy(),
        Vx2.copy(),
        Vy2.copy(),
        Vz2.copy(),
        boxsize,
        axis,
        MAS,
        threads,
    )

    power_data["spectra_p1_x_p2"] = [k, Pk_11, Pk_22, Pk_12, Nmodes]

    # Velocity auto/cross spectra using your custom Pylians function.
    # Pass copies for safety.
    k, Pk1, Pk2, PkXv1v2, Nmodes = PKL.XPk_velvel(
        Vx1.copy(),
        Vy1.copy(),
        Vz1.copy(),
        Vx2.copy(),
        Vy2.copy(),
        Vz2.copy(),
        boxsize,
        axis,
        MAS,
        threads,
    )

    power_data["spectra_v1_x_v2"] = [k, Pk1, Pk2, PkXv1v2, Nmodes]

    # Component-by-component scalar spectra for the mean velocity fields.
    # These are useful diagnostics for anisotropy/grid effects and for constructing
    # P_{Delta V_a Delta V_a}=P_{1a,1a}+P_{2a,2a}-2P_{1a,2a}.
    velocity_components = {
        "x": scalar_cross_spectrum(Vx1, Vx2, boxsize, axis, MAS, threads),
        "y": scalar_cross_spectrum(Vy1, Vy2, boxsize, axis, MAS, threads),
        "z": scalar_cross_spectrum(Vz1, Vz2, boxsize, axis, MAS, threads),
    }
    power_data["spectra_velocity_components"] = velocity_components
    power_data["spectra_velocity_components_mean_xyz"] = average_component_spectra(velocity_components)

    # Component-by-component scalar spectra for the normalized number-current fields
    # J_a=(1+delta)V_a=P_a/<density>. These are the component analogues of the
    # momentum/current field entering theta=div[(1+delta)V].
    current_components = {
        "x": scalar_cross_spectrum(Jx1, Jx2, boxsize, axis, MAS, threads),
        "y": scalar_cross_spectrum(Jy1, Jy2, boxsize, axis, MAS, threads),
        "z": scalar_cross_spectrum(Jz1, Jz2, boxsize, axis, MAS, threads),
    }
    power_data["spectra_current_components"] = current_components
    power_data["spectra_current_components_mean_xyz"] = average_component_spectra(current_components)

    # Full component tensor spectra. These store all i,j component cross-spectra
    # for velocity and current fields. They are useful diagnostics and allow you
    # to inspect anisotropy/cross-component leakage. They are NOT by themselves
    # enough to exactly reconstruct theta=div(F), because theta requires k_i k_j
    # directional weighting before angular binning. For theta/vorticity use the
    # direct derivative spectra below.
    power_data["spectra_velocity_tensor_components"] = vector_tensor_component_spectra(
        (Vx1, Vy1, Vz1), (Vx2, Vy2, Vz2), boxsize, axis, MAS, threads
    )
    power_data["spectra_current_tensor_components"] = vector_tensor_component_spectra(
        (Jx1, Jy1, Jz1), (Jx2, Jy2, Jz2), boxsize, axis, MAS, threads
    )

    # Direct divergence and vorticity spectra from FFT derivatives.
    # For current fields J=(1+delta)V=P/<rho>, div(J) should be conceptually
    # close to the quantity computed by XPk_vv. The FFT derivative route also
    # gives curl(J), i.e. vorticity/current-vorticity.
    power_data["spectra_velocity_divergence_fft"] = divergence_cross_spectrum_fft(
        (Vx1, Vy1, Vz1), (Vx2, Vy2, Vz2), boxsize, axis, threads
    )
    power_data["spectra_current_divergence_fft"] = divergence_cross_spectrum_fft(
        (Jx1, Jy1, Jz1), (Jx2, Jy2, Jz2), boxsize, axis, threads
    )
    power_data["spectra_velocity_vorticity_fft"] = vorticity_cross_spectrum_fft(
        (Vx1, Vy1, Vz1), (Vx2, Vy2, Vz2), boxsize, axis, threads
    )
    power_data["spectra_current_vorticity_fft"] = vorticity_cross_spectrum_fft(
        (Jx1, Jy1, Jz1), (Jx2, Jy2, Jz2), boxsize, axis, threads
    )

    print("Spectra are computed!\n", flush=True)



def scalar_cross_only(field1, field2, boxsize, axis, MAS, threads):
    """
    Scalar cross spectrum only for two gridded scalar fields.

    Returns [k, P12, Nmodes], using the monopole.
    """
    Pk = PKL.XPk(
        [field1.copy(), field2.copy()],
        boxsize,
        axis,
        [MAS, MAS],
        threads,
    )
    return [Pk.k3D, Pk.XPk[:, 0, 0], Pk.Nmodes3D]


def vector_tensor_component_spectra(vec1, vec2, boxsize, axis, MAS, threads):
    """
    Save all component spectra for two vector fields.

    Parameters
    ----------
    vec1, vec2 : tuple of ndarray
        (Fx, Fy, Fz) for field 1 and field 2.

    Returns
    -------
    dict
        Contains:
          cross_1i_2j: cross spectra P_{F1_i,F2_j} for all i,j.
          auto1_i_j:   same-field spectra P_{F1_i,F1_j} for all i,j.
          auto2_i_j:   same-field spectra P_{F2_i,F2_j} for all i,j.

        Each entry is [k, Pij, Nmodes].

    Note
    ----
    These are angle-averaged component spectra. They are useful diagnostics,
    but do not exactly reconstruct divergence/curl spectra after angular binning,
    because divergence/curl require k_i k_j directional weights mode-by-mode.
    """
    labels = ("x", "y", "z")
    out = {"cross_1i_2j": {}, "auto1_i_j": {}, "auto2_i_j": {}}

    for a, Fa1 in zip(labels, vec1):
        for b, Fb2 in zip(labels, vec2):
            out["cross_1i_2j"][f"{a}{b}"] = scalar_cross_only(
                Fa1, Fb2, boxsize, axis, MAS, threads
            )

    for a, Fa1 in zip(labels, vec1):
        for b, Fb1 in zip(labels, vec1):
            out["auto1_i_j"][f"{a}{b}"] = scalar_cross_only(
                Fa1, Fb1, boxsize, axis, MAS, threads
            )

    for a, Fa2 in zip(labels, vec2):
        for b, Fb2 in zip(labels, vec2):
            out["auto2_i_j"][f"{a}{b}"] = scalar_cross_only(
                Fa2, Fb2, boxsize, axis, MAS, threads
            )

    return out


def fft_divergence_and_curl(vec, boxsize):
    """
    Compute divergence and curl of a 3D vector field using FFT derivatives.

    Parameters
    ----------
    vec : tuple of ndarray
        (Fx, Fy, Fz), each with shape (N,N,N).
    boxsize : float
        Box size in Mpc/h. Fourier wave numbers are in h/Mpc.

    Returns
    -------
    div : ndarray
        div(F) in real space.
    curl : tuple of ndarray
        (curl_x, curl_y, curl_z) in real space.

    Notes
    -----
    Uses derivatives d/dx -> i k_x, with k_x = 2*pi*n/L.
    The returned fields contain a derivative, so their units include h/Mpc
    times the units of F.
    """
    Fx, Fy, Fz = (np.asarray(v, dtype=np.float32) for v in vec)

    if Fx.shape != Fy.shape or Fx.shape != Fz.shape:
        raise ValueError("Vector components must have identical shapes.")
    if Fx.ndim != 3 or Fx.shape[0] != Fx.shape[1] or Fx.shape[0] != Fx.shape[2]:
        raise ValueError(f"Expected cubic 3D fields; got shape {Fx.shape}.")

    n = Fx.shape[0]
    dx = float(boxsize) / n

    kx = (2.0 * np.pi * np.fft.fftfreq(n, d=dx)).astype(np.float32)[:, None, None]
    ky = (2.0 * np.pi * np.fft.fftfreq(n, d=dx)).astype(np.float32)[None, :, None]
    kz = (2.0 * np.pi * np.fft.rfftfreq(n, d=dx)).astype(np.float32)[None, None, :]

    Fx_k = np.fft.rfftn(Fx)
    Fy_k = np.fft.rfftn(Fy)
    Fz_k = np.fft.rfftn(Fz)

    div_k = 1j * (kx * Fx_k + ky * Fy_k + kz * Fz_k)

    curl_x_k = 1j * (ky * Fz_k - kz * Fy_k)
    curl_y_k = 1j * (kz * Fx_k - kx * Fz_k)
    curl_z_k = 1j * (kx * Fy_k - ky * Fx_k)

    div = np.fft.irfftn(div_k, s=Fx.shape).real.astype(np.float32)
    curl_x = np.fft.irfftn(curl_x_k, s=Fx.shape).real.astype(np.float32)
    curl_y = np.fft.irfftn(curl_y_k, s=Fx.shape).real.astype(np.float32)
    curl_z = np.fft.irfftn(curl_z_k, s=Fx.shape).real.astype(np.float32)

    return div, (curl_x, curl_y, curl_z)


def divergence_cross_spectrum_fft(vec1, vec2, boxsize, axis, threads):
    """
    Cross spectra of FFT-computed divergence fields div(F1), div(F2).

    Returns [k, P_div1_div1, P_div2_div2, P_div1_div2, Nmodes].
    """
    div1, _ = fft_divergence_and_curl(vec1, boxsize)
    div2, _ = fft_divergence_and_curl(vec2, boxsize)

    # div fields are derived fields, not directly MAS-assigned density fields.
    # Use MAS='None' to avoid applying an additional MAS correction.
    return scalar_cross_spectrum(div1, div2, boxsize, axis, "None", threads)


def vorticity_cross_spectrum_fft(vec1, vec2, boxsize, axis, threads):
    """
    Cross spectra of FFT-computed curl/vorticity fields.

    Returns a dictionary with:
      components: component spectra for curl_x, curl_y, curl_z.
      total:      [k, P_omega1omega1, P_omega2omega2, P_omega1omega2, Nmodes]
                  where total is the sum over curl components.

    This computes curl(F), not curl(v) unless F is the velocity field.
    For F=J=(1+delta)v it is the curl of the current field.
    """
    _, curl1 = fft_divergence_and_curl(vec1, boxsize)
    _, curl2 = fft_divergence_and_curl(vec2, boxsize)

    components = {
        "x": scalar_cross_spectrum(curl1[0], curl2[0], boxsize, axis, "None", threads),
        "y": scalar_cross_spectrum(curl1[1], curl2[1], boxsize, axis, "None", threads),
        "z": scalar_cross_spectrum(curl1[2], curl2[2], boxsize, axis, "None", threads),
    }

    k = components["x"][0]
    Nmodes = components["x"][4]
    P11 = components["x"][1] + components["y"][1] + components["z"][1]
    P22 = components["x"][2] + components["y"][2] + components["z"][2]
    P12 = components["x"][3] + components["y"][3] + components["z"][3]

    return {"components": components, "total": [k, P11, P22, P12, Nmodes]}


def scalar_cross_spectrum(field1, field2, boxsize, axis, MAS, threads):
    """
    Scalar auto/cross spectra for two gridded scalar fields.

    Returns
    -------
    list
        [k, P11, P22, P12, Nmodes], using the monopole.
    """
    Pk = PKL.XPk(
        [field1.copy(), field2.copy()],
        boxsize,
        axis,
        [MAS, MAS],
        threads,
    )

    k = Pk.k3D
    P11 = Pk.Pk[:, 0, 0]
    P22 = Pk.Pk[:, 0, 1]
    P12 = Pk.XPk[:, 0, 0]
    Nmodes = Pk.Nmodes3D

    return [k, P11, P22, P12, Nmodes]


def average_component_spectra(component_dict):
    """
    Average x/y/z component spectra.

    Input values must be [k, P11, P22, P12, Nmodes].
    The same k/Nmodes are assumed for all components.
    """
    k = component_dict["x"][0]
    Nmodes = component_dict["x"][4]

    P11 = (component_dict["x"][1] + component_dict["y"][1] + component_dict["z"][1]) / 3.0
    P22 = (component_dict["x"][2] + component_dict["y"][2] + component_dict["z"][2]) / 3.0
    P12 = (component_dict["x"][3] + component_dict["y"][3] + component_dict["z"][3]) / 3.0

    return [k, P11, P22, P12, Nmodes]


def build_delta_and_velocity_fields(data, field_label="field"):
    """
    Convert saved density and CIC velocity-sum fields into delta and mean velocity fields.

    Saved fields:
        density = sum_i W_CIC
        P_x     = sum_i W_CIC v_x,i

    Constructed fields:
        delta = density / <density> - 1
        V_x   = P_x / density
        J_x   = (1+delta) V_x = P_x / <density>

    Empty cells are assigned V=0. The normalized current J is zero in empty cells.
    """
    sub = data["sub_box_data"]

    density = np.asarray(sub["density"], dtype=np.float32)
    Px = np.asarray(sub["P_x"], dtype=np.float32)
    Py = np.asarray(sub["P_y"], dtype=np.float32)
    Pz = np.asarray(sub["P_z"], dtype=np.float32)

    mean_density = np.mean(density, dtype=np.float64)

    if not np.isfinite(mean_density) or mean_density <= 0:
        raise ValueError(f"{field_label} has non-positive or non-finite mean density: {mean_density}")

    delta = (density / mean_density - 1.0).astype(np.float32)

    mask = density > 0.0

    Vx = np.zeros_like(Px, dtype=np.float32)
    Vy = np.zeros_like(Py, dtype=np.float32)
    Vz = np.zeros_like(Pz, dtype=np.float32)

    Vx[mask] = Px[mask] / density[mask]
    Vy[mask] = Py[mask] / density[mask]
    Vz[mask] = Pz[mask] / density[mask]

    Jx = (Px / mean_density).astype(np.float32)
    Jy = (Py / mean_density).astype(np.float32)
    Jz = (Pz / mean_density).astype(np.float32)

    return density, delta, Vx, Vy, Vz, Jx, Jy, Jz


def build_spectra_metadata(
    data1,
    bulk_species1,
    bulk_species2,
    spec,
    sim,
    ngrid,
    string,
    mass_cut,
    mass_width,
    boxsize,
):
    metadata = copy.deepcopy(data1.get("metadata", {}))

    metadata.update(
        {
            "bulk_species": [bulk_species1, bulk_species2],
            "spec": spec,
            "sim_type": sim,
            "ngrid": int(ngrid),
            "boxsize": float(boxsize),
            "mass_selection": string,
            "mass_cut": float(mass_cut),
            "mass_width": float(mass_width),
            "information_spectra_delta1_x_delta2": [
                "Stored as [k, Pk_delta1, Pk_delta2, Pk_delta1_delta2, Nmodes].",
                "k has units h/Mpc.",
                "Pk_delta fields have units (Mpc/h)^3 for dimensionless delta fields.",
            ],
            "information_spectra_p1_x_p2": [
                "Stored as [k, Pk_11, Pk_22, Pk_12, Nmodes].",
                "These are auto/cross spectra of momentum-divergence-like fields theta_i = div[(1+delta_i) V_i], not raw vector momentum spectra.",
                "Since saved P_i=sum W_CIC v_i and V_i=P_i/density_i, (1+delta_i)V_i=P_i/<density_i>.",
                "If velocities are km/s, units are approximately (km/s)^2 (Mpc/h)^3, up to derivative/implementation conventions in Pylians.",
            ],
            "information_spectra_v1_x_v2": [
                "Stored as [k, Pk_v1v1, Pk_v2v2, Pk_v1v2, Nmodes].",
                "Computed from mean velocity fields V_i=P_i/density_i with empty cells set to zero.",
                "If velocities are km/s, units are approximately (km/s)^2 (Mpc/h)^3.",
            ],
            "information_spectra_velocity_components": [
                "Dictionary with keys x, y, z. Each entry is [k, P11, P22, P12, Nmodes] for the scalar component V_a.",
                "Also stored: spectra_velocity_components_mean_xyz = average of x/y/z component spectra.",
                "These are component spectra of mean velocity fields V_a=P_a/density; they are sensitive to empty-cell treatment for sparse tracers.",
            ],
            "information_spectra_current_components": [
                "Dictionary with keys x, y, z. Each entry is [k, P11, P22, P12, Nmodes] for J_a=(1+delta)V_a=P_a/<density>.",
                "Also stored: spectra_current_components_mean_xyz = average of x/y/z component spectra.",
                "These component spectra are closer to the pair-weighted/current field entering theta=div[(1+delta)V].",
            ],
            "information_spectra_velocity_tensor_components": [
                "Full component spectra for V_i. Contains cross_1i_2j, auto1_i_j, auto2_i_j for i,j in {x,y,z}.",
                "Each entry is [k, Pij, Nmodes]. These are angle-averaged component spectra and are diagnostics, not an exact post-binned route to divergence.",
            ],
            "information_spectra_current_tensor_components": [
                "Full component spectra for J_i=(1+delta)V_i=P_i/<density>. Contains cross_1i_2j, auto1_i_j, auto2_i_j for i,j in {x,y,z}.",
                "Each entry is [k, Pij, Nmodes]. Exact theta/curl spectra require mode-by-mode k weighting; use the direct FFT derivative spectra.",
            ],
            "information_spectra_velocity_divergence_fft": [
                "Stored as [k, P_divV1_divV1, P_divV2_divV2, P_divV1_divV2, Nmodes].",
                "Computed by FFT derivatives div(V)=i k dot V, then binned using PKL.XPk with MAS='None'.",
            ],
            "information_spectra_current_divergence_fft": [
                "Stored as [k, P_divJ1_divJ1, P_divJ2_divJ2, P_divJ1_divJ2, Nmodes].",
                "Computed by FFT derivatives div(J)=i k dot J, with J=(1+delta)V=P/<density>.",
                "This should be conceptually comparable to spectra_p1_x_p2 from XPk_vv, modulo implementation/MAS conventions.",
            ],
            "information_spectra_velocity_vorticity_fft": [
                "Dictionary with components and total for curl(V). total=[k, sum_i P_w1i_w1i, sum_i P_w2i_w2i, sum_i P_w1i_w2i, Nmodes].",
                "Computed by FFT derivatives curl(V)=i k x V, then binned with MAS='None'.",
            ],
            "information_spectra_current_vorticity_fft": [
                "Dictionary with components and total for curl(J). This is current-vorticity, not bare velocity vorticity.",
                "Useful as a diagnostic of transverse/nonlinear/sampling contamination.",
            ],
            "input_field_interpretation": [
                "density is CIC-deposited count field density(cell)=sum_i W_CIC.",
                "P_x/P_y/P_z are CIC-deposited number-weighted velocity fields P_a(cell)=sum_i W_CIC v_{a,i}; they are not mean velocities.",
                "Mean velocities are reconstructed internally as V_a=P_a/density.",
            ],
        }
    )

    return metadata


def simulation_def(boxsize, spec, sim_type, species1, species2, string, mass_cut, mass_width):
    """
    Output-name convention matching the gridded-field production script.
    """
    if species1 == "halo":
        halo_species = species1
        other_species = species2
    elif species2 == "halo":
        halo_species = species2
        other_species = species1
    else:
        return f"{sim_type}_{spec}_{species1}_{species2}"

    if string == "+":
        halo_tag = f"{halo_species}_mass_{mass_cut:.1e}"
    elif string == "bin":
        halo_tag = f"{halo_species}_mass_{mass_cut:.1e}_mass_bin_{mass_width}"
    elif string == "-":
        halo_tag = f"{halo_species}_mass_le_{mass_cut:.1e}"
    elif string == "all":
        halo_tag = f"{halo_species}_all"
    else:
        raise ValueError(f"Unknown halo selection string={string!r}")

    return f"{sim_type}_{spec}_{halo_tag}_{other_species}"


def halo_directory_name(string, mass_cut, mass_width):
    """
    Directory naming convention used by the batch scripts for halo gridded fields.

    Adjust this function if your Run_all.sh uses different directory names.
    """
    if string == "+":
        return f"halo_mass_{mass_cut:.0e}".replace("+", "")
    elif string == "bin":
        return f"halo_mass_{mass_cut:.0e}".replace("+", "")
    elif string == "-":
        return f"halo_mass_le_{mass_cut:.0e}".replace("+", "")
    elif string == "all":
        return "halo_all"
    else:
        raise ValueError(f"Unknown halo selection string={string!r}")


def halo_pickle_basename(ngrid, sim, spec, string, mass_cut, mass_width):
    """
    File basename convention produced by bulk_velocity_pylians.simulation_def().
    """
    if string == "+":
        sim_name = f"{sim}_{spec}_halo_mass_{mass_cut:.1e}"
    elif string == "bin":
        sim_name = f"{sim}_{spec}_halo_mass_{mass_cut:.1e}_mass_bin_{mass_width}"
    elif string == "-":
        sim_name = f"{sim}_{spec}_halo_mass_le_{mass_cut:.1e}"
    elif string == "all":
        sim_name = f"{sim}_{spec}_halo_all"
    else:
        raise ValueError(f"Unknown halo selection string={string!r}")

    return f"data_ngrid_{ngrid}_sim_{sim_name}.pickle"


def load_files_pickles(file_path, bulk_species, spec, ngrid, sim, string, mass_cut, mass_width):
    """
    Load one precomputed gridded-field pickle.

    Expected directory layout:
        {file_path}/{spec}/{sim}/{species}/output/data_ngrid_...pickle
    for cdm/nu, and
        {file_path}/{spec}/{sim}/{halo_dir}/output/data_ngrid_...pickle
    for halos.

    The halo directory and basename conventions are factored into helper functions above.
    """
    if bulk_species == "halo":
        halo_dir = halo_directory_name(string, mass_cut, mass_width)
        basename = halo_pickle_basename(ngrid, sim, spec, string, mass_cut, mass_width)
        path = os.path.join(file_path, spec, sim, halo_dir, "output", basename)
    else:
        basename = f"data_ngrid_{ngrid}_sim_{sim}_{spec}_{bulk_species}.pickle"
        path = os.path.join(file_path, spec, sim, bulk_species, "output", basename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input pickle: {path}")

    return load(path)


def save_and_print_usage(start_time, start_mem, simulation, ngrid, power_data, save_path):
    save_dataframe(power_data, simulation, ngrid, save_path)
    print(f"{simulation}\n", flush=True)
    print_usage(start_time, start_mem, f" n_grid={ngrid} Finished!\n")


def save_dataframe(power_data, simulation, ngrid, save_path):
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(
        save_path,
        f"Cross_spectra_ngrid_{ngrid}_sim_{simulation}.pickle",
    )

    with open(output_file, "wb") as handle:
        pickle.dump(power_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def print_usage(start_time, start_mem, message=""):
    end_time = time.time()
    end_mem = psutil.Process().memory_info().rss

    elapsed_time = end_time - start_time
    elapsed_mem = end_mem - start_mem

    print(
        f"{message} - Time: {elapsed_time:.2f} s, Memory: {elapsed_mem / 1024 / 1024:.2f} MB",
        flush=True,
    )


def synchronize_processes(comm):
    comm.Barrier()


def print_total_usage(rank, start_time_all, start_mem_all):
    if rank == 0:
        print_usage(start_time_all, start_mem_all, "- Total time and memory!")
