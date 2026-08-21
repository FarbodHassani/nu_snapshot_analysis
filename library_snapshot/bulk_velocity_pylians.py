import sys
import os
import time
import pickle
import gc

import numpy as np
import psutil

from mpi4py import MPI
from scipy import fft as sfft


# ---------------------------------------------------------------------------
# Local libraries
# ---------------------------------------------------------------------------

sys.path.append(
    '/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/'
)

sys.path.append(
    '/mn/stornext/u3/hassanif/neutrino_niayesh/'
    'Analysis/nu_code/library_snapshot'
)

import MAS_library as MASL
import readsnap

from colossus.lss import bias
from colossus.cosmology import cosmology

from ReadHalos import *
from ReadHalos_dm import *
from ReadParticles import *


cosmology.setCosmology('planck18')


# ===========================================================================
# Main driver
# ===========================================================================

def bulk_calculation_pylians(
    boxsize,
    num_pcl,
    sim_type,
    spec,
    ngrid_min,
    ngrid_max,
    ngrid_step,
    bulk_species,
    file_path,
    string,
    mass_limit,
    mass_width,
    save_path,
    type_data="normal",
    smoothing_scales=None,
    fft_workers=20,
):
    """
    Construct CIC density and first velocity-moment fields for CDM,
    neutrinos, or halos.

    For CDM/neutrinos, optional Gaussian-smoothed primitive fields can also
    be constructed:

        n^(R)   = W_R * n
        P_i^(R) = W_R * P_i

    with

        W_R(k) = exp[-k^2 R^2 / 2].

    The smoothed bulk velocity should subsequently be reconstructed as

        V_i^(R) = P_i^(R) / n^(R),

    rather than by smoothing an already reconstructed velocity field.

    Parameters
    ----------
    boxsize : float
        Simulation box size in Mpc/h.

    num_pcl : int
        Number of simulation particles per dimension.

    sim_type : str
        Simulation label.

    spec : str
        Additional simulation specification.

    ngrid_min, ngrid_max, ngrid_step : int
        Analysis-grid range. The values follow Python's range convention:
        range(ngrid_min, ngrid_max, ngrid_step).

    bulk_species : {"cdm", "nu", "halo"}
        Species to process.

    file_path : str
        Snapshot/catalogue path.

    string : str
        Halo mass-selection mode.

    mass_limit : float
        Halo mass-selection threshold or center.

    mass_width : float
        Width for halo mass-bin selection.

    save_path : str
        Output directory.

    type_data : {"normal", "JD"}
        Input-data format.

    smoothing_scales : iterable of float or None
        Gaussian smoothing scales R in Mpc/h.
        Smoothing is available only for CDM/neutrino fields.

    fft_workers : int
        Number of threads used by scipy.fft.
        For one process on a large-memory node, values around 10--30
        are generally sensible starting points.
    """

    # ------------------------------------------------------------------
    # Validate options
    # ------------------------------------------------------------------

    smoothing_scales = normalize_smoothing_scales(
        smoothing_scales
    )

    fft_workers = validate_fft_workers(
        fft_workers
    )

    if bulk_species not in {"cdm", "nu", "halo"}:
        raise ValueError(
            "bulk_species must be one of "
            "'cdm', 'nu', or 'halo'. "
            f"Got {bulk_species!r}."
        )

    if type_data not in {"normal", "JD"}:
        raise ValueError(
            "type_data must be 'normal' or 'JD'. "
            f"Got {type_data!r}."
        )

    if ngrid_step <= 0:
        raise ValueError(
            "ngrid_step must be positive."
        )

    if ngrid_max <= ngrid_min:
        raise ValueError(
            "ngrid_max must be larger than ngrid_min."
        )

    if bulk_species == "halo" and len(smoothing_scales) > 0:
        raise ValueError(
            "Physical smoothing is currently defined only for "
            "CDM/neutrino primitive moment fields, not halos."
        )

    boxsize = np.float64(boxsize)

    # ------------------------------------------------------------------
    # MPI
    # ------------------------------------------------------------------

    start_time_all = time.time()
    start_mem_all = psutil.Process().memory_info().rss

    comm, rank, size = get_mpi_info()

    check_simulation_errors(
        sim_type,
        bulk_species,
        rank,
        comm,
    )

    # ------------------------------------------------------------------
    # Check that the requested input exists
    # ------------------------------------------------------------------

    if type_data == "normal":

        if bulk_species == "halo":
            check_file_exists_all_ranks(
                file_path
            )
        else:
            check_file_exists_all_ranks(
                file_path + ".0"
            )

    elif type_data == "JD":

        if bulk_species == "halo":
            check_file_exists_all_ranks(
                os.path.join(
                    file_path,
                    "0.000halo0.dat",
                )
            )

        elif bulk_species == "cdm":
            check_file_exists_all_ranks(
                os.path.join(
                    file_path,
                    "0.000xv0.dat",
                )
            )

        elif bulk_species == "nu":
            check_file_exists_all_ranks(
                os.path.join(
                    file_path,
                    "0.000xv0_nu.dat",
                )
            )

    # ------------------------------------------------------------------
    # Print useful information once
    # ------------------------------------------------------------------

    if rank == 0:

        print(
            "\n-------------------------------------------",
            flush=True,
        )

        print(
            f"Species: {bulk_species}",
            flush=True,
        )

        print(
            f"MPI ranks: {size}",
            flush=True,
        )

        print(
            f"FFT workers per rank: {fft_workers}",
            flush=True,
        )

        if len(smoothing_scales) > 0:
            print(
                "Gaussian smoothing scales [Mpc/h]: "
                f"{smoothing_scales}",
                flush=True,
            )

        print(
            "-------------------------------------------\n",
            flush=True,
        )

        if bulk_species in {"cdm", "nu"} and size > 1:
            print(
                "WARNING: this implementation distributes ngrid values "
                "between MPI ranks, not particles. Each active MPI rank "
                "will independently load the complete particle snapshot. "
                "For very large particle sets, one MPI rank per node/job "
                "is strongly recommended.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    loop_ngrid_list(
        rank,
        boxsize,
        num_pcl,
        sim_type,
        spec,
        ngrid_min,
        ngrid_max,
        ngrid_step,
        size,
        bulk_species,
        file_path,
        string,
        mass_limit,
        mass_width,
        save_path,
        type_data,
        smoothing_scales,
        fft_workers,
    )

    comm.Barrier()

    print_total_usage(
        rank,
        start_time_all,
        start_mem_all,
    )

    if rank == 0:
        print(
            "\n*********** "
            "All bulk velocity computation finished! "
            "***********\n",
            flush=True,
        )

    comm.Barrier()


# ===========================================================================
# Analysis-grid loop
# ===========================================================================

def loop_ngrid_list(
    rank,
    boxsize,
    num_pcl,
    sim_type,
    spec,
    ngrid_min,
    ngrid_max,
    ngrid_step,
    size,
    bulk_species,
    file_path,
    string,
    mass_limit,
    mass_width,
    save_path,
    type_data,
    smoothing_scales,
    fft_workers,
):

    boxsize = np.float64(boxsize)

    ngrid_list = list(
        range(
            ngrid_min,
            ngrid_max,
            ngrid_step,
        )
    )

    ngrid_list_split = np.array_split(
        ngrid_list,
        size,
    )

    if len(ngrid_list_split[rank]) == 0:
        return

    # ------------------------------------------------------------------
    # Halo catalogues can be loaded once per rank.
    # ------------------------------------------------------------------

    halo_cache = None

    if type_data == "normal" and bulk_species == "halo":

        pos, vel, masses, counts = load_data_halo(
            file_path,
            string,
            mass_limit,
            mass_width=mass_width,
            Remove_subhalo="yes",
            return_counts=True,
        )

        halo_cache = {
            "pos": pos,
            "vel": vel,
            "masses": masses,
            "counts": counts,
        }

    elif type_data == "JD" and bulk_species == "halo":

        pos, vel, masses = load_data_JD_all(
            bulk_species,
            file_path,
            sim_type,
            mass_limit=mass_limit,
            mass_width=mass_width,
            string=string,
        )

        halo_cache = {
            "pos": np.asarray(
                pos,
                dtype=np.float32,
            ),
            "vel": np.asarray(
                vel,
                dtype=np.float32,
            ),
            "masses": np.asarray(
                masses,
                dtype=np.float32,
            ),
            "counts": {
                "num_halos": int(len(masses)),
                "total_halos_before_subhalo_cut": -1,
                "total_halos_after_subhalo_cut": int(len(masses)),
                "subhalo_removal_applied": False,
                "note": (
                    "JD halo catalogues have no subhalo PID column; "
                    f"count is after mass selection string={string}."
                ),
            },
        }

    # ------------------------------------------------------------------
    # Process assigned analysis-grid resolutions.
    # ------------------------------------------------------------------

    for ngrid in ngrid_list_split[rank]:

        ngrid = int(ngrid)

        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss

        sub_box_data = {}

        dx = float(boxsize) / float(ngrid)

        print(
            f"[rank {rank}] "
            f"type={type_data}, species={bulk_species}, "
            f"ngrid={ngrid}, dx={dx:.6f} Mpc/h",
            flush=True,
        )

        # ==============================================================
        # JD
        # ==============================================================

        if type_data == "JD":

            if bulk_species == "halo":

                metadata = compute_metadata_from_halo_counts(
                    boxsize,
                    num_pcl,
                    sim_type,
                    spec,
                    ngrid,
                    ngrid_min,
                    ngrid_max,
                    ngrid_step,
                    size,
                    bulk_species,
                    file_path,
                    string,
                    mass_limit,
                    mass_width,
                    save_path,
                    halo_cache["counts"],
                )

                process_halo_arrays(
                    halo_cache["pos"],
                    halo_cache["vel"],
                    halo_cache["masses"],
                    ngrid,
                    boxsize,
                    sub_box_data,
                )

            else:

                metadata = compute_metadata(
                    boxsize,
                    num_pcl,
                    sim_type,
                    spec,
                    ngrid,
                    ngrid_min,
                    ngrid_max,
                    ngrid_step,
                    size,
                    bulk_species,
                    file_path,
                    string,
                    mass_limit,
                    mass_width,
                    save_path,
                    smoothing_scales,
                    fft_workers,
                )

                process_cdm_nu_JD(
                    bulk_species,
                    sim_type,
                    file_path,
                    ngrid,
                    boxsize,
                    sub_box_data,
                    smoothing_scales=smoothing_scales,
                    fft_workers=fft_workers,
                )

        # ==============================================================
        # Normal Gadget snapshots
        # ==============================================================

        elif type_data == "normal":

            if bulk_species == "halo":

                metadata = compute_metadata_from_halo_counts(
                    boxsize,
                    num_pcl,
                    sim_type,
                    spec,
                    ngrid,
                    ngrid_min,
                    ngrid_max,
                    ngrid_step,
                    size,
                    bulk_species,
                    file_path,
                    string,
                    mass_limit,
                    mass_width,
                    save_path,
                    halo_cache["counts"],
                )

                process_halo_arrays(
                    halo_cache["pos"],
                    halo_cache["vel"],
                    halo_cache["masses"],
                    ngrid,
                    boxsize,
                    sub_box_data,
                )

            else:

                metadata = compute_metadata(
                    boxsize,
                    num_pcl,
                    sim_type,
                    spec,
                    ngrid,
                    ngrid_min,
                    ngrid_max,
                    ngrid_step,
                    size,
                    bulk_species,
                    file_path,
                    string,
                    mass_limit,
                    mass_width,
                    save_path,
                    smoothing_scales,
                    fft_workers,
                )

                process_cdm_nu_data_gadget(
                    rank,
                    file_path,
                    bulk_species,
                    ngrid,
                    boxsize,
                    num_pcl,
                    sub_box_data,
                    smoothing_scales=smoothing_scales,
                    fft_workers=fft_workers,
                )

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------

        simulation = simulation_def(
            boxsize,
            spec,
            num_pcl,
            sim_type,
            bulk_species,
            string,
            mass_limit,
            mass_width,
        )

        save_and_print_usage(
            start_time,
            start_mem,
            simulation,
            ngrid,
            sub_box_data,
            bulk_species,
            metadata,
            save_path,
        )

        # Important when several ngrid values are processed sequentially.
        del sub_box_data
        del metadata

        gc.collect()


# ===========================================================================
# Smoothing utilities
# ===========================================================================

def normalize_smoothing_scales(smoothing_scales):
    """
    Validate, sort, and deduplicate smoothing scales.
    """

    if smoothing_scales is None:
        return ()

    scales = tuple(
        float(R)
        for R in smoothing_scales
    )

    if any(
        not np.isfinite(R)
        for R in scales
    ):
        raise ValueError(
            "All smoothing scales must be finite."
        )

    if any(
        R <= 0.0
        for R in scales
    ):
        raise ValueError(
            "All smoothing scales must be positive."
        )

    return tuple(
        sorted(
            set(scales)
        )
    )


def validate_fft_workers(fft_workers):
    """
    Validate the number of scipy FFT worker threads.
    """

    fft_workers = int(fft_workers)

    if fft_workers < 1:
        raise ValueError(
            "fft_workers must be >= 1."
        )

    cpu_count = os.cpu_count()

    if (
        cpu_count is not None
        and fft_workers > cpu_count
    ):
        print(
            f"WARNING: fft_workers={fft_workers} exceeds "
            f"os.cpu_count()={cpu_count}.",
            flush=True,
        )

    return fft_workers


def build_gaussian_fourier_windows(
    ngrid,
    boxsize,
    smoothing_scales,
):
    """
    Construct the 1-D factors of the separable Gaussian Fourier window.

    For each R,

        W_R(kx,ky,kz)
        =
        exp[-R^2 kx^2/2]
        exp[-R^2 ky^2/2]
        exp[-R^2 kz^2/2].

    Only 1-D arrays are stored, avoiding a full Ngrid^3 window.
    """

    dx = float(boxsize) / float(ngrid)

    k_full = (
        2.0
        * np.pi
        * sfft.fftfreq(
            ngrid,
            d=dx,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    k_rfft = (
        2.0
        * np.pi
        * sfft.rfftfreq(
            ngrid,
            d=dx,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    windows = {}

    for R in smoothing_scales:

        R32 = np.float32(R)

        W_full = np.exp(
            -np.float32(0.5)
            * (R32 * k_full) ** 2
        ).astype(
            np.float32,
            copy=False,
        )

        W_rfft = np.exp(
            -np.float32(0.5)
            * (R32 * k_rfft) ** 2
        ).astype(
            np.float32,
            copy=False,
        )

        # x and y have the same Fourier frequencies
        # because the grid is cubic.
        windows[R] = (
            W_full,
            W_full,
            W_rfft,
        )

    return windows


def iter_gaussian_smoothed_fields(
    field,
    smoothing_scales,
    gaussian_windows,
    fft_workers=20,
):
    """
    Smooth one primitive periodic 3-D field at several Gaussian scales.

    The expensive forward transform is performed only once:

        field(x) -> field(k).

    For each R, a temporary Fourier copy is multiplied by W_R(k) and
    inverse transformed.

    Parameters
    ----------
    field : ndarray
        Cubic float32 real-space field.

    smoothing_scales : tuple
        Gaussian scales in Mpc/h.

    gaussian_windows : dict
        Precomputed separable Fourier windows from
        build_gaussian_fourier_windows().

    fft_workers : int
        Number of scipy FFT threads.

    Yields
    ------
    R : float
        Smoothing scale.

    field_R : ndarray
        Smoothed float32 real-space field.
    """

    if len(smoothing_scales) == 0:
        return

    if field.ndim != 3:
        raise ValueError(
            f"Expected a 3-D field, got ndim={field.ndim}."
        )

    ngrid = field.shape[0]

    if field.shape != (
        ngrid,
        ngrid,
        ngrid,
    ):
        raise ValueError(
            "Expected a cubic 3-D field, "
            f"got shape={field.shape}."
        )

    if field.dtype != np.float32:
        print(
            f"WARNING: smoothing input dtype is {field.dtype}; "
            "float32 is recommended for memory efficiency.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # One forward FFT for this primitive field.
    # scipy.fft preserves single precision for float32 input.
    # ------------------------------------------------------------------

    field_k = sfft.rfftn(
        field,
        workers=fft_workers,
    )

    for R in smoothing_scales:

        Wx, Wy, Wz = gaussian_windows[R]

        # Must preserve field_k for subsequent R values.
        work_k = field_k.copy()

        work_k *= Wx[:, None, None]
        work_k *= Wy[None, :, None]
        work_k *= Wz[None, None, :]

        field_R = sfft.irfftn(
            work_k,
            s=field.shape,
            workers=fft_workers,
        )

        field_R = np.asarray(
            field_R,
            dtype=np.float32,
        )

        del work_k

        yield R, field_R

    del field_k


def print_smoothing_resolution(
    ngrid,
    boxsize,
    smoothing_scales,
):
    """
    Print R/dx so that the physical smoothing scale can be compared
    directly with the analysis-grid spacing.
    """

    dx = float(boxsize) / float(ngrid)

    print(
        f"Grid spacing dx = {dx:.6f} Mpc/h",
        flush=True,
    )

    for R in smoothing_scales:

        print(
            f"  R={R:g} Mpc/h : "
            f"R/dx={R / dx:.3f} cells",
            flush=True,
        )


def print_smoothing_conservation(
    field_name,
    R,
    sum_before,
    sum_after,
):
    """
    Print a zero-mode conservation diagnostic.

    A normalized Gaussian satisfies W_R(k=0)=1, so smoothing should
    preserve the total deposited zeroth/first moment up to numerical
    precision.
    """

    difference = (
        sum_after
        - sum_before
    )

    if field_name == "density":

        relative_difference = (
            abs(difference)
            / max(
                abs(sum_before),
                1.0e-30,
            )
        )

        print(
            f"[smoothing check] "
            f"{field_name}, R={R:g}: "
            f"sum_before={sum_before:.8e}, "
            f"sum_after={sum_after:.8e}, "
            f"delta={difference:.8e}, "
            f"relative={relative_difference:.3e}",
            flush=True,
        )

        if relative_difference > 1.0e-5:
            print(
                "WARNING: density zero-mode conservation differs "
                "by more than 1e-5 relative.",
                flush=True,
            )

    else:

        # The total P_i can naturally be close to zero.
        # Therefore an ordinary relative error can be misleading.
        print(
            f"[smoothing check] "
            f"{field_name}, R={R:g}: "
            f"sum_before={sum_before:.8e}, "
            f"sum_after={sum_after:.8e}, "
            f"delta={difference:.8e}",
            flush=True,
        )


def add_smoothed_primitive_fields(
    sub_box_data,
    boxsize,
    smoothing_scales,
    fft_workers=20,
):
    """
    Construct Gaussian-smoothed primitive zeroth and first moments.

    The saved structure is

        sub_box_data["smoothed"]["R_10"]["density"]
        sub_box_data["smoothed"]["R_10"]["P_x"]
        sub_box_data["smoothed"]["R_10"]["P_y"]
        sub_box_data["smoothed"]["R_10"]["P_z"]

    etc.

    The smoothed bulk velocity is intentionally NOT saved because it is
    exactly reconstructible as

        V_i^(R) = P_i^(R) / density^(R).

    This avoids storing three redundant Ngrid^3 fields for every R.
    """

    if len(smoothing_scales) == 0:
        return

    primitive_names = (
        "density",
        "P_x",
        "P_y",
        "P_z",
    )

    for field_name in primitive_names:

        if field_name not in sub_box_data:
            raise KeyError(
                f"Missing primitive field {field_name!r}."
            )

    ngrid = sub_box_data["density"].shape[0]

    # ------------------------------------------------------------------
    # Physical-resolution information
    # ------------------------------------------------------------------

    print_smoothing_resolution(
        ngrid,
        boxsize,
        smoothing_scales,
    )

    # ------------------------------------------------------------------
    # Build tiny 1-D Gaussian windows once.
    # ------------------------------------------------------------------

    gaussian_windows = build_gaussian_fourier_windows(
        ngrid,
        boxsize,
        smoothing_scales,
    )

    # ------------------------------------------------------------------
    # Allocate output hierarchy.
    # ------------------------------------------------------------------

    sub_box_data["smoothed"] = {
        f"R_{R:g}": {}
        for R in smoothing_scales
    }

    # ------------------------------------------------------------------
    # Smooth primitive fields independently.
    # ------------------------------------------------------------------

    for field_name in primitive_names:

        field = sub_box_data[field_name]

        # Use float64 accumulation for the conservation diagnostic.
        sum_before = np.sum(
            field,
            dtype=np.float64,
        )

        print(
            f"\nStarting Gaussian smoothing of {field_name}.",
            flush=True,
        )

        for R, field_R in iter_gaussian_smoothed_fields(
            field,
            smoothing_scales,
            gaussian_windows,
            fft_workers=fft_workers,
        ):

            R_key = f"R_{R:g}"

            sum_after = np.sum(
                field_R,
                dtype=np.float64,
            )

            print_smoothing_conservation(
                field_name,
                R,
                sum_before,
                sum_after,
            )

            sub_box_data[
                "smoothed"
            ][
                R_key
            ][
                field_name
            ] = field_R

        gc.collect()

    del gaussian_windows

    gc.collect()


# ===========================================================================
# MPI utilities
# ===========================================================================

def get_mpi_info():

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    return comm, rank, size


def check_file_exists_all_ranks(path):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        exists = os.path.exists(path)
    else:
        exists = None

    exists = comm.bcast(
        exists,
        root=0,
    )

    if not exists:

        if rank == 0:
            print(
                f"Error: File not found: {path}",
                flush=True,
            )

        comm.Abort(1)


def check_simulation_errors(
    sim_type,
    bulk_species,
    rank,
    comm,
):

    if (
        sim_type == "0.0ev"
        and bulk_species == "nu"
    ):

        if rank == 0:
            print(
                "In the case of LCDM we don't have nu snapshots!",
                flush=True,
            )

        comm.Abort(1)


# ===========================================================================
# Metadata
# ===========================================================================

def compute_metadata(
    boxsize,
    Num_pcl_sim,
    sim_type,
    spec,
    ngrid,
    ngrid_min,
    ngrid_max,
    ngrid_step,
    size,
    bulk_species,
    file_path,
    string,
    mass_limit,
    mass_width,
    save_path,
    smoothing_scales,
    fft_workers,
):

    if bulk_species == "halo":
        raise ValueError(
            "For halo metadata use "
            "compute_metadata_from_halo_counts()."
        )

    dx = float(boxsize) / int(ngrid)

    smoothing_R_over_dx = {
        f"{R:g}": float(R) / dx
        for R in smoothing_scales
    }

    meta_data = {

        'spec': spec,

        'boxsize': float(boxsize),

        # Keep original key for backward compatibility.
        'N_grids_simulation': Num_pcl_sim,

        # More explicit meaning.
        'N_particles_per_dimension': int(Num_pcl_sim),

        'sim_type': sim_type,

        'ngrid': int(ngrid),

        'grid_spacing_Mpc_h': dx,

        'ngrid_min': int(ngrid_min),
        'ngrid_max': int(ngrid_max),
        'ngrid_step': int(ngrid_step),

        'mpi_size': int(size),

        'fft_workers': int(fft_workers),

        'file_path': file_path,

        'bulk_species': bulk_species,

        'save_path': save_path,

        'density': (
            'CIC-deposited particle-count/zeroth-moment field: '
            'density(cell)=sum_i W_CIC(x_cell-x_i).'
        ),

        'P_x': (
            'CIC-deposited first velocity moment / velocity-sum field: '
            'P_x(cell)=sum_i W_CIC(x_cell-x_i) v_{x,i}. '
            'For equal-mass particles, P_x/cell_volume is proportional '
            'to the physical momentum density. '
            'The local bulk velocity is P_x/density.'
        ),

        'P_y': (
            'CIC-deposited first velocity moment / velocity-sum field: '
            'P_y(cell)=sum_i W_CIC(x_cell-x_i) v_{y,i}. '
            'For equal-mass particles, P_y/cell_volume is proportional '
            'to the physical momentum density. '
            'The local bulk velocity is P_y/density.'
        ),

        'P_z': (
            'CIC-deposited first velocity moment / velocity-sum field: '
            'P_z(cell)=sum_i W_CIC(x_cell-x_i) v_{z,i}. '
            'For equal-mass particles, P_z/cell_volume is proportional '
            'to the physical momentum density. '
            'The local bulk velocity is P_z/density.'
        ),

        'smoothing_kernel': (
            'Gaussian W_R(k)=exp[-k^2 R^2/2], applied separately '
            'to the primitive CIC zeroth and first velocity moments.'
        ),

        'smoothing_definition': (
            'density^(R)=W_R*density and '
            'P_i^(R)=W_R*P_i.'
        ),

        'smoothed_velocity_definition': (
            'V_i^(R)=P_i^(R)/density^(R). '
            'The already reconstructed velocity field is not '
            'directly smoothed.'
        ),

        'smoothing_scales_Mpc_h': list(
            smoothing_scales
        ),

        'smoothing_R_over_dx': smoothing_R_over_dx,

        'smoothing_zero_mode': (
            'The Gaussian is normalized so W_R(k=0)=1; therefore '
            'the total deposited density and first moments should '
            'be preserved by smoothing up to FFT precision.'
        ),
    }

    return meta_data


def compute_metadata_from_halo_counts(
    boxsize,
    Num_pcl_sim,
    sim_type,
    spec,
    ngrid,
    ngrid_min,
    ngrid_max,
    ngrid_step,
    size,
    bulk_species,
    file_path,
    string,
    mass_limit,
    mass_width,
    save_path,
    counts,
):

    dx = float(boxsize) / int(ngrid)

    meta = {

        'spec': spec,

        'boxsize': float(boxsize),

        # Retain for compatibility.
        'N_grids_simulation': Num_pcl_sim,

        'N_particles_per_dimension': int(Num_pcl_sim),

        'sim_type': sim_type,

        'ngrid': int(ngrid),

        'grid_spacing_Mpc_h': dx,

        'ngrid_min': int(ngrid_min),
        'ngrid_max': int(ngrid_max),
        'ngrid_step': int(ngrid_step),

        'mpi_size': int(size),

        'mass cut': "{:.4e}".format(
            mass_limit
        ),

        'masses': string,

        'mass_width': mass_width,

        'file_path': file_path,

        'bulk_species': bulk_species,

        'save_path': save_path,

        'num_halos': counts[
            "num_halos"
        ],

        'total_halos_before_subhalo_cut': counts[
            "total_halos_before_subhalo_cut"
        ],

        'total_halos_after_subhalo_cut': counts[
            "total_halos_after_subhalo_cut"
        ],

        'subhalo_removal_applied': counts.get(
            "subhalo_removal_applied",
            True,
        ),

        'note': counts.get(
            "note",
            "",
        ),

        'density': (
            'CIC-deposited halo count field: '
            'density(cell)=sum_h W_CIC(x_cell-x_h).'
        ),

        'P_x': (
            'CIC-deposited halo-number first velocity moment: '
            'P_x(cell)=sum_h W_CIC(x_cell-x_h) v_{x,h}; '
            'P_x/density is the mean halo velocity.'
        ),

        'P_y': (
            'CIC-deposited halo-number first velocity moment: '
            'P_y(cell)=sum_h W_CIC(x_cell-x_h) v_{y,h}; '
            'P_y/density is the mean halo velocity.'
        ),

        'P_z': (
            'CIC-deposited halo-number first velocity moment: '
            'P_z(cell)=sum_h W_CIC(x_cell-x_h) v_{z,h}; '
            'P_z/density is the mean halo velocity.'
        ),

        'sum_b_M': (
            'CIC-deposited sum of halo weights '
            'w_h=b_h+(M_h/1.3e14)^0.85.'
        ),

        'sum_b_M_vel_h_x': (
            'CIC-deposited weighted halo first velocity moment: '
            'sum_h W_CIC w_h v_{x,h}; divide by sum_b_M '
            'for weighted mean velocity.'
        ),

        'sum_b_M_vel_h_y': (
            'CIC-deposited weighted halo first velocity moment: '
            'sum_h W_CIC w_h v_{y,h}; divide by sum_b_M '
            'for weighted mean velocity.'
        ),

        'sum_b_M_vel_h_z': (
            'CIC-deposited weighted halo first velocity moment: '
            'sum_h W_CIC w_h v_{z,h}; divide by sum_b_M '
            'for weighted mean velocity.'
        ),
    }

    return meta


# ===========================================================================
# Halo processing
# ===========================================================================

def process_halo_arrays(
    pos,
    vel,
    masses,
    ngrid,
    boxsize,
    sub_box_data,
):

    if masses.size == 0:
        raise ValueError(
            "No halos selected. "
            "Check mass_limit, mass_width, and subhalo cut."
        )

    bias_h = np.float32(
        bias.haloBias(
            masses,
            model='sheth01',
            z=0.0,
            mdef='200m',
        )
    )

    print(
        f"{pos.shape[0]} number of haloes loaded\n",
        flush=True,
    )

    MAS = 'CIC'
    verbose = False

    grid_shape = (
        ngrid,
        ngrid,
        ngrid,
    )

    # ------------------------------------------------------------------
    # Halo count field
    # ------------------------------------------------------------------

    rho = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        rho,
        boxsize,
        MAS,
        verbose=verbose,
    )

    print(
        "Density field is computed!\n",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Halo weights
    # ------------------------------------------------------------------

    bM_weight = (
        bias_h
        + (masses / 1.3e14) ** 0.85
    ).astype(
        np.float32,
        copy=False,
    )

    sum_b_M_field = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        sum_b_M_field,
        boxsize,
        MAS,
        W=bM_weight,
        verbose=verbose,
    )

    print(
        "sum_b_M_field field is computed!\n",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Halo first velocity moments
    # ------------------------------------------------------------------

    Vx = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        Vx,
        boxsize,
        MAS,
        W=vel[:, 0],
        verbose=verbose,
    )

    Vy = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        Vy,
        boxsize,
        MAS,
        W=vel[:, 1],
        verbose=verbose,
    )

    Vz = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        Vz,
        boxsize,
        MAS,
        W=vel[:, 2],
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Weighted halo velocity moments
    # ------------------------------------------------------------------

    sum_b_M_vel_h_x_field = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        sum_b_M_vel_h_x_field,
        boxsize,
        MAS,
        W=vel[:, 0] * bM_weight,
        verbose=verbose,
    )

    sum_b_M_vel_h_y_field = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        sum_b_M_vel_h_y_field,
        boxsize,
        MAS,
        W=vel[:, 1] * bM_weight,
        verbose=verbose,
    )

    sum_b_M_vel_h_z_field = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        sum_b_M_vel_h_z_field,
        boxsize,
        MAS,
        W=vel[:, 2] * bM_weight,
        verbose=verbose,
    )

    print(
        "All halo fields are computed!\n",
        flush=True,
    )

    sub_box_data['density'] = rho

    sub_box_data[
        'sum_b_M'
    ] = sum_b_M_field

    sub_box_data[
        'sum_b_M_vel_h_x'
    ] = sum_b_M_vel_h_x_field

    sub_box_data[
        'sum_b_M_vel_h_y'
    ] = sum_b_M_vel_h_y_field

    sub_box_data[
        'sum_b_M_vel_h_z'
    ] = sum_b_M_vel_h_z_field

    sub_box_data['P_x'] = Vx
    sub_box_data['P_y'] = Vy
    sub_box_data['P_z'] = Vz


# ===========================================================================
# Gadget CDM/neutrino processing
# ===========================================================================

def process_cdm_nu_data_gadget(
    rank,
    file_path,
    bulk_species,
    ngrid,
    boxsize,
    num_pcl,
    sub_box_data,
    smoothing_scales=(),
    fft_workers=20,
):

    # Read the snapshot header from the base filename.
    head = readsnap.snapshot_header(file_path)
    num_files = int(head.filenum)

    if rank == 0:
        print(
            f"Number of Gadget files to be loaded: {num_files}",
            flush=True,
        )

    MAS = "CIC"
    verbose = False

    grid_shape = (ngrid, ngrid, ngrid)

    # Allocate primitive fields ONCE.
    rho = np.zeros(grid_shape, dtype=np.float32)
    Vx = np.zeros(grid_shape, dtype=np.float32)
    Vy = np.zeros(grid_shape, dtype=np.float32)
    Vz = np.zeros(grid_shape, dtype=np.float32)

    number_of_particles = 0

    # --------------------------------------------------------------
    # Read and deposit one Gadget file at a time.
    # --------------------------------------------------------------

    for num in range(num_files):

        current_file = file_path + "." + str(num)

        if not os.path.exists(current_file):
            raise FileNotFoundError(
                f"Gadget snapshot file not found: {current_file}"
            )

        # Your existing reader already converts positions kpc/h -> Mpc/h.
        # Use particle type 1, consistent with the current Pylians reader.
        pos, vel = load_data_gadget(
            current_file,
            1,
        )

        pos = np.asarray(
            pos,
            dtype=np.float32,
        )

        vel = np.asarray(
            vel,
            dtype=np.float32,
        )

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(
                f"Expected positions with shape (N,3), got {pos.shape}."
            )

        if vel.ndim != 2 or vel.shape[1] != 3:
            raise ValueError(
                f"Expected velocities with shape (N,3), got {vel.shape}."
            )

        if pos.shape[0] != vel.shape[0]:
            raise ValueError(
                f"Position/velocity particle counts differ for "
                f"{current_file}: {pos.shape[0]} vs {vel.shape[0]}."
            )

        number_of_particles += pos.shape[0]

        if rank == 0:
            print(
                f"Loading {current_file}: "
                f"{pos.shape[0]} particles",
                flush=True,
            )

        # CIC accumulation into the SAME arrays.
        MASL.MA(
            pos,
            rho,
            boxsize,
            MAS,
            verbose=verbose,
        )

        MASL.MA(
            pos,
            Vx,
            boxsize,
            MAS,
            W=vel[:, 0],
            verbose=verbose,
        )

        MASL.MA(
            pos,
            Vy,
            boxsize,
            MAS,
            W=vel[:, 1],
            verbose=verbose,
        )

        MASL.MA(
            pos,
            Vz,
            boxsize,
            MAS,
            W=vel[:, 2],
            verbose=verbose,
        )

        # Important for the 2600^3 simulations.
        del pos
        del vel

        gc.collect()

    if rank == 0:
        print(
            f"All Gadget files deposited. "
            f"Total particles processed: {number_of_particles}",
            flush=True,
        )

    # --------------------------------------------------------------
    # Store completed primitive fields.
    # --------------------------------------------------------------

    sub_box_data["density"] = rho
    sub_box_data["P_x"] = Vx
    sub_box_data["P_y"] = Vy
    sub_box_data["P_z"] = Vz

    # --------------------------------------------------------------
    # Physical smoothing AFTER all files have been accumulated.
    # --------------------------------------------------------------

    if len(smoothing_scales) > 0:

        if rank == 0:
            print(
                "\nConstructing physically smoothed density "
                "and first-moment fields for "
                f"R={smoothing_scales} Mpc/h.",
                flush=True,
            )

        add_smoothed_primitive_fields(
            sub_box_data,
            boxsize,
            smoothing_scales,
            fft_workers=fft_workers,
        )


# ===========================================================================
# JD CDM/neutrino processing
# ===========================================================================

def process_cdm_nu_JD(
    bulk_species,
    sim_type,
    file_path,
    ngrid,
    boxsize,
    sub_box_data,
    smoothing_scales=(),
    fft_workers=20,
):

    pos, vel = load_data_JD_all(
        bulk_species,
        file_path,
        sim_type,
    )

    pos = np.asarray(
        pos
    ).astype(
        np.float32,
        copy=False,
    )

    vel = np.asarray(
        vel
    ).astype(
        np.float32,
        copy=False,
    )

    MAS = 'CIC'
    verbose = False

    grid_shape = (
        ngrid,
        ngrid,
        ngrid,
    )

    # ------------------------------------------------------------------
    # Zeroth moment
    # ------------------------------------------------------------------

    rho = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        rho,
        boxsize,
        MAS,
        verbose=verbose,
    )

    print(
        "Density field is computed!\n",
        flush=True,
    )

    # ------------------------------------------------------------------
    # First moment x
    # ------------------------------------------------------------------

    weight = vel[:, 0]

    Vx = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        Vx,
        boxsize,
        MAS,
        W=weight,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # First moment y
    # ------------------------------------------------------------------

    weight = vel[:, 1]

    Vy = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        Vy,
        boxsize,
        MAS,
        W=weight,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # First moment z
    # ------------------------------------------------------------------

    weight = vel[:, 2]

    Vz = np.zeros(
        grid_shape,
        dtype=np.float32,
    )

    MASL.MA(
        pos,
        Vz,
        boxsize,
        MAS,
        W=weight,
        verbose=verbose,
    )

    number_of_particles = pos.shape[0]

    print(
        "Momentum/first-moment fields are computed. "
        f"Loaded number of particles is {number_of_particles}.\n",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Save primitive fields
    # ------------------------------------------------------------------

    sub_box_data['density'] = rho
    sub_box_data['P_x'] = Vx
    sub_box_data['P_y'] = Vy
    sub_box_data['P_z'] = Vz

    # ------------------------------------------------------------------
    # Remove the last velocity-column view BEFORE deleting vel.
    # Otherwise `weight` can keep the underlying vel allocation alive.
    # ------------------------------------------------------------------

    del weight
    del pos
    del vel

    gc.collect()

    # ------------------------------------------------------------------
    # Physical smoothing
    # ------------------------------------------------------------------

    if len(smoothing_scales) > 0:

        print(
            "\nConstructing physically smoothed density "
            "and first-moment fields for "
            f"R={smoothing_scales} Mpc/h.",
            flush=True,
        )

        add_smoothed_primitive_fields(
            sub_box_data,
            boxsize,
            smoothing_scales,
            fft_workers=fft_workers,
        )


# ===========================================================================
# Timing / memory
# ===========================================================================

def save_and_print_usage(
    start_time,
    start_mem,
    simulation,
    ngrid,
    sub_box_data,
    bulk_species,
    metadata,
    save_path,
):

    save_dataframe(
        sub_box_data,
        simulation,
        bulk_species,
        metadata,
        ngrid,
        save_path,
    )

    print(
        f"{simulation}\n",
        flush=True,
    )

    print_usage(
        start_time,
        start_mem,
        f" n_grid={ngrid} Finished!\n",
    )


def print_usage(
    start_time,
    start_mem,
    message="",
):

    end_time = time.time()

    end_mem = (
        psutil.Process()
        .memory_info()
        .rss
    )

    elapsed_time = (
        end_time
        - start_time
    )

    elapsed_mem = (
        end_mem
        - start_mem
    )

    current_mem_gb = (
        end_mem
        / 1024**3
    )

    print(
        f"{message} "
        f"- Time: {elapsed_time:.2f} s, "
        f"Memory change: {elapsed_mem / 1024**2:.2f} MB, "
        f"Current RSS: {current_mem_gb:.2f} GB",
        flush=True,
    )


def synchronize_processes(comm):

    comm.Barrier()


def print_total_usage(
    rank,
    start_time_all,
    start_mem_all,
):

    if rank == 0:

        print_usage(
            start_time_all,
            start_mem_all,
            '- Total time and memory!',
        )


# ===========================================================================
# Simulation naming
# ===========================================================================

def simulation_def(
    boxsize,
    spec,
    N_pcl_sim,
    sim_type,
    species,
    string,
    mass_limit,
    mass_width,
):

    if species == "halo":

        if string == "+":
            return (
                sim_type
                + '_'
                + spec
                + '_'
                + species
                + f'_mass_{mass_limit:.1e}'
            )

        elif string == "bin":
            return (
                sim_type
                + '_'
                + spec
                + '_'
                + species
                + f'_mass_{mass_limit:.1e}'
                + f'_mass_bin_{mass_width}'
            )

        elif string == "all":
            return (
                sim_type
                + '_'
                + spec
                + '_'
                + species
                + '_all'
            )

        elif string == "-":
            return (
                sim_type
                + '_'
                + spec
                + '_'
                + species
                + f'_mass_le_{mass_limit:.1e}'
            )

        else:
            raise ValueError(
                "Unknown halo mass-selection "
                f"string={string!r}"
            )

    else:

        return (
            sim_type
            + '_'
            + spec
            + '_'
            + species
        )


# ===========================================================================
# Generic Gadget loader
# ===========================================================================

def load_data_gadget(
    sim_path,
    ptype,
):

    pos = (
        readsnap.read_block(
            sim_path,
            "POS ",
            ptype,
        )
        / 1e3
    )

    vel = readsnap.read_block(
        sim_path,
        "VEL ",
        ptype,
    )

    return pos, vel


# ===========================================================================
# Halo catalogue loading
# ===========================================================================

def load_data_halo(
    sim_path,
    string,
    mass_limit=1,
    mass_width=0.5,
    Remove_subhalo="yes",
    return_counts=False,
):

    result = halo_selection(
        sim_path,
        string,
        mass_limit,
        mass_width=mass_width,
        Remove_subhalo=Remove_subhalo,
        return_counts=return_counts,
    )

    if return_counts:

        (
            halo_all,
            num_halos,
            total_before_subhalo_cut,
            total_after_subhalo_cut,
        ) = result

    else:

        halo_all = result

    pos = halo_all[
        :,
        :3,
    ].astype(
        np.float32,
    )

    vel = halo_all[
        :,
        3:6,
    ].astype(
        np.float32,
    )

    mass = halo_all[
        :,
        6,
    ].astype(
        np.float32,
    )

    if return_counts:

        counts = {

            "num_halos": int(
                num_halos
            ),

            "total_halos_before_subhalo_cut": int(
                total_before_subhalo_cut
            ),

            "total_halos_after_subhalo_cut": int(
                total_after_subhalo_cut
            ),
        }

        return (
            pos,
            vel,
            mass,
            counts,
        )

    return (
        pos,
        vel,
        mass,
    )


# ===========================================================================
# Halo mass selection
# ===========================================================================

def get_mass_selection_mask(
    masses,
    string,
    mass_limit,
    mass_width=0.5,
):
    """
    Return a boolean mask for halo mass selection.
    """

    masses = np.asarray(
        masses
    )

    if string == "+":

        return (
            masses
            > mass_limit
        )

    elif string == "-":

        return (
            masses
            <= mass_limit
        )

    elif string == "all":

        return np.ones(
            masses.shape,
            dtype=bool,
        )

    elif string == "bin":

        if mass_limit <= 0:
            raise ValueError(
                "mass_limit must be positive "
                "for string='bin'."
            )

        exponent = np.floor(
            np.log10(
                mass_limit
            )
        )

        coefficient = (
            mass_limit
            / (10**exponent)
        )

        if np.isclose(
            coefficient,
            1.0,
        ):

            lower = (
                coefficient
                - 0.1 * mass_width
            ) * 10**exponent

            upper = (
                coefficient
                + mass_width
            ) * 10**exponent

        else:

            lower = (
                coefficient
                - mass_width
            ) * 10**exponent

            upper = (
                coefficient
                + mass_width
            ) * 10**exponent

        return (
            (masses >= lower)
            & (masses <= upper)
        )

    else:

        raise ValueError(
            "string must be one of "
            "'+', '-', 'bin', or 'all'."
        )


# ===========================================================================
# JD loading
# ===========================================================================

def load_data_JD_all(
    bulk_species,
    sim_path,
    sim_type,
    mass_limit=1,
    mass_width=0.5,
    string="+",
    ranknum=512,
):

    pos_data = []
    vel_data = []

    # ------------------------------------------------------------------
    # CDM
    # ------------------------------------------------------------------

    if bulk_species == 'cdm':

        for rank in range(ranknum):

            input_file = (
                sim_path
                + "/0.000xv"
                + str(rank)
                + ".dat"
            )

            file_data = ReadParticleFile(
                input_file
            )

            pos_data_add = np.vstack(
                (
                    file_data[0],
                    file_data[1],
                    file_data[2],
                )
            ).T.astype(
                np.float32,
                copy=False,
            )

            vel_data_add = np.vstack(
                (
                    file_data[3],
                    file_data[4],
                    file_data[5],
                )
            ).T.astype(
                np.float32,
                copy=False,
            )

            pos_data.append(
                pos_data_add
            )

            vel_data.append(
                vel_data_add
            )

        pos_data = np.vstack(
            pos_data
        ).astype(
            np.float32,
            copy=False,
        )

        vel_data = np.vstack(
            vel_data
        ).astype(
            np.float32,
            copy=False,
        )

        return (
            pos_data,
            vel_data,
        )

    # ------------------------------------------------------------------
    # Neutrinos
    # ------------------------------------------------------------------

    elif bulk_species == 'nu':

        for rank in range(ranknum):

            input_file = (
                sim_path
                + "/0.000xv"
                + str(rank)
                + "_nu.dat"
            )

            file_data = ReadParticleFile(
                input_file
            )

            pos_data_add = np.vstack(
                (
                    file_data[0],
                    file_data[1],
                    file_data[2],
                )
            ).T.astype(
                np.float32,
                copy=False,
            )

            vel_data_add = np.vstack(
                (
                    file_data[3],
                    file_data[4],
                    file_data[5],
                )
            ).T.astype(
                np.float32,
                copy=False,
            )

            pos_data.append(
                pos_data_add
            )

            vel_data.append(
                vel_data_add
            )

        pos_data = np.vstack(
            pos_data
        ).astype(
            np.float32,
            copy=False,
        )

        vel_data = np.vstack(
            vel_data
        ).astype(
            np.float32,
            copy=False,
        )

        return (
            pos_data,
            vel_data,
        )

    # ------------------------------------------------------------------
    # Halos
    # ------------------------------------------------------------------

    elif bulk_species == 'halo':

        mass_data = []

        if sim_type == "0.0ev":

            for rank in range(ranknum):

                input_file = (
                    sim_path
                    + "/0.000halo"
                    + str(rank)
                    + ".dat"
                )

                a = 1.0

                file_data = ReadHaloFile_lcdm(
                    input_file,
                    a,
                )

                cond = get_mass_selection_mask(
                    file_data[6],
                    string,
                    mass_limit,
                    mass_width=mass_width,
                )

                pos_data_add = np.vstack(
                    (
                        file_data[0][cond],
                        file_data[1][cond],
                        file_data[2][cond],
                    )
                ).T

                vel_data_add = np.vstack(
                    (
                        file_data[3][cond],
                        file_data[4][cond],
                        file_data[5][cond],
                    )
                ).T

                pos_data.append(
                    pos_data_add
                )

                vel_data.append(
                    vel_data_add
                )

                mass_data.append(
                    file_data[6][cond]
                )

            pos_data = np.vstack(
                pos_data
            )

            vel_data = np.vstack(
                vel_data
            )

            mass_data = np.concatenate(
                mass_data
            )

            return (
                pos_data,
                vel_data,
                mass_data,
            )

        else:

            for rank in range(ranknum):

                input_file = (
                    sim_path
                    + "/0.000halo"
                    + str(rank)
                    + ".dat"
                )

                a = 1.0

                file_data = ReadHaloFile_data(
                    input_file,
                    a,
                )

                cond = get_mass_selection_mask(
                    file_data[6],
                    string,
                    mass_limit,
                    mass_width=mass_width,
                )

                pos_data_add = np.vstack(
                    (
                        file_data[0][cond],
                        file_data[1][cond],
                        file_data[2][cond],
                    )
                ).T

                vel_data_add = np.vstack(
                    (
                        file_data[3][cond],
                        file_data[4][cond],
                        file_data[5][cond],
                    )
                ).T

                pos_data.append(
                    pos_data_add
                )

                vel_data.append(
                    vel_data_add
                )

                mass_data.append(
                    file_data[6][cond]
                )

            pos_data = np.vstack(
                pos_data
            )

            vel_data = np.vstack(
                vel_data
            )

            mass_data = np.concatenate(
                mass_data
            )

            return (
                pos_data,
                vel_data,
                mass_data,
            )

    else:

        raise ValueError(
            "bulk_species must be one of "
            "'cdm', 'nu', or 'halo'."
        )


# ===========================================================================
# Rockstar halo selection
# ===========================================================================

def halo_selection(
    data_address,
    string,
    mass_limit,
    mass_width=0.5,
    Remove_subhalo="yes",
    extra_columns=False,
    return_counts=False,
):
    """
    Filter a Rockstar halo catalogue.

    Parameters
    ----------
    data_address : str
        Path to the halo catalogue.

    string : {"+", "-", "bin", "all"}
        Halo mass-selection mode.

    mass_limit : float
        Mass threshold or bin center.

    mass_width : float
        Width parameter for mass bins.

    Remove_subhalo : {"yes", "no"}
        If "yes", require PID == -1.

    extra_columns : bool
        If True, append Rvir.

    return_counts : bool
        If True, also return catalogue counts.
    """

    data = np.loadtxt(
        data_address
    )

    data = np.atleast_2d(
        data
    )

    total_before_subhalo_cut = (
        data.shape[0]
    )

    # ------------------------------------------------------------------
    # Remove subhalos
    # ------------------------------------------------------------------

    if Remove_subhalo == "yes":

        if data.shape[1] < 42:

            raise ValueError(
                "Remove_subhalo='yes' requires a parent-processed "
                "Rockstar catalogue with PID as the last column. "
                "This file appears to have fewer than 42 columns."
            )

        pid = data[
            :,
            -1,
        ].astype(
            np.int64
        )

        if not np.any(
            pid == -1
        ):

            raise ValueError(
                "Remove_subhalo='yes' but no PID == -1 entries "
                "were found. Check that this is a parent-processed "
                "catalogue with PID as the last column."
            )

        data = data[
            pid == -1
        ]

    elif Remove_subhalo != "no":

        raise ValueError(
            "Remove_subhalo must be 'yes' or 'no'."
        )

    total_after_subhalo_cut = (
        data.shape[0]
    )

    # ------------------------------------------------------------------
    # Mass selection
    # ------------------------------------------------------------------

    condition = get_mass_selection_mask(
        data[:, 20],
        string,
        mass_limit,
        mass_width=mass_width,
    )

    selected = data[
        condition
    ]

    # ------------------------------------------------------------------
    # Output catalogue
    # ------------------------------------------------------------------

    if extra_columns:

        halo_pop = np.zeros(
            (
                selected.shape[0],
                8,
            ),
            dtype=np.float32,
        )

    else:

        halo_pop = np.zeros(
            (
                selected.shape[0],
                7,
            ),
            dtype=np.float32,
        )

    halo_pop[
        :,
        0:6,
    ] = selected[
        :,
        8:14,
    ]

    # M200b
    halo_pop[
        :,
        6,
    ] = selected[
        :,
        20,
    ]

    if extra_columns:

        # Rvir in kpc/h
        halo_pop[
            :,
            7,
        ] = selected[
            :,
            5,
        ]

    if return_counts:

        num_halos = (
            halo_pop.shape[0]
        )

        return (
            halo_pop,
            num_halos,
            total_before_subhalo_cut,
            total_after_subhalo_cut,
        )

    return halo_pop


# ===========================================================================
# Saving
# ===========================================================================

def make_smoothing_filename_tag(
    metadata,
):
    """
    Add a smoothing tag only when smoothing is present.

    This prevents runs with different sets of R from silently overwriting
    one another.
    """

    smoothing_scales = metadata.get(
        "smoothing_scales_Mpc_h",
        [],
    )

    if len(smoothing_scales) == 0:
        return ""

    scale_string = "-".join(
        f"{float(R):g}"
        for R in smoothing_scales
    )

    return (
        f"_smooth_R{scale_string}"
    )


def save_dataframe(
    df,
    simulation,
    bulk_species,
    metadata,
    ngrid,
    save_path,
):

    data_to_save = {
        'metadata': metadata,
        'sub_box_data': df,
    }

    os.makedirs(
        save_path,
        exist_ok=True,
    )

    smoothing_tag = (
        make_smoothing_filename_tag(
            metadata
        )
    )

    output_file = os.path.join(
        save_path,
        (
            f"data_ngrid_{ngrid}"
            f"_sim_{simulation}"
            f"{smoothing_tag}.pickle"
        ),
    )

    print(
        f"Saving output to:\n{output_file}",
        flush=True,
    )

    with open(
        output_file,
        'wb',
    ) as handle:

        pickle.dump(
            data_to_save,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )