"""
Extended CDM--halo--neutrino response and convergence analysis.

Primitive CIC fields stored in the bulk-velocity files are

    n_a(x)   = sum_p W_CIC(x-x_p),
    P_a,i(x) = sum_p W_CIC(x-x_p) v_p,i.

Raw reconstructed fields:

    1 + delta_a = n_a / <n_a>,
    V_a         = P_a / n_a,
    J_a         = P_a / <n_a>
                = (1+delta_a)V_a.

Physically smoothed species fields:

    n_a^(R)   = W_R * n_a,
    P_a,i^(R) = W_R * P_a,i,

    V_a^(R)   = P_a,i^(R) / n_a^(R).

Halo response fields:

    Y = (1+delta_h)(V_h-V_c)
      = J_h-(1+delta_h)V_c,

    X = (1+delta_h)(V_nu-V_c).

The code computes raw and smoothed variants and several convergence
diagnostics intended to isolate grid-resolution effects.
"""

import gc
import glob
import os
import pickle
import sys
import time

import numpy as np
import psutil

from mpi4py import MPI
from scipy import fft as sfft


# ---------------------------------------------------------------------------
# Local libraries
# ---------------------------------------------------------------------------

sys.path.append(
    "/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/"
)

sys.path.append(
    "/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/"
    "nu_code/library_snapshot"
)

import Pk_library as PKL


_COMPONENTS = ("x", "y", "z")


# ===========================================================================
# Basic I/O
# ===========================================================================

def load(filename):
    """Load one pickle file."""

    with open(filename, "rb") as handle:
        return pickle.load(handle)


# ===========================================================================
# Main driver
# ===========================================================================

def pairwise_power_spectra_pylians(
    boxsize,
    sim_type,
    spec,
    ngrid_min,
    ngrid_max,
    ngrid_step,
    file_path,
    string,
    mass_cut,
    mass_width,
    save_path,
    save_density_diagnostics=True,
    save_species_diagnostics=True,
    compute_convergence_diagnostics=True,
    compute_smoothed_responses=True,
    constructed_smoothing_method="spectrum",
    fft_workers=4,
):
    """
    Compute raw/smoothed response spectra and convergence diagnostics.

    Parameters
    ----------
    compute_smoothed_responses : bool
    If True, compute same-R smoothed response variants for all
    available common halo/CDM/neutrino smoothing scales, using both
    the raw and smoothed halo-density weighting.

    fft_workers : int
        Number of scipy.fft workers used for spectral divergences.
    """

    boxsize = float(boxsize)

    validate_inputs(
        string,
        ngrid_step,
    )

    start_time_all = time.time()
    start_mem_all = psutil.Process().memory_info().rss

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    loop_one_sided_spectra_computation(
        rank=rank,
        size=size,
        boxsize=boxsize,
        sim_type=sim_type,
        spec=spec,
        ngrid_min=ngrid_min,
        ngrid_max=ngrid_max,
        ngrid_step=ngrid_step,
        file_path=file_path,
        string=string,
        mass_cut=mass_cut,
        mass_width=mass_width,
        save_path=save_path,
        save_density_diagnostics=save_density_diagnostics,
        save_species_diagnostics=save_species_diagnostics,
        compute_convergence_diagnostics=compute_convergence_diagnostics,
        compute_smoothed_responses=compute_smoothed_responses,
        constructed_smoothing_method=constructed_smoothing_method,
        fft_workers=fft_workers,
    )

    comm.Barrier()

    if rank == 0:

        print_usage(
            start_time_all,
            start_mem_all,
            "- Total time and memory",
        )

        print(
            "\n*********** "
            "All response/convergence spectra finished! "
            "***********\n",
            flush=True,
        )


# ===========================================================================
# Ngrid MPI loop
# ===========================================================================

def loop_one_sided_spectra_computation(
    rank,
    size,
    boxsize,
    sim_type,
    spec,
    ngrid_min,
    ngrid_max,
    ngrid_step,
    file_path,
    string,
    mass_cut,
    mass_width,
    save_path,
    save_density_diagnostics,
    save_species_diagnostics,
    compute_convergence_diagnostics,
    compute_smoothed_responses,
    constructed_smoothing_method,
    fft_workers,
):

    ngrid_list = list(
        range(
            ngrid_min,
            ngrid_max,
            ngrid_step,
        )
    )

    ngrid_per_rank = np.array_split(
        ngrid_list,
        size,
    )

    for ngrid_value in ngrid_per_rank[rank]:

        ngrid = int(ngrid_value)

        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss

        print(
            f"[rank {rank}] Starting ngrid={ngrid}",
            flush=True,
        )

        power_data = compute_joint_one_sided_spectra(
            file_path=file_path,
            spec=spec,
            ngrid=ngrid,
            sim=sim_type,
            boxsize=boxsize,
            string=string,
            mass_cut=mass_cut,
            mass_width=mass_width,
            save_density_diagnostics=save_density_diagnostics,
            save_species_diagnostics=save_species_diagnostics,
            compute_convergence_diagnostics=(
                compute_convergence_diagnostics
            ),
            compute_smoothed_responses=(
                compute_smoothed_responses
            ),
            constructed_smoothing_method=(
                constructed_smoothing_method
            ),
            fft_workers=fft_workers,
        )

        simulation = simulation_def(
            spec=spec,
            sim_type=sim_type,
            string=string,
            mass_cut=mass_cut,
            mass_width=mass_width,
        )

        save_and_print_usage(
            start_time=start_time,
            start_mem=start_mem,
            simulation=simulation,
            ngrid=ngrid,
            power_data=power_data,
            save_path=save_path,
        )

        del power_data
        gc.collect()


# ===========================================================================
# Main computation
# ===========================================================================

def compute_joint_one_sided_spectra(
    file_path,
    spec,
    ngrid,
    sim,
    boxsize,
    string,
    mass_cut=1.0,
    mass_width=0.5,
    save_density_diagnostics=True,
    save_species_diagnostics=True,
    compute_convergence_diagnostics=True,
    compute_smoothed_responses=True,
    constructed_smoothing_method="spectrum",
    fft_workers=4,
):

    is_lcdm = sim == "0.0ev"

    # ===================================================================
    # Halo
    # ===================================================================

    data_h = load_files_pickles(
        file_path=file_path,
        bulk_species="halo",
        spec=spec,
        ngrid=ngrid,
        sim=sim,
        string=string,
        mass_cut=mass_cut,
        mass_width=mass_width,
    )

    h = build_halo_fields(
        data_h
    )

    del data_h
    gc.collect()

    # ===================================================================
    # CDM
    # ===================================================================

    data_c = load_files_pickles(
        file_path=file_path,
        bulk_species="cdm",
        spec=spec,
        ngrid=ngrid,
        sim=sim,
        string=string,
        mass_cut=mass_cut,
        mass_width=mass_width,
    )

    c = build_matter_fields(
        data_c,
        field_label="cdm",
    )

    del data_c
    gc.collect()

    # ===================================================================
    # Neutrinos
    # ===================================================================

    if is_lcdm:

        nu = build_zero_neutrino_fields_like(
            c
        )

    else:

        data_nu = load_files_pickles(
            file_path=file_path,
            bulk_species="nu",
            spec=spec,
            ngrid=ngrid,
            sim=sim,
            string=string,
            mass_cut=mass_cut,
            mass_width=mass_width,
        )

        nu = build_matter_fields(
            data_nu,
            field_label="nu",
        )

        del data_nu
        gc.collect()

    # ===================================================================
    # Output
    # ===================================================================

    power_data = {

        "metadata": build_metadata(
            spec=spec,
            sim=sim,
            ngrid=ngrid,
            string=string,
            mass_cut=mass_cut,
            mass_width=mass_width,
            boxsize=boxsize,
            is_lcdm=is_lcdm,
            save_density_diagnostics=save_density_diagnostics,
            save_species_diagnostics=save_species_diagnostics,
            compute_smoothed_responses=compute_smoothed_responses,
            compute_convergence_diagnostics=(
                compute_convergence_diagnostics
            ),
        ),

        "field_summary": {
            "halo": h["summary"],
            "cdm": c["summary"],
            "nu": nu["summary"],
        },

        "available_velocity_fields": {
        
            "halo": {
                "raw": True,
                "smoothed": tuple(
                    sorted_R_keys(
                        h["smoothed"].keys()
                    )
                ),
            },
        
            "cdm": {
                "raw": True,
                "smoothed": tuple(
                    sorted_R_keys(
                        c["smoothed"].keys()
                    )
                ),
            },
        
            "nu": {
                "raw": True,
                "smoothed": tuple(
                    sorted_R_keys(
                        nu["smoothed"].keys()
                    )
                ),
                "lcdm_zero_field": bool(
                    nu.get(
                        "is_zero_neutrino",
                        False,
                    )
                ),
            },
        },
    }

    # ===================================================================
    # Raw response
    # ===================================================================

    Y_raw = build_Y_variant(
        h,
        c,
    )

    X_raw = build_X_variant(
        h,
        c,
        nu,
    )

    div_Y_raw = fft_divergence(
        Y_raw,
        boxsize,
        workers=fft_workers,
    )

    div_X_raw = fft_divergence(
        X_raw,
        boxsize,
        workers=fft_workers,
    )

    del Y_raw
    del X_raw

    response_raw = (
        compute_one_sided_response_spectra(
            delta_c=c["delta"],
            div_Y=div_Y_raw,
            div_X=div_X_raw,
            boxsize=boxsize,
            physical_neutrino_driver_available=(
                not is_lcdm
            ),
            lcdm_zero_neutrino_convention=(
                is_lcdm
            ),
        )
    )

    power_data[
        "one_sided_cdm_halo_response"
    ] = response_raw

    # ===================================================================
    # Preserve old decomposition analysis
    # ===================================================================

    power_data[
        "response_decomposition"
    ] = compute_response_decomposition_spectra(
        h=h,
        c=c,
        nu=nu,
        delta_c=c["delta"],
        full_response=response_raw,
        boxsize=boxsize,
        fft_workers=fft_workers,
    )

    # ===================================================================
    # Smoothed X/Y variants
    #
    # All halo, CDM, and neutrino velocities use the same physical R.
    # We save versions with raw and smoothed halo-density weighting.
    # ===================================================================

    if compute_smoothed_responses:

        power_data[
            "response_variants"
        ] = compute_response_variants(
            h=h,
            c=c,
            nu=nu,
            delta_c=c["delta"],
            div_Y_raw=div_Y_raw,
            div_X_raw=div_X_raw,
            raw_response=response_raw,
            boxsize=boxsize,
            is_lcdm=is_lcdm,
            constructed_smoothing_method=(
                constructed_smoothing_method
            ),
            fft_workers=fft_workers,
        )

    # Raw divergences are no longer needed after all response variants
    # have been constructed.
    del div_Y_raw
    del div_X_raw
    gc.collect()

    # ===================================================================
    # Convergence tests
    # ===================================================================

    if compute_convergence_diagnostics:

        power_data[
            "convergence_diagnostics"
        ] = compute_convergence_tests(
            h=h,
            c=c,
            nu=nu,
            boxsize=boxsize,
        )

    # ===================================================================
    # Existing optional diagnostics
    # ===================================================================

    if save_species_diagnostics:

        power_data[
            "species_diagnostics"
        ] = compute_species_velocity_diagnostics(
            h=h,
            c=c,
            nu=nu,
            boxsize=boxsize,
        )

    elif save_density_diagnostics:

        power_data[
            "density_spectra"
        ] = compute_density_diagnostics(
            h=h,
            c=c,
            nu=nu,
            boxsize=boxsize,
        )

    del h
    del c
    del nu

    gc.collect()

    return power_data


# ===========================================================================
# Field construction
# ===========================================================================

def build_halo_fields(data):
    """
    Construct halo density, velocity and current.

        V_h = P_h / n_h
        J_h = P_h / <n_h>
            = (1+delta_h)V_h

    V_h is set to zero in cells with no halo CIC support.
    """

    sub = data["sub_box_data"]
    metadata = data.get(
        "metadata",
        {},
    )

    density = np.asarray(
        sub["density"],
        dtype=np.float32,
    )

    mean_density = validate_mean_density(
        density,
        "halo",
    )

    occupied = density > 0.0

    one_plus_delta = (
        density / mean_density
    ).astype(
        np.float32
    )

    delta = (
        one_plus_delta - 1.0
    ).astype(
        np.float32
    )

    summary = summarize_density(
        density,
        mean_density,
        "halo",
    )

    total_halos_metadata = metadata.get(
        "num_halos",
        None,
    )

    summary.update(
        {
            "CIC_total_halo_weight": float(
                np.sum(
                    density,
                    dtype=np.float64,
                )
            ),

            "number_cells": int(
                density.size
            ),

            "number_occupied_cells": int(
                np.count_nonzero(
                    occupied
                )
            ),

            "occupied_cell_fraction": float(
                np.mean(
                    occupied
                )
            ),

            "fraction_cells_CIC_weight_ge_1": float(
                np.mean(
                    density >= 1.0
                )
            ),

            "fraction_cells_CIC_weight_ge_2": float(
                np.mean(
                    density >= 2.0
                )
            ),
        }
    )

    if total_halos_metadata is not None:

        summary[
            "num_halos_metadata"
        ] = int(
            total_halos_metadata
        )

        summary[
            "mean_halos_per_cell"
        ] = (
            float(
                total_halos_metadata
            )
            / float(
                density.size
            )
        )

    fields = {
        "label": "halo",
        "one_plus_delta": one_plus_delta,
        "delta": delta,
        "summary": summary,
        "smoothed": {},
    }

    primitive_sums = {
        "density": float(
            np.sum(
                density,
                dtype=np.float64,
            )
        )
    }

    for comp in _COMPONENTS:

        p_comp = np.asarray(
            sub[f"P_{comp}"],
            dtype=np.float32,
        )

        J_comp = (
            p_comp / mean_density
        ).astype(
            np.float32
        )

        V_comp = np.zeros_like(
            p_comp,
            dtype=np.float32,
        )

        np.divide(
            p_comp,
            density,
            out=V_comp,
            where=occupied,
        )

        fields[
            f"J_{comp}"
        ] = J_comp

        fields[
            f"V_{comp}"
        ] = V_comp

        primitive_sums[
            f"P_{comp}"
        ] = float(
            np.sum(
                p_comp,
                dtype=np.float64,
            )
        )

    fields[
        "primitive_sums"
    ] = primitive_sums
    
    
    # ------------------------------------------------------------------
    # Smoothed halo primitive fields
    #
    # V_h^(R) = P_h^(R) / n_h^(R)
    # ------------------------------------------------------------------
    
    smoothed_input = sub.get(
        "smoothed",
        {},
    )
    
    for R_key in sorted_R_keys(
        smoothed_input.keys()
    ):
    
        smooth_data = smoothed_input[R_key]
    
        density_R = np.asarray(
            smooth_data["density"],
            dtype=np.float32,
        )
    
        occupied_R = density_R > 0.0

        one_plus_delta_R = (
            density_R / mean_density
        ).astype(
            np.float32
        )
        smooth = {
            "one_plus_delta": one_plus_delta_R,
        
            "summary": {
                "mean_density": float(
                    np.mean(
                        density_R,
                        dtype=np.float64,
                    )
                ),
        
                "density_sum": float(
                    np.sum(
                        density_R,
                        dtype=np.float64,
                    )
                ),
        
                "empty_cell_fraction": float(
                    np.mean(
                        ~occupied_R
                    )
                ),
            }
        }    

    
        for comp in _COMPONENTS:
    
            p_R = np.asarray(
                smooth_data[f"P_{comp}"],
                dtype=np.float32,
            )
    
            V_R = np.zeros_like(
                p_R,
                dtype=np.float32,
            )
    
            np.divide(
                p_R,
                density_R,
                out=V_R,
                where=occupied_R,
            )
    
            smooth[f"V_{comp}"] = V_R
    
            smooth[f"P_sum_{comp}"] = float(
                np.sum(
                    p_R,
                    dtype=np.float64,
                )
            )
    
        fields["smoothed"][R_key] = smooth
    
    
    fields[
        "summary"
    ][
        "available_smoothing_scales"
    ] = tuple(
        fields["smoothed"].keys()
    )
    
    
    return fields


def build_matter_fields(
    data,
    field_label,
):
    """
    Construct raw and Gaussian-smoothed matter fields.

    Raw:
        V = P/n
        J = P/<n>

    Smoothed:
        V^(R) = P^(R)/n^(R).

    Smoothed primitive fields themselves are NOT copied unnecessarily;
    reconstructed velocities are held in the returned dictionary.
    """

    sub = data[
        "sub_box_data"
    ]

    density = np.asarray(
        sub["density"],
        dtype=np.float32,
    )

    mean_density = validate_mean_density(
        density,
        field_label,
    )

    occupied = density > 0.0

    one_plus_delta = (
        density / mean_density
    ).astype(
        np.float32
    )

    delta = (
        one_plus_delta - 1.0
    ).astype(
        np.float32
    )

    fields = {
        "label": field_label,
        "one_plus_delta": one_plus_delta,
        "delta": delta,
        "summary": summarize_density(
            density,
            mean_density,
            field_label,
        ),
        "smoothed": {},
        "is_zero_neutrino": False,
    }

    primitive_sums = {
        "density": float(
            np.sum(
                density,
                dtype=np.float64,
            )
        )
    }

    # ------------------------------------------------------------------
    # Raw
    # ------------------------------------------------------------------

    for comp in _COMPONENTS:

        p_comp = np.asarray(
            sub[f"P_{comp}"],
            dtype=np.float32,
        )

        v_comp = np.zeros_like(
            p_comp,
            dtype=np.float32,
        )

        np.divide(
            p_comp,
            density,
            out=v_comp,
            where=occupied,
        )

        fields[
            f"V_{comp}"
        ] = v_comp

        fields[
            f"J_{comp}"
        ] = (
            p_comp
            / mean_density
        ).astype(
            np.float32
        )

        primitive_sums[
            f"P_{comp}"
        ] = float(
            np.sum(
                p_comp,
                dtype=np.float64,
            )
        )

    fields[
        "primitive_sums"
    ] = primitive_sums

    # ------------------------------------------------------------------
    # Smoothed primitive fields
    # ------------------------------------------------------------------

    smoothed_input = sub.get(
        "smoothed",
        {},
    )

    for R_key in sorted_R_keys(
        smoothed_input.keys()
    ):

        smooth_data = (
            smoothed_input[
                R_key
            ]
        )

        density_R = np.asarray(
            smooth_data[
                "density"
            ],
            dtype=np.float32,
        )

        occupied_R = (
            density_R > 0.0
        )

        smooth = {
            "summary": {
                "mean_density": float(
                    np.mean(
                        density_R,
                        dtype=np.float64,
                    )
                ),

                "density_sum": float(
                    np.sum(
                        density_R,
                        dtype=np.float64,
                    )
                ),

                "empty_cell_fraction": float(
                    np.mean(
                        ~occupied_R
                    )
                ),
            }
        }

        for comp in _COMPONENTS:

            p_R = np.asarray(
                smooth_data[
                    f"P_{comp}"
                ],
                dtype=np.float32,
            )

            V_R = np.zeros_like(
                p_R,
                dtype=np.float32,
            )

            np.divide(
                p_R,
                density_R,
                out=V_R,
                where=occupied_R,
            )

            smooth[
                f"V_{comp}"
            ] = V_R

            smooth[
                f"P_sum_{comp}"
            ] = float(
                np.sum(
                    p_R,
                    dtype=np.float64,
                )
            )

        fields[
            "smoothed"
        ][
            R_key
        ] = smooth

    fields[
        "summary"
    ][
        "available_smoothing_scales"
    ] = tuple(
        fields[
            "smoothed"
        ].keys()
    )

    return fields


def build_zero_neutrino_fields_like(
    reference_field,
):
    """
    LCDM convention:

        delta_nu = 0
        V_nu = 0
        J_nu = 0.

    A requested smoothing radius still corresponds to zero velocity.
    """

    shape = (
        reference_field[
            "delta"
        ].shape
    )

    zero = np.zeros(
        shape,
        dtype=np.float32,
    )

    one = np.ones(
        shape,
        dtype=np.float32,
    )

    fields = {
        "label": "nu_zero_lcdm",
        "one_plus_delta": one,
        "delta": zero,

        "summary": {
            "label": "nu_zero_lcdm",
            "mean_density": 1.0,
            "density_min": 1.0,
            "density_max": 1.0,
            "empty_cell_fraction": 0.0,
            "lcdm_zero_neutrino_convention": True,
            "available_smoothing_scales": (),
        },

        "smoothed": {},

        "is_zero_neutrino": True,

        "primitive_sums": {
            "density": float(
                zero.size
            ),
            "P_x": 0.0,
            "P_y": 0.0,
            "P_z": 0.0,
        },
    }

    for comp in _COMPONENTS:

        fields[
            f"V_{comp}"
        ] = zero

        fields[
            f"J_{comp}"
        ] = zero

    return fields


# ===========================================================================
# Velocity source helpers
# ===========================================================================

def sorted_R_keys(keys):
    """Sort keys such as R_5, R_10, R_16 numerically."""

    return sorted(
        keys,
        key=lambda key: float(
            str(key).replace(
                "R_",
                "",
            )
        ),
    )


def get_velocity_source(
    species,
    R_key=None,
):
    """
    Return raw or smoothed velocity bundle.

    Synthetic LCDM neutrinos remain zero for every R.
    """

    if R_key is None:
        return species

    if species.get(
        "is_zero_neutrino",
        False,
    ):
        return species

    if R_key not in species[
        "smoothed"
    ]:
        raise KeyError(
            f"Smoothing scale {R_key!r} "
            f"is unavailable for {species['label']}."
        )

    return species[
        "smoothed"
    ][
        R_key
    ]


# ===========================================================================
# Physical vector fields
# ===========================================================================

def build_Y_variant(
    h,
    c,
    R_key=None,
    smooth_halo_weight=False,
):
    """
    Construct the halo-CDM response field.

    Raw:
        Y = (1+delta_h)(V_h-V_c)
          = J_h-(1+delta_h)V_c

    Common smoothing scale R:
        Y_R = W_h (V_h^R-V_c^R)

    where

        W_h = 1+delta_h

    if smooth_halo_weight=False, and

        W_h = 1+delta_h^R

    if smooth_halo_weight=True.
    """

    # ---------------------------------------------------------------
    # Completely raw case
    # ---------------------------------------------------------------

    if R_key is None:

        one_h = h[
            "one_plus_delta"
        ]

        return tuple(

            (
                h[f"J_{comp}"]
                - one_h
                * c[
                    f"V_{comp}"
                ]
            ).astype(
                np.float32
            )

            for comp in _COMPONENTS
        )

    # ---------------------------------------------------------------
    # Same-R smoothed velocities
    # ---------------------------------------------------------------

    h_source = get_velocity_source(
        h,
        R_key,
    )

    c_source = get_velocity_source(
        c,
        R_key,
    )

    if smooth_halo_weight:

        one_h = h[
            "smoothed"
        ][
            R_key
        ][
            "one_plus_delta"
        ]

    else:

        one_h = h[
            "one_plus_delta"
        ]

    return tuple(

        (
            one_h
            * (
                h_source[
                    f"V_{comp}"
                ]
                - c_source[
                    f"V_{comp}"
                ]
            )
        ).astype(
            np.float32
        )

        for comp in _COMPONENTS
    )


def build_X_variant(
    h,
    c,
    nu,
    R_key=None,
    smooth_halo_weight=False,
):
    """
    Construct the neutrino-CDM driver field.

    Raw:
        X = (1+delta_h)(V_nu-V_c)

    Common smoothing scale R:
        X_R = W_h (V_nu^R-V_c^R)

    where

        W_h = 1+delta_h

    if smooth_halo_weight=False, and

        W_h = 1+delta_h^R

    if smooth_halo_weight=True.
    """

    # ---------------------------------------------------------------
    # Raw
    # ---------------------------------------------------------------

    if R_key is None:

        one_h = h[
            "one_plus_delta"
        ]

        c_source = c
        nu_source = nu

    # ---------------------------------------------------------------
    # Same-R smoothing
    # ---------------------------------------------------------------

    else:

        c_source = get_velocity_source(
            c,
            R_key,
        )

        nu_source = get_velocity_source(
            nu,
            R_key,
        )

        if smooth_halo_weight:

            one_h = h[
                "smoothed"
            ][
                R_key
            ][
                "one_plus_delta"
            ]

        else:

            one_h = h[
                "one_plus_delta"
            ]

    return tuple(

        (
            one_h
            * (
                nu_source[
                    f"V_{comp}"
                ]
                - c_source[
                    f"V_{comp}"
                ]
            )
        ).astype(
            np.float32
        )

        for comp in _COMPONENTS
    )

def build_halo_weighted_velocity(
    h,
    species,
    R_key=None,
):
    """
    Return

        H_a = (1+delta_h)V_a

    for convergence diagnostics.
    """

    one_h = h[
        "one_plus_delta"
    ]

    source = get_velocity_source(
        species,
        R_key,
    )

    return tuple(

        (
            one_h
            * source[
                f"V_{comp}"
            ]
        ).astype(
            np.float32
        )

        for comp in _COMPONENTS
    )


def get_current_vector(
    species,
):
    """Return J=(Jx,Jy,Jz)."""

    return tuple(
        species[
            f"J_{comp}"
        ]
        for comp in _COMPONENTS
    )


def get_velocity_vector(
    species,
    R_key=None,
):
    """Return raw or smoothed V vector."""

    source = get_velocity_source(
        species,
        R_key,
    )

    return tuple(
        source[
            f"V_{comp}"
        ]
        for comp in _COMPONENTS
    )


# ===========================================================================
# Raw response decomposition
# ===========================================================================

def build_X_unweighted_components(
    c,
    nu,
):

    return tuple(
        (
            nu[f"V_{comp}"]
            - c[f"V_{comp}"]
        ).astype(
            np.float32,
            copy=False,
        )
        for comp in _COMPONENTS
    )


def build_Y_unweighted_components(
    h,
    c,
):

    one_plus_delta_h = (
        h[
            "one_plus_delta"
        ]
    )

    occupied = (
        one_plus_delta_h > 0.0
    )

    components = []

    for comp in _COMPONENTS:

        component = np.zeros_like(
            h[
                f"J_{comp}"
            ],
            dtype=np.float32,
        )

        np.divide(
            h[
                f"J_{comp}"
            ],
            one_plus_delta_h,
            out=component,
            where=occupied,
        )

        component -= c[
            f"V_{comp}"
        ]

        components.append(
            component
        )

    return tuple(
        components
    )


def compute_response_decomposition_spectra(
    h,
    c,
    nu,
    delta_c,
    full_response,
    boxsize,
    fft_workers=1,
):

    # ------------------------------------------------------------------
    # X unweighted
    # ------------------------------------------------------------------

    X_unweighted = (
        build_X_unweighted_components(
            c,
            nu,
        )
    )

    div_X_unweighted = (
        fft_divergence(
            X_unweighted,
            boxsize,
            workers=fft_workers,
        )
    )

    del X_unweighted

    spec_X_unweighted = (
        scalar_cross_spectrum(
            delta_c,
            div_X_unweighted,
            boxsize,
            MAS="None",
        )
    )

    del div_X_unweighted

    # ------------------------------------------------------------------
    # Y unweighted
    # ------------------------------------------------------------------

    Y_unweighted = (
        build_Y_unweighted_components(
            h,
            c,
        )
    )

    div_Y_unweighted = (
        fft_divergence(
            Y_unweighted,
            boxsize,
            workers=fft_workers,
        )
    )

    del Y_unweighted

    spec_Y_unweighted = (
        scalar_cross_spectrum(
            delta_c,
            div_Y_unweighted,
            boxsize,
            MAS="None",
        )
    )

    del div_Y_unweighted

    assert_same_binning(
        spec_X_unweighted,
        spec_Y_unweighted,
        labels=(
            "delta-divX-unweighted",
            "delta-divY-unweighted",
        ),
    )

    assert_matching_k_and_modes(
        reference_k=full_response[
            "k_h_per_Mpc"
        ],
        reference_modes=full_response[
            "Nmodes"
        ],
        test_k=spec_X_unweighted[0],
        test_modes=spec_X_unweighted[4],
        label="delta-divX-unweighted",
    )

    assert_matching_k_and_modes(
        reference_k=full_response[
            "k_h_per_Mpc"
        ],
        reference_modes=full_response[
            "Nmodes"
        ],
        test_k=spec_Y_unweighted[0],
        test_modes=spec_Y_unweighted[4],
        label="delta-divY-unweighted",
    )

    P_X_full = np.asarray(
        full_response[
            "P_delta_c_divX"
        ]
    )

    P_Y_full = np.asarray(
        full_response[
            "P_delta_c_divY"
        ]
    )

    P_X_unweighted = np.asarray(
        spec_X_unweighted[3]
    )

    P_Y_unweighted = np.asarray(
        spec_Y_unweighted[3]
    )

    return {
        "k_h_per_Mpc": np.asarray(
            full_response[
                "k_h_per_Mpc"
            ]
        ),

        "Nmodes": np.asarray(
            full_response[
                "Nmodes"
            ]
        ),

        "P_delta_c_divX_full": (
            P_X_full
        ),

        "P_delta_c_divX_unweighted": (
            P_X_unweighted
        ),

        "P_delta_c_divX_weighted": (
            P_X_full
            - P_X_unweighted
        ),

        "P_delta_c_divY_full": (
            P_Y_full
        ),

        "P_delta_c_divY_unweighted": (
            P_Y_unweighted
        ),

        "P_delta_c_divY_weighted": (
            P_Y_full
            - P_Y_unweighted
        ),

        "definitions": {
            "X_unweighted": "V_nu-V_c",
            "X_weighted": (
                "delta_h*(V_nu-V_c)"
            ),
            "Y_unweighted": "V_h-V_c",
            "Y_weighted": (
                "delta_h*(V_h-V_c)"
            ),
        },
    }


# ===========================================================================
# Smoothed response variants
# ===========================================================================
def compute_response_variants(
    h,
    c,
    nu,
    delta_c,
    div_Y_raw,
    div_X_raw,
    raw_response,
    boxsize,
    is_lcdm,
    constructed_smoothing_method="spectrum",
    fft_workers=1,
):
    """
    Compute only the response variants used in the final analysis.

    Variants
    --------

    raw:
        No smoothing anywhere.

        X = (1+delta_h)(V_nu-V_c)
        Y = (1+delta_h)(V_h-V_c)

    same_R_raw_halo_weight:
        Halo, CDM and neutrino velocities are all smoothed
        using the same physical radius R, while the external
        halo-density weighting remains unsmoothed.

        X_R = (1+delta_h)(V_nu^R-V_c^R)
        Y_R = (1+delta_h)(V_h^R-V_c^R)

    same_R_smoothed_halo_weight:
        Halo, CDM and neutrino velocities are all smoothed
        using the same R, and the halo-density weighting is
        also smoothed with the same R.

        X_R = (1+delta_h^R)(V_nu^R-V_c^R)
        Y_R = (1+delta_h^R)(V_h^R-V_c^R)
        
    constructed_field_smoothed:
        First construct the complete raw halo-weighted response fields,

            X = (1+delta_h)(V_nu-V_c)
            Y = (1+delta_h)(V_h-V_c),

        and only then Gaussian smooth the complete fields:

            X_R = W_R * X
            Y_R = W_R * Y.

        Since differentiation commutes with convolution, the
        implementation applies W_R directly to the previously computed
        raw divergence fields:

            div(X_R) = W_R * div(X)
            div(Y_R) = W_R * div(Y).
    """

    if constructed_smoothing_method not in {
        "spectrum",
        "field",
    }:
        raise ValueError(
            "constructed_smoothing_method must be "
            "'spectrum' or 'field'; got "
            f"{constructed_smoothing_method!r}"
        )

    out = {

        "raw": raw_response,

        "same_R_raw_halo_weight": {},

        "same_R_smoothed_halo_weight": {},

        "constructed_field_smoothed": {},

        "definitions": {

            "raw": (
                "X=(1+delta_h)(V_nu-V_c), "
                "Y=(1+delta_h)(V_h-V_c)"
            ),

            "same_R_raw_halo_weight": (
                "X=(1+delta_h)(V_nu^R-V_c^R), "
                "Y=(1+delta_h)(V_h^R-V_c^R)"
            ),

            "same_R_smoothed_halo_weight": (
                "X=(1+delta_h^R)(V_nu^R-V_c^R), "
                "Y=(1+delta_h^R)(V_h^R-V_c^R)"
            ),
            "constructed_field_smoothed": (
                "X_R=W_R*[(1+delta_h)(V_nu-V_c)], "
                "Y_R=W_R*[(1+delta_h)(V_h-V_c)]"
            ),
        },
    }

    # ------------------------------------------------------------------
    # Only use smoothing radii available for all relevant species.
    # ------------------------------------------------------------------

    h_scales = set(
        h[
            "smoothed"
        ].keys()
    )

    c_scales = set(
        c[
            "smoothed"
        ].keys()
    )

    if is_lcdm:

        # Synthetic neutrino velocity is zero at every R.
        common_scales = sorted_R_keys(
            h_scales
            & c_scales
        )

    else:

        nu_scales = set(
            nu[
                "smoothed"
            ].keys()
        )

        common_scales = sorted_R_keys(
            h_scales
            & c_scales
            & nu_scales
        )

    # ===================================================================
    # Same physical smoothing radius for all species
    # ===================================================================

    for R_key in common_scales:

        # ---------------------------------------------------------------
        # Version 1:
        # smoothed velocities, raw halo-density weighting
        # ---------------------------------------------------------------

        Y_R_raw_weight = build_Y_variant(
            h,
            c,
            R_key=R_key,
            smooth_halo_weight=False,
        )

        X_R_raw_weight = build_X_variant(
            h,
            c,
            nu,
            R_key=R_key,
            smooth_halo_weight=False,
        )

        div_Y_R_raw_weight = fft_divergence(
            Y_R_raw_weight,
            boxsize,
            workers=fft_workers,
        )

        div_X_R_raw_weight = fft_divergence(
            X_R_raw_weight,
            boxsize,
            workers=fft_workers,
        )

        del Y_R_raw_weight
        del X_R_raw_weight

        out[
            "same_R_raw_halo_weight"
        ][
            R_key
        ] = compute_one_sided_response_spectra(
            delta_c=delta_c,
            div_Y=div_Y_R_raw_weight,
            div_X=div_X_R_raw_weight,
            boxsize=boxsize,
            physical_neutrino_driver_available=(
                not is_lcdm
            ),
            lcdm_zero_neutrino_convention=(
                is_lcdm
            ),
        )

        del div_Y_R_raw_weight
        del div_X_R_raw_weight

        # ---------------------------------------------------------------
        # Version 2:
        # smoothed velocities AND smoothed halo-density weighting
        # ---------------------------------------------------------------

        Y_R_smooth_weight = build_Y_variant(
            h,
            c,
            R_key=R_key,
            smooth_halo_weight=True,
        )

        X_R_smooth_weight = build_X_variant(
            h,
            c,
            nu,
            R_key=R_key,
            smooth_halo_weight=True,
        )

        div_Y_R_smooth_weight = fft_divergence(
            Y_R_smooth_weight,
            boxsize,
            workers=fft_workers,
        )

        div_X_R_smooth_weight = fft_divergence(
            X_R_smooth_weight,
            boxsize,
            workers=fft_workers,
        )

        del Y_R_smooth_weight
        del X_R_smooth_weight

        out[
            "same_R_smoothed_halo_weight"
        ][
            R_key
        ] = compute_one_sided_response_spectra(
            delta_c=delta_c,
            div_Y=div_Y_R_smooth_weight,
            div_X=div_X_R_smooth_weight,
            boxsize=boxsize,
            physical_neutrino_driver_available=(
                not is_lcdm
            ),
            lcdm_zero_neutrino_convention=(
                is_lcdm
            ),
        )

        del div_Y_R_smooth_weight
        del div_X_R_smooth_weight

        # ---------------------------------------------------------------
        # Version 3:
        # Gaussian smoothing of the FULLY CONSTRUCTED X/Y fields.
        #
        #     X_R = W_R * [(1+delta_h)(V_nu-V_c)]
        #     Y_R = W_R * [(1+delta_h)(V_h-V_c)]
        #
        # Two implementations are available:
        #
        #   spectrum : fast production path. Apply the Gaussian window
        #              directly to the already measured raw spectra.
        #
        #   field    : exact mode-level validation path. Smooth the
        #              already computed raw divergence fields and then
        #              recompute the spectra.
        # ---------------------------------------------------------------

        R = float(
            str(R_key).replace(
                "R_",
                "",
            )
        )

        if constructed_smoothing_method == "spectrum":

            out[
                "constructed_field_smoothed"
            ][
                R_key
            ] = constructed_smoothed_response_from_raw_spectra(
                raw_response=raw_response,
                R=R,
            )

        elif constructed_smoothing_method == "field":

            div_Y_constructed_R = gaussian_smooth_scalar(
                div_Y_raw,
                boxsize,
                R,
                workers=fft_workers,
            )

            div_X_constructed_R = gaussian_smooth_scalar(
                div_X_raw,
                boxsize,
                R,
                workers=fft_workers,
            )

            out[
                "constructed_field_smoothed"
            ][
                R_key
            ] = compute_one_sided_response_spectra(
                delta_c=delta_c,
                div_Y=div_Y_constructed_R,
                div_X=div_X_constructed_R,
                boxsize=boxsize,
                physical_neutrino_driver_available=(
                    not is_lcdm
                ),
                lcdm_zero_neutrino_convention=(
                    is_lcdm
                ),
            )

            out[
                "constructed_field_smoothed"
            ][
                R_key
            ][
                "constructed_smoothing"
            ] = {
                "R_Mpc_over_h": R,
                "window": (
                    "Gaussian W_R(k)=exp[-(kR)^2/2]"
                ),
                "implementation": (
                    "exact_mode_level_field_smoothing"
                ),
            }

            del div_Y_constructed_R
            del div_X_constructed_R

        gc.collect()

    
    gc.collect()
    return out



def constructed_smoothed_response_from_raw_spectra(
    raw_response,
    R,
):
    """
    Construct spectra for

        X_R = W_R * X
        Y_R = W_R * Y

    directly from the already measured raw spectra, using

        W_R(k) = exp[-(k R)^2 / 2].

    This avoids any additional 3-D FFTs or real-space fields.

    The raw spectra are already shell-binned by Pylians, so the
    Gaussian window is evaluated at the reported bin-center k.
    """

    k = np.asarray(
        raw_response[
            "k_h_per_Mpc"
        ],
        dtype=np.float64,
    )

    R = float(R)

    W = np.exp(
        -0.5 * (k * R) ** 2
    )

    W2 = W * W

    return {
        "beta_estimator_available": (
            raw_response[
                "beta_estimator_available"
            ]
        ),

        "physical_neutrino_driver_available": (
            raw_response[
                "physical_neutrino_driver_available"
            ]
        ),

        "lcdm_zero_neutrino_convention": (
            raw_response[
                "lcdm_zero_neutrino_convention"
            ]
        ),

        "k_h_per_Mpc": np.asarray(
            raw_response[
                "k_h_per_Mpc"
            ]
        ),

        "P_delta_c_delta_c": np.asarray(
            raw_response[
                "P_delta_c_delta_c"
            ]
        ),

        "P_delta_c_divY": (
            W
            * np.asarray(
                raw_response[
                    "P_delta_c_divY"
                ]
            )
        ),

        "P_delta_c_divX": (
            W
            * np.asarray(
                raw_response[
                    "P_delta_c_divX"
                ]
            )
        ),

        "P_divY_divY": (
            W2
            * np.asarray(
                raw_response[
                    "P_divY_divY"
                ]
            )
        ),

        "P_divX_divX": (
            W2
            * np.asarray(
                raw_response[
                    "P_divX_divX"
                ]
            )
        ),

        "P_divY_divX": (
            W2
            * np.asarray(
                raw_response[
                    "P_divY_divX"
                ]
            )
        ),

        "Nmodes": np.asarray(
            raw_response[
                "Nmodes"
            ]
        ),

        "postprocessing_estimator": (
            raw_response[
                "postprocessing_estimator"
            ]
        ),

        "normalization_note": (
            raw_response[
                "normalization_note"
            ]
        ),

        "mas_note": (
            raw_response[
                "mas_note"
            ]
        ),

        "constructed_smoothing": {
            "R_Mpc_over_h": R,
            "window": (
                "Gaussian W_R(k)=exp[-(kR)^2/2]"
            ),
            "implementation": (
                "spectrum_level_bin_center"
            ),
            "binning_note": (
                "W_R is evaluated at the Pylians shell-bin k. "
                "This approximates mode-by-mode Gaussian smoothing "
                "when the window varies negligibly across each bin."
            ),
        },
    }


def gaussian_smooth_scalar(
    field,
    boxsize,
    R,
    workers=1,
):
    """
    Gaussian smooth an already constructed scalar field:

        field_R = W_R * field

    with

        W_R(k) = exp[-(k R)^2 / 2].
    """

    field = np.asarray(
        field,
        dtype=np.float32,
    )

    nx, ny, nz = field.shape

    dx = float(boxsize) / nx
    dy = float(boxsize) / ny
    dz = float(boxsize) / nz

    kx = (
        2.0 * np.pi
        * sfft.fftfreq(nx, d=dx)
    ).astype(np.float32)

    ky = (
        2.0 * np.pi
        * sfft.fftfreq(ny, d=dy)
    ).astype(np.float32)

    kz = (
        2.0 * np.pi
        * sfft.rfftfreq(nz, d=dz)
    ).astype(np.float32)

    field_k = sfft.rfftn(
        field,
        workers=workers,
    )

    R = np.float32(R)

    k2 = (
        kx[:, None, None]**2
        + ky[None, :, None]**2
        + kz[None, None, :]**2
    )

    W_R = np.exp(
        -0.5 * R**2 * k2
    ).astype(
        np.float32,
        copy=False,
    )

    del k2

    field_k *= W_R

    del W_R

    field_R = sfft.irfftn(
        field_k,
        s=field.shape,
        workers=workers,
    ).astype(
        np.float32,
        copy=False,
    )

    del field_k

    return field_R


# ===========================================================================
# Convergence diagnostics
# ===========================================================================

def compute_convergence_tests(
    h,
    c,
    nu,
    boxsize,
):
    """
    Quantities intended explicitly for Ngrid convergence tests.

    Direct CIC currents:

        J_h  = (1+delta_h)V_h
        J_c  = (1+delta_c)V_c
        J_nu = (1+delta_nu)V_nu.

    Mixed nonlinear fields:

        H_c  = (1+delta_h)V_c
        H_nu = (1+delta_h)V_nu.

    The direct currents inherit primitive CIC conservation.
    H_c and H_nu are products of separately reconstructed grid fields and
    need not have the same Ngrid behavior.

    For the primitive current spectra we save two versions:

        MAS_None:
            spectra measured directly from the gridded CIC fields,
            without applying an additional mass-assignment correction.

        MAS_CIC:
            spectra obtained with MAS="CIC" passed to the Pylians
            velocity-spectrum routine.

    Saving both allows the CIC correction itself to be diagnosed later,
    rather than building it implicitly into the convergence test.
    """

    # ===================================================================
    # Primitive normalized currents
    # ===================================================================

    Jh = get_current_vector(
        h
    )

    Jc = get_current_vector(
        c
    )

    Jnu = get_current_vector(
        nu
    )

    # ===================================================================
    # Output structure
    # ===================================================================

    out = {

        "primitive_zero_modes": {
            "halo": h[
                "primitive_sums"
            ],
            "cdm": c[
                "primitive_sums"
            ],
            "nu": nu[
                "primitive_sums"
            ],
        },

"smoothed_zero_modes": {
    "halo": {},
    "cdm": {},
    "nu": {},
},

        "halo_occupancy": h[
            "summary"
        ],

        # ---------------------------------------------------------------
        # Save both the raw gridded current spectra and the spectra
        # evaluated with the CIC MAS option.
        # ---------------------------------------------------------------

        "current_spectra": {
            "MAS_None": {},
            "MAS_CIC": {},
        },

        "halo_weighted_velocity_spectra": {
            "raw": {},
            "cdm_smoothed": {},
            "nu_smoothed": {},
        },

        "definitions": {

            "J_h": (
                "(1+delta_h)V_h=P_h/<n_h>"
            ),

            "J_c": (
                "(1+delta_c)V_c=P_c/<n_c>"
            ),

            "J_nu": (
                "(1+delta_nu)V_nu=P_nu/<n_nu>"
            ),

            "H_c": (
                "(1+delta_h)V_c"
            ),

            "H_nu": (
                "(1+delta_h)V_nu"
            ),

            "current_spectra_MAS_None": (
                "Primitive current spectra measured directly from the "
                "gridded CIC current fields with no additional MAS "
                "correction."
            ),

            "current_spectra_MAS_CIC": (
                "The same primitive current spectra evaluated with "
                'MAS="CIC" in the Pylians velocity-spectrum routine. '
                "Comparing these with MAS_None isolates the effect of "
                "the CIC mass-assignment treatment."
            ),

            "interpretation": (
                "J_a is directly proportional to a deposited primitive "
                "first moment and therefore has a well-defined CIC zero "
                "mode. H_c and H_nu are nonlinear products of separately "
                "constructed fields and can exhibit stronger Ngrid "
                "dependence. No universal CIC correction is applied to "
                "the mixed nonlinear H fields."
            ),
        },
    }

    # ===================================================================
    # Zero-mode conservation of smoothed primitive fields
    # ===================================================================

    for species_name, species in (
        ("halo", h),
        ("cdm", c),
        ("nu", nu),
    ):
        raw_sums = species[
            "primitive_sums"
        ]

        density_sum_raw = float(
            raw_sums[
                "density"
            ]
        )

        Px_sum_raw = float(
            raw_sums[
                "P_x"
            ]
        )

        Py_sum_raw = float(
            raw_sums[
                "P_y"
            ]
        )

        Pz_sum_raw = float(
            raw_sums[
                "P_z"
            ]
        )

        for R_key in sorted_R_keys(
            species[
                "smoothed"
            ].keys()
        ):

            smooth = species[
                "smoothed"
            ][
                R_key
            ]

            density_sum_R = float(
                smooth[
                    "summary"
                ][
                    "density_sum"
                ]
            )

            Px_sum_R = float(
                smooth[
                    "P_sum_x"
                ]
            )

            Py_sum_R = float(
                smooth[
                    "P_sum_y"
                ]
            )

            Pz_sum_R = float(
                smooth[
                    "P_sum_z"
                ]
            )

            density_difference = (
                density_sum_R
                -
                density_sum_raw
            )

            Px_difference = (
                Px_sum_R
                -
                Px_sum_raw
            )

            Py_difference = (
                Py_sum_R
                -
                Py_sum_raw
            )

            Pz_difference = (
                Pz_sum_R
                -
                Pz_sum_raw
            )

            # -----------------------------------------------------------
            # For density a relative difference is meaningful because
            # the zero mode is large and strictly positive.
            #
            # We deliberately do NOT form relative differences for the
            # momentum components, because their global sums can be close
            # to zero. Dividing by such a number would produce a large
            # and misleading relative error.
            # -----------------------------------------------------------

            if density_sum_raw != 0.0:

                density_relative_difference = (
                    density_difference
                    /
                    density_sum_raw
                )

            else:

                density_relative_difference = np.nan

            out[
                "smoothed_zero_modes"
            ][
                species_name
            ][
                R_key
            ] = {

                # -------------------------------------------------------
                # Smoothed zero modes
                # -------------------------------------------------------

                "density_sum": density_sum_R,

                "P_x_sum": Px_sum_R,

                "P_y_sum": Py_sum_R,

                "P_z_sum": Pz_sum_R,

                # -------------------------------------------------------
                # Raw zero modes for direct comparison
                # -------------------------------------------------------

                "density_sum_raw": density_sum_raw,

                "P_x_sum_raw": Px_sum_raw,

                "P_y_sum_raw": Py_sum_raw,

                "P_z_sum_raw": Pz_sum_raw,

                # -------------------------------------------------------
                # Conservation diagnostics
                # -------------------------------------------------------

                "density_difference": (
                    density_difference
                ),

                "density_relative_difference": (
                    density_relative_difference
                ),

                "P_x_difference": (
                    Px_difference
                ),

                "P_y_difference": (
                    Py_difference
                ),

                "P_z_difference": (
                    Pz_difference
                ),
            }

    # ===================================================================
    # All primitive current auto/cross spectra
    #
    # Save BOTH:
    #
    #   1. MAS=None
    #      literal spectra of the gridded CIC current fields;
    #
    #   2. MAS=CIC
    #      spectra with the CIC option supplied to Pylians.
    #
    # This comparison lets us determine later how important the CIC
    # treatment is as a function of k and Ngrid.
    # ===================================================================

    current_vectors = {
        "h": Jh,
        "c": Jc,
        "nu": Jnu,
    }

    current_names = (
        "h",
        "c",
        "nu",
    )

    for i, a in enumerate(
        current_names
    ):

        for b in current_names[
            i:
        ]:

            spectrum_name = (
                f"{a}_x_{b}"
            )

            # -----------------------------------------------------------
            # Raw gridded spectrum: no additional MAS correction
            # -----------------------------------------------------------

            out[
                "current_spectra"
            ][
                "MAS_None"
            ][
                spectrum_name
            ] = vector_cross_spectrum(
                current_vectors[a],
                current_vectors[b],
                boxsize,
                MAS="None",
            )

            # -----------------------------------------------------------
            # Same spectrum with CIC treatment in Pylians
            # -----------------------------------------------------------

            out[
                "current_spectra"
            ][
                "MAS_CIC"
            ][
                spectrum_name
            ] = vector_cross_spectrum(
                current_vectors[a],
                current_vectors[b],
                boxsize,
                MAS="CIC",
            )

    # ===================================================================
    # Raw Hc and Hnu
    # ===================================================================
    #
    # These are nonlinear mixed fields:
    #
    #     Hc  = (1+delta_h) V_c
    #     Hnu = (1+delta_h) V_nu
    #
    # We therefore keep MAS="None".  A single CIC transfer function cannot
    # in general be factored out of these products.
    # ===================================================================

    Hc = build_halo_weighted_velocity(
        h,
        c,
    )

    Hnu = build_halo_weighted_velocity(
        h,
        nu,
    )

    out[
        "halo_weighted_velocity_spectra"
    ][
        "raw"
    ] = {

        "Jh_x_Hc": vector_cross_spectrum(
            Jh,
            Hc,
            boxsize,
            MAS="None",
        ),

        "Jh_x_Hnu": vector_cross_spectrum(
            Jh,
            Hnu,
            boxsize,
            MAS="None",
        ),

        "Hc_x_Hc": vector_cross_spectrum(
            Hc,
            Hc,
            boxsize,
            MAS="None",
        ),

        "Hnu_x_Hnu": vector_cross_spectrum(
            Hnu,
            Hnu,
            boxsize,
            MAS="None",
        ),

        "Hc_x_Hnu": vector_cross_spectrum(
            Hc,
            Hnu,
            boxsize,
            MAS="None",
        ),
    }

    del Hc
    del Hnu

    gc.collect()

    # ===================================================================
    # Smoothed CDM Hc^R
    # ===================================================================

    for Rc in sorted_R_keys(
        c[
            "smoothed"
        ].keys()
    ):

        Hc_R = build_halo_weighted_velocity(
            h,
            c,
            R_key=Rc,
        )

        out[
            "halo_weighted_velocity_spectra"
        ][
            "cdm_smoothed"
        ][
            Rc
        ] = {

            "Jh_x_HcR": vector_cross_spectrum(
                Jh,
                Hc_R,
                boxsize,
                MAS="None",
            ),

            "HcR_x_HcR": vector_cross_spectrum(
                Hc_R,
                Hc_R,
                boxsize,
                MAS="None",
            ),
        }

        del Hc_R

        gc.collect()

    # ===================================================================
    # Smoothed neutrino Hnu^R
    # ===================================================================

    for Rnu in sorted_R_keys(
        nu[
            "smoothed"
        ].keys()
    ):

        Hnu_R = build_halo_weighted_velocity(
            h,
            nu,
            R_key=Rnu,
        )

        out[
            "halo_weighted_velocity_spectra"
        ][
            "nu_smoothed"
        ][
            Rnu
        ] = {

            "Jh_x_HnuR": vector_cross_spectrum(
                Jh,
                Hnu_R,
                boxsize,
                MAS="None",
            ),

            "HnuR_x_HnuR": vector_cross_spectrum(
                Hnu_R,
                Hnu_R,
                boxsize,
                MAS="None",
            ),
        }

        del Hnu_R

        gc.collect()

    gc.collect()

    return out

# ===========================================================================
# Optional existing species diagnostics
# ===========================================================================

def compute_species_velocity_diagnostics(
    h,
    c,
    nu,
    boxsize,
):
    """
    Raw density, current and velocity auto/cross spectra.
    """

    species = {
        "h": h,
        "c": c,
        "nu": nu,
    }

    out = {
        "available_species": tuple(
            species.keys()
        ),

        "density_spectra": {},

        "current_spectra": {},

        "velocity_spectra": {},

        "notes": {
            "current": (
                "J_a=(1+delta_a)V_a."
            ),

            "halo_velocity": (
                "V_h=P_h/n_h is set to zero where n_h=0."
            ),
        },
    }

    names = tuple(
        species.keys()
    )

    for i, a in enumerate(
        names
    ):

        for b in names[
            i:
        ]:

            A = species[a]
            B = species[b]

            pair_key = (
                f"{a}_x_{b}"
            )

            # -----------------------------------------------------------
            # Density
            # -----------------------------------------------------------

            density_result = (
                scalar_cross_spectrum(
                    A["delta"],
                    B["delta"],
                    boxsize,
                    MAS="CIC",
                )
            )

            out[
                "density_spectra"
            ][
                pair_key
            ] = spectrum_to_named_dict(
                density_result,
                f"delta_{a}",
                f"delta_{b}",
            )

            # -----------------------------------------------------------
            # Current
            # -----------------------------------------------------------

            out[
                "current_spectra"
            ][
                pair_key
            ] = vector_cross_spectrum(
                get_current_vector(
                    A
                ),
                get_current_vector(
                    B
                ),
                boxsize,
                MAS="CIC",
            )

            # -----------------------------------------------------------
            # Velocity
            # -----------------------------------------------------------

            out[
                "velocity_spectra"
            ][
                pair_key
            ] = vector_cross_spectrum(
                get_velocity_vector(
                    A
                ),
                get_velocity_vector(
                    B
                ),
                boxsize,
                MAS="None",
            )

    return out


# ===========================================================================
# Response spectra
# ===========================================================================

def compute_one_sided_response_spectra(
    delta_c,
    div_Y,
    div_X,
    boxsize,
    physical_neutrino_driver_available,
    lcdm_zero_neutrino_convention,
):

    delta_x_div_y = (
        scalar_cross_spectrum(
            delta_c,
            div_Y,
            boxsize,
            MAS="None",
        )
    )

    delta_x_div_x = (
        scalar_cross_spectrum(
            delta_c,
            div_X,
            boxsize,
            MAS="None",
        )
    )

    div_y_x_div_x = (
        scalar_cross_spectrum(
            div_Y,
            div_X,
            boxsize,
            MAS="None",
        )
    )

    assert_same_binning(
        delta_x_div_y,
        delta_x_div_x,
        div_y_x_div_x,
        labels=(
            "delta-divY",
            "delta-divX",
            "divY-divX",
        ),
    )

    return {

        "beta_estimator_available": True,

        "physical_neutrino_driver_available": bool(
            physical_neutrino_driver_available
        ),

        "lcdm_zero_neutrino_convention": bool(
            lcdm_zero_neutrino_convention
        ),

        "k_h_per_Mpc": (
            delta_x_div_y[0]
        ),

        "P_delta_c_delta_c": (
            delta_x_div_y[1]
        ),

        "P_delta_c_divY": (
            delta_x_div_y[3]
        ),

        "P_delta_c_divX": (
            delta_x_div_x[3]
        ),

        "P_divY_divY": (
            delta_x_div_y[2]
        ),

        "P_divX_divX": (
            delta_x_div_x[2]
        ),

        "P_divY_divX": (
            div_y_x_div_x[3]
        ),

        "Nmodes": (
            delta_x_div_y[4]
        ),

        "postprocessing_estimator": (
            "beta_h(k)="
            "P_delta_c_divY(k)/"
            "P_delta_c_divX(k)"
        ),

        "normalization_note": (
            "div(Y) and div(X) are saved without -1/(aH); "
            "the common factor cancels in beta_h."
        ),

        "mas_note": (
            "No additional CIC deconvolution is applied to X/Y "
            "because they are nonlinear products of separately "
            "gridded fields."
        ),
    }


# ===========================================================================
# Density diagnostics
# ===========================================================================

def compute_density_diagnostics(
    h,
    c,
    nu,
    boxsize,
):

    h_x_c = scalar_cross_spectrum(
        h["delta"],
        c["delta"],
        boxsize,
        MAS="CIC",
    )

    h_x_nu = scalar_cross_spectrum(
        h["delta"],
        nu["delta"],
        boxsize,
        MAS="CIC",
    )

    c_x_nu = scalar_cross_spectrum(
        c["delta"],
        nu["delta"],
        boxsize,
        MAS="CIC",
    )

    assert_same_binning(
        h_x_c,
        h_x_nu,
        c_x_nu,
        labels=(
            "h-c",
            "h-nu",
            "c-nu",
        ),
    )

    return {
        "k_h_per_Mpc": h_x_c[0],

        "P_hh": h_x_c[1],
        "P_cc": h_x_c[2],
        "P_hc": h_x_c[3],

        "P_nunu": h_x_nu[2],
        "P_hnu": h_x_nu[3],
        "P_cnu": c_x_nu[3],

        "Nmodes": h_x_c[4],

        "mas_note": (
            "Density fields are direct CIC assignments; "
            "standard CIC correction is requested from Pylians."
        ),
    }


# ===========================================================================
# FFT divergence
# ===========================================================================

def fft_divergence(
    vec,
    boxsize,
    workers=1,
):
    """
    Compute div(F)=i k.F with scipy.fft.

    scipy.fft preserves float32/complex64, unlike numpy.fft which normally
    promotes to double precision and substantially increases memory use.

    Nyquist derivative modes are set to zero for even grids.
    """

    fx, fy, fz = (
        np.asarray(
            component,
            dtype=np.float32,
        )
        for component in vec
    )

    if (
        fx.shape != fy.shape
        or fx.shape != fz.shape
    ):
        raise ValueError(
            "Vector components must have identical shapes."
        )

    if fx.ndim != 3:
        raise ValueError(
            f"Expected 3D fields; got shape {fx.shape}."
        )

    nx, ny, nz = fx.shape

    dx = float(boxsize) / nx
    dy = float(boxsize) / ny
    dz = float(boxsize) / nz

    kx = (
        2.0
        * np.pi
        * sfft.fftfreq(
            nx,
            d=dx,
        )
    ).astype(
        np.float32
    )

    ky = (
        2.0
        * np.pi
        * sfft.fftfreq(
            ny,
            d=dy,
        )
    ).astype(
        np.float32
    )

    kz = (
        2.0
        * np.pi
        * sfft.rfftfreq(
            nz,
            d=dz,
        )
    ).astype(
        np.float32
    )

    if nx % 2 == 0:
        kx[
            nx // 2
        ] = 0.0

    if ny % 2 == 0:
        ky[
            ny // 2
        ] = 0.0

    if nz % 2 == 0:
        kz[-1] = 0.0

    # ------------------------------------------------------------------
    # x
    # ------------------------------------------------------------------

    fx_k = sfft.rfftn(
        fx,
        workers=workers,
    )

    div_k = (
        np.complex64(1j)
        * kx[:, None, None]
        * fx_k
    )

    del fx_k

    # ------------------------------------------------------------------
    # y
    # ------------------------------------------------------------------

    fy_k = sfft.rfftn(
        fy,
        workers=workers,
    )

    div_k += (
        np.complex64(1j)
        * ky[None, :, None]
        * fy_k
    )

    del fy_k

    # ------------------------------------------------------------------
    # z
    # ------------------------------------------------------------------

    fz_k = sfft.rfftn(
        fz,
        workers=workers,
    )

    div_k += (
        np.complex64(1j)
        * kz[None, None, :]
        * fz_k
    )

    del fz_k

    div = sfft.irfftn(
        div_k,
        s=fx.shape,
        workers=workers,
    ).astype(
        np.float32,
        copy=False,
    )

    del div_k

    return div



# ===========================================================================
# Scalar spectra
# ===========================================================================

def scalar_cross_spectrum(
    field1,
    field2,
    boxsize,
    MAS="None",
    axis=0,
    threads=1,
):

    field1 = np.asarray(
        field1,
        dtype=np.float32,
    )

    field2 = np.asarray(
        field2,
        dtype=np.float32,
    )

    if (
        field1.shape
        != field2.shape
    ):
        raise ValueError(
            "Scalar fields have inconsistent shapes: "
            f"{field1.shape} and {field2.shape}."
        )

    pk = PKL.XPk(
        [
            field1.copy(),
            field2.copy(),
        ],
        float(boxsize),
        axis,
        [
            MAS,
            MAS,
        ],
        threads,
    )

    return [
        np.asarray(
            pk.k3D
        ),

        np.asarray(
            pk.Pk[
                :,
                0,
                0,
            ]
        ),

        np.asarray(
            pk.Pk[
                :,
                0,
                1,
            ]
        ),

        np.asarray(
            pk.XPk[
                :,
                0,
                0,
            ]
        ),

        np.asarray(
            pk.Nmodes3D
        ),
    ]


# ===========================================================================
# Vector spectra
# ===========================================================================

def vector_cross_spectrum(
    vec_a,
    vec_b,
    boxsize,
    MAS="None",
    threads=1,
):
    """
    Full vector auto/cross spectrum using the existing user-provided
    PKL.XPk_velvel routine.
    """

    result = PKL.XPk_velvel(
        vec_a[0].copy(),
        vec_a[1].copy(),
        vec_a[2].copy(),

        vec_b[0].copy(),
        vec_b[1].copy(),
        vec_b[2].copy(),

        float(boxsize),

        0,
        MAS,
        threads,
    )

    return {
        "k_h_per_Mpc": np.asarray(
            result[0]
        ),

        "P_AA": np.asarray(
            result[1]
        ),

        "P_BB": np.asarray(
            result[2]
        ),

        "P_AB": np.asarray(
            result[3]
        ),

        "Nmodes": np.asarray(
            result[4]
        ),
    }


def spectrum_to_named_dict(
    spectrum,
    name_a,
    name_b,
):

    return {
        "k_h_per_Mpc": spectrum[0],
        f"P_{name_a}_{name_a}": spectrum[1],
        f"P_{name_b}_{name_b}": spectrum[2],
        f"P_{name_a}_{name_b}": spectrum[3],
        "Nmodes": spectrum[4],
    }


# ===========================================================================
# Binning checks
# ===========================================================================

def assert_same_binning(
    *spectra,
    labels=None,
):

    if len(
        spectra
    ) < 2:
        return

    if labels is None:

        labels = tuple(
            f"spectrum-{index}"
            for index in range(
                len(
                    spectra
                )
            )
        )

    reference_k = np.asarray(
        spectra[0][0]
    )

    reference_modes = np.asarray(
        spectra[0][4]
    )

    for spectrum, label in zip(
        spectra[1:],
        labels[1:],
    ):

        if not np.allclose(
            reference_k,
            spectrum[0],
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                f"Inconsistent k bins in {label}."
            )

        if not np.array_equal(
            reference_modes,
            spectrum[4],
        ):
            raise RuntimeError(
                f"Inconsistent Nmodes in {label}."
            )


def assert_matching_k_and_modes(
    reference_k,
    reference_modes,
    test_k,
    test_modes,
    label,
):

    if not np.allclose(
        np.asarray(
            reference_k
        ),
        np.asarray(
            test_k
        ),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            f"Inconsistent k bins for {label}."
        )

    if not np.array_equal(
        np.asarray(
            reference_modes
        ),
        np.asarray(
            test_modes
        ),
    ):
        raise RuntimeError(
            f"Inconsistent Nmodes for {label}."
        )


# ===========================================================================
# Density helpers
# ===========================================================================

def validate_mean_density(
    density,
    field_label,
):

    mean_density = float(
        np.mean(
            density,
            dtype=np.float64,
        )
    )

    if (
        not np.isfinite(
            mean_density
        )
        or mean_density <= 0.0
    ):
        raise ValueError(
            f"{field_label} has invalid mean density: "
            f"{mean_density}"
        )

    return mean_density


def summarize_density(
    density,
    mean_density,
    field_label,
):

    return {
        "label": field_label,

        "mean_density": float(
            mean_density
        ),

        "density_sum": float(
            np.sum(
                density,
                dtype=np.float64,
            )
        ),

        "density_min": float(
            np.min(
                density
            )
        ),

        "density_max": float(
            np.max(
                density
            )
        ),

        "empty_cell_fraction": float(
            np.mean(
                density <= 0.0
            )
        ),
    }


# ===========================================================================
# Metadata
# ===========================================================================

def build_metadata(
    spec,
    sim,
    ngrid,
    string,
    mass_cut,
    mass_width,
    boxsize,
    is_lcdm,
    save_density_diagnostics,
    save_species_diagnostics,
    compute_smoothed_responses,
    compute_convergence_diagnostics,
):

    return {
        "analysis_type": (
            "extended_one_sided_cdm_halo_response_and_convergence"
        ),

        "spec": spec,

        "sim_type": sim,

        "ngrid": int(
            ngrid
        ),

        "grid_spacing_Mpc_over_h": (
            float(
                boxsize
            )
            / int(
                ngrid
            )
        ),

        "boxsize_Mpc_over_h": float(
            boxsize
        ),

        "mass_selection": string,

        "mass_cut": float(
            mass_cut
        ),

        "mass_width": float(
            mass_width
        ),

        "lcdm_mode": bool(
            is_lcdm
        ),

        "density_diagnostics_saved": bool(
            save_density_diagnostics
            or save_species_diagnostics
        ),

        "species_diagnostics_saved": bool(
            save_species_diagnostics
        ),

        "smoothed_response_variants_saved": bool(
            compute_smoothed_responses
        ),

        "convergence_diagnostics_saved": bool(
            compute_convergence_diagnostics
        ),

        "primitive_fields": {
            "density": (
                "n_a=sum_p W_CIC"
            ),

            "P_i": (
                "P_a,i=sum_p W_CIC v_i"
            ),

            "V_i": (
                "V_a,i=P_a,i/n_a"
            ),

            "J_i": (
                "J_a,i=P_a,i/<n_a>="
                "(1+delta_a)V_a,i"
            ),

            "V_i_R": (
                "V_a,i^(R)=P_a,i^(R)/n_a^(R)"
            ),
        },

        "response_fields": {
            "Y": (
                "J_h-(1+delta_h)V_c"
            ),

            "X": (
                "(1+delta_h)(V_nu-V_c)"
            ),
        },

        "postprocessing_estimator": (
            "beta_h(k)="
            "P_delta_c_divY(k)/"
            "P_delta_c_divX(k)"
        ),
    }


# ===========================================================================
# Validation
# ===========================================================================

def validate_inputs(
    string,
    ngrid_step,
):

    if string not in {
        "+",
        "-",
        "bin",
        "all",
    }:

        raise ValueError(
            "string must be one of "
            "'+', '-', 'bin', or 'all'; "
            f"got {string!r}"
        )

    if ngrid_step <= 0:

        raise ValueError(
            "ngrid_step must be positive; "
            f"got {ngrid_step}"
        )


# ===========================================================================
# Naming
# ===========================================================================

def simulation_def(
    spec,
    sim_type,
    string,
    mass_cut,
    mass_width,
):

    if string == "+":

        halo_tag = (
            f"halo_mass_{mass_cut:.1e}"
        )

    elif string == "bin":

        halo_tag = (
            f"halo_mass_{mass_cut:.1e}"
            f"_mass_bin_{mass_width}"
        )

    elif string == "-":

        halo_tag = (
            f"halo_mass_le_{mass_cut:.1e}"
        )

    elif string == "all":

        halo_tag = (
            "halo_all"
        )

    else:

        raise ValueError(
            f"Unknown halo selection "
            f"string={string!r}"
        )

    return (
        f"{sim_type}_{spec}_"
        f"extended_{halo_tag}_cdm_nu"
    )


def halo_directory_name(
    string,
    mass_cut,
    mass_width,
):

    if string in {
        "+",
        "bin",
    }:

        return (
            f"halo_mass_{mass_cut:.0e}"
            .replace(
                "+",
                "",
            )
        )

    if string == "-":

        return (
            f"halo_mass_le_{mass_cut:.0e}"
            .replace(
                "+",
                "",
            )
        )

    if string == "all":
        return "halo_all"

    raise ValueError(
        f"Unknown halo selection "
        f"string={string!r}"
    )


def halo_pickle_basename(
    ngrid,
    sim,
    spec,
    string,
    mass_cut,
    mass_width,
):

    if string == "+":

        sim_name = (
            f"{sim}_{spec}_"
            f"halo_mass_{mass_cut:.1e}"
        )

    elif string == "bin":

        sim_name = (
            f"{sim}_{spec}_"
            f"halo_mass_{mass_cut:.1e}_"
            f"mass_bin_{mass_width}"
        )

    elif string == "-":

        sim_name = (
            f"{sim}_{spec}_"
            f"halo_mass_le_{mass_cut:.1e}"
        )

    elif string == "all":

        sim_name = (
            f"{sim}_{spec}_halo_all"
        )

    else:

        raise ValueError(
            f"Unknown halo selection "
            f"string={string!r}"
        )

    return (
        f"data_ngrid_{ngrid}_"
        f"sim_{sim_name}.pickle"
    )


# ===========================================================================
# Input-file discovery
# ===========================================================================

def load_files_pickles(
    file_path,
    bulk_species,
    spec,
    ngrid,
    sim,
    string,
    mass_cut,
    mass_width,
):
    """
    Load one halo/CDM/neutrino bulk-field pickle.

    For CDM/neutrinos, prefer the new smoothing-enabled file when present.
    """

    # ===================================================================
    # Halo
    # ===================================================================

    if bulk_species == "halo":

        halo_dir = halo_directory_name(
            string=string,
            mass_cut=mass_cut,
            mass_width=mass_width,
        )
        
        basename = halo_pickle_basename(
            ngrid=ngrid,
            sim=sim,
            spec=spec,
            string=string,
            mass_cut=mass_cut,
            mass_width=mass_width,
        )
        directory = os.path.join(
            file_path,
            spec,
            sim,
            halo_dir,
            "output",
        )
        
        exact_path = os.path.join(
            directory,
            basename,
        )
        
        base_no_ext = os.path.splitext(
            basename
        )[0]
        
        smoothed_candidates = sorted(
            glob.glob(
                os.path.join(
                    directory,
                    base_no_ext
                    + "_smooth_R*.pickle",
                )
            )
        )
        
        # Prefer the smoothing-enabled halo file because it contains
        # both the raw and smoothed primitive fields.
        if len(smoothed_candidates) == 1:
        
            path = smoothed_candidates[0]
        
        elif len(smoothed_candidates) > 1:
        
            raise RuntimeError(
                "Multiple smoothing-enabled halo files found:\n"
                + "\n".join(
                    smoothed_candidates
                )
            )
        
        elif os.path.exists(
            exact_path
        ):
        
            path = exact_path
        
        else:
        
            raise FileNotFoundError(
                "Missing halo input pickle. Tried:\n"
                f"{exact_path}\n"
                "and smoothing-enabled variants."
            )
        
        print(
            f"Loading halo:\n{path}",
            flush=True,
        )
        
        return load(
            path
        )

    # ===================================================================
    # CDM / neutrino
    # ===================================================================

    directory = os.path.join(
        file_path,
        spec,
        sim,
        bulk_species,
        "output",
    )

    base = (
        f"data_ngrid_{ngrid}_"
        f"sim_{sim}_{spec}_{bulk_species}"
    )

    exact_path = os.path.join(
        directory,
        base + ".pickle",
    )

    smoothed_candidates = sorted(
        glob.glob(
            os.path.join(
                directory,
                base
                + "_smooth_R*.pickle",
            )
        )
    )

    # Prefer the smoothing file because it contains both raw and
    # smoothed primitive fields.
    if len(
        smoothed_candidates
    ) == 1:

        path = smoothed_candidates[0]

    elif len(
        smoothed_candidates
    ) > 1:

        raise RuntimeError(
            "Multiple smoothing-enabled input files found. "
            "Choose/remove the unwanted smoothing set:\n"
            + "\n".join(
                smoothed_candidates
            )
        )

    elif os.path.exists(
        exact_path
    ):

        path = exact_path

    else:

        raise FileNotFoundError(
            "Missing matter input pickle. Tried:\n"
            f"{exact_path}\n"
            f"and pattern:\n"
            f"{base}_smooth_R*.pickle"
        )

    print(
        f"Loading {bulk_species}:\n{path}",
        flush=True,
    )

    return load(
        path
    )


# ===========================================================================
# Saving
# ===========================================================================

def save_and_print_usage(
    start_time,
    start_mem,
    simulation,
    ngrid,
    power_data,
    save_path,
):

    save_dataframe(
        power_data,
        simulation,
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
        f"ngrid={ngrid} finished",
    )


def save_dataframe(
    power_data,
    simulation,
    ngrid,
    save_path,
):

    os.makedirs(
        save_path,
        exist_ok=True,
    )

    output_file = os.path.join(
        save_path,
        (
            f"Extended_response_spectra_"
            f"ngrid_{ngrid}_"
            f"sim_{simulation}.pickle"
        ),
    )

    print(
        f"Saving:\n{output_file}",
        flush=True,
    )

    with open(
        output_file,
        "wb",
    ) as handle:

        pickle.dump(
            power_data,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# ===========================================================================
# Resource diagnostics
# ===========================================================================

def print_usage(
    start_time,
    start_mem,
    message="",
):

    elapsed_time = (
        time.time()
        - start_time
    )

    current_mem = (
        psutil.Process()
        .memory_info()
        .rss
    )

    elapsed_mem = (
        current_mem
        - start_mem
    )

    print(
        f"{message} - "
        f"Time: {elapsed_time:.2f} s, "
        f"Memory change: "
        f"{elapsed_mem / 1024**2:.2f} MB, "
        f"Current RSS: "
        f"{current_mem / 1024**3:.2f} GB",
        flush=True,
    )