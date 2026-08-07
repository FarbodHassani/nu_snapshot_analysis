"""
Compute spectra for the one-sided CDM--halo response estimator.

Primitive CIC fields stored in the bulk-velocity files are

    n_a(x)   = sum_p W_CIC(x-x_p),
    P_a,i(x) = sum_p W_CIC(x-x_p) v_p,i.

The reconstructed fields are

    1 + delta_a = n_a / <n_a>,
    V_a         = P_a / n_a,
    J_a         = P_a / <n_a> = (1+delta_a)V_a.

The one-sided response fields are

    Y = (1+delta_h)(V_h-V_c)
      = J_h-(1+delta_h)V_c,

    X = (1+delta_h)(V_nu-V_c).

The response coefficient is formed later in post-processing as

    beta_h(k) = P_{delta_c,divY}(k) / P_{delta_c,divX}(k).

The code uses a spectral FFT derivative for divY and divX, and Pylians for
the scalar auto- and cross-spectra. Optional Pylians diagnostics save all
density, current, and bulk-velocity spectra for the available species.
"""

import os
import pickle
import sys
import time

import numpy as np
import psutil
from mpi4py import MPI

sys.path.append(
    "/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/"
)
sys.path.append(
    "/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/"
    "nu_code/library_snapshot"
)

import Pk_library as PKL



_COMPONENTS = ("x", "y", "z")

def load(filename):
    """Load a pickle file."""
    with open(filename, "rb") as handle:
        return pickle.load(handle)

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
):
    """
    Compute one-sided CDM--halo response spectra over a range of grids.

    Parameters
    ----------
    save_density_diagnostics : bool
        Save compact density auto- and cross-spectra when the more complete
        species diagnostics are not requested.
    save_species_diagnostics : bool
        Save Pylians density, current, and bulk-velocity auto- and
        cross-spectra for every available species pair. This option implies
        density diagnostics and requires constructing the sparse halo bulk
        velocity for diagnostic use.

    Notes
    -----
    ``ngrid_max`` follows the Python ``range`` convention and is exclusive.

    For ``sim_type == "0.0ev"``, the code adopts the convention

    delta_nu = 0,
    V_nu     = 0,
    J_nu     = 0,

    so that
    
        X_LCDM = -(1+delta_h)V_c.
    
    The same response spectra and beta estimator are then computed for LCDM
    and for the massive-neutrino simulations.
    """

    boxsize = float(boxsize)
    validate_inputs(string, ngrid_step)

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
    )

    comm.Barrier()
    if rank == 0:
        print_usage(
            start_time_all,
            start_mem_all,
            "- Total time and memory",
        )
        print(
            "\n*********** All one-sided response spectra finished! ***********\n",
            flush=True,
        )


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
):
    """Split the requested grid sizes across MPI ranks."""

    ngrid_list = list(range(ngrid_min, ngrid_max, ngrid_step))
    ngrid_per_rank = np.array_split(ngrid_list, size)

    for ngrid_value in ngrid_per_rank[rank]:
        ngrid = int(ngrid_value)
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss

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

def compute_response_decomposition_spectra(
    h,
    c,
    nu,
    delta_c,
    full_response,
    boxsize,
):
    """
    Compute the two contributions to the driver and response cross-spectra.

    The exact field decompositions are

        X = X_unweighted + X_weighted,

        X_unweighted = V_nu - V_c,
        X_weighted   = delta_h (V_nu - V_c),

    and

        Y = Y_unweighted + Y_weighted,

        Y_unweighted = V_h - V_c,
        Y_weighted   = delta_h (V_h - V_c).

    Because both the divergence and cross-spectrum with delta_c are linear,

        P_delta_c,divX_weighted
            = P_delta_c,divX_full
            - P_delta_c,divX_unweighted,

    and similarly for Y.

    Therefore, only the unweighted vector fields need to be explicitly
    constructed. This avoids constructing the weighted vector fields and
    avoids recomputing the already available full spectra.
    """

    # ===============================================================
    # X unweighted contribution
    # ===============================================================

    X_unweighted = build_X_unweighted_components(
        c=c,
        nu=nu,
    )

    div_X_unweighted = fft_divergence(
        X_unweighted,
        boxsize,
    )

    del X_unweighted

    spec_X_unweighted = scalar_cross_spectrum(
        delta_c,
        div_X_unweighted,
        boxsize,
        MAS="None",
    )

    del div_X_unweighted

    # ===============================================================
    # Y unweighted contribution
    # ===============================================================

    Y_unweighted = build_Y_unweighted_components(
        h=h,
        c=c,
    )

    div_Y_unweighted = fft_divergence(
        Y_unweighted,
        boxsize,
    )

    del Y_unweighted

    spec_Y_unweighted = scalar_cross_spectrum(
        delta_c,
        div_Y_unweighted,
        boxsize,
        MAS="None",
    )

    del div_Y_unweighted

    # ===============================================================
    # Verify that all spectra use exactly the same bins
    # ===============================================================

    assert_same_binning(
        spec_X_unweighted,
        spec_Y_unweighted,
        labels=(
            "delta-divX-unweighted",
            "delta-divY-unweighted",
        ),
    )

    assert_matching_k_and_modes(
        reference_k=full_response["k_h_per_Mpc"],
        reference_modes=full_response["Nmodes"],
        test_k=spec_X_unweighted[0],
        test_modes=spec_X_unweighted[4],
        label="delta-divX-unweighted",
    )

    assert_matching_k_and_modes(
        reference_k=full_response["k_h_per_Mpc"],
        reference_modes=full_response["Nmodes"],
        test_k=spec_Y_unweighted[0],
        test_modes=spec_Y_unweighted[4],
        label="delta-divY-unweighted",
    )

    # ===============================================================
    # Obtain the weighted terms from the exact linear decomposition
    # ===============================================================

    P_delta_c_divX_full = np.asarray(
        full_response["P_delta_c_divX"]
    )

    P_delta_c_divY_full = np.asarray(
        full_response["P_delta_c_divY"]
    )

    P_delta_c_divX_unweighted = np.asarray(
        spec_X_unweighted[3]
    )

    P_delta_c_divY_unweighted = np.asarray(
        spec_Y_unweighted[3]
    )

    P_delta_c_divX_weighted = (
        P_delta_c_divX_full
        - P_delta_c_divX_unweighted
    )

    P_delta_c_divY_weighted = (
        P_delta_c_divY_full
        - P_delta_c_divY_unweighted
    )

    return {
        "k_h_per_Mpc": np.asarray(
            full_response["k_h_per_Mpc"]
        ),
        "Nmodes": np.asarray(
            full_response["Nmodes"]
        ),

        "P_delta_c_divX_full": P_delta_c_divX_full,
        "P_delta_c_divX_unweighted": (
            P_delta_c_divX_unweighted
        ),
        "P_delta_c_divX_weighted": (
            P_delta_c_divX_weighted
        ),

        "P_delta_c_divY_full": P_delta_c_divY_full,
        "P_delta_c_divY_unweighted": (
            P_delta_c_divY_unweighted
        ),
        "P_delta_c_divY_weighted": (
            P_delta_c_divY_weighted
        ),

        "definitions": {
            "X_unweighted": "V_nu-V_c",
            "X_weighted": "delta_h*(V_nu-V_c)",
            "Y_unweighted": "V_h-V_c",
            "Y_weighted": "delta_h*(V_h-V_c)",
        },
    }

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
    save_species_diagnostics=False,
):
    """Load the primitive grids and compute the retained spectra."""

    is_lcdm = sim == "0.0ev"

    # Load and reduce one species at a time so the full input pickles are not
    # retained simultaneously in memory.
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
        data_h,
        build_velocity_diagnostics=save_species_diagnostics,
    )
    del data_h

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
    c = build_matter_fields(data_c, field_label="cdm")
    del data_c

    if is_lcdm:
        # LCDM convention requested here:
        # delta_nu = 0, V_nu = 0, J_nu = 0.
        nu = build_zero_neutrino_fields_like(c)
    
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
        ),
        "field_summary": {
            "halo": h["summary"],
            "cdm": c["summary"],
            "nu": nu["summary"],
        },
    }

    # First construct the physical response from untouched fields.
    Y = build_Y_components(h, c)
    div_Y = fft_divergence(Y, boxsize)
    del Y
    
    X = build_X_components(h, c, nu)
    div_X = fft_divergence(X, boxsize)
    del X
    
    response_spectra = compute_one_sided_response_spectra(
        delta_c=c["delta"],
        div_Y=div_Y,
        div_X=div_X,
        boxsize=boxsize,
        physical_neutrino_driver_available=not is_lcdm,
        lcdm_zero_neutrino_convention=is_lcdm,
    )
    
    power_data["one_sided_cdm_halo_response"] = response_spectra
    
    # The full response spectra are now saved, so the full divergence fields
    # are no longer needed.
    del div_X
    del div_Y
    
    power_data["response_decomposition"] = (
        compute_response_decomposition_spectra(
            h=h,
            c=c,
            nu=nu,
            delta_c=c["delta"],
            full_response=response_spectra,
            boxsize=boxsize,
        )
    )
    
    # Only afterward compute optional diagnostics.
    if save_species_diagnostics:
        power_data["species_diagnostics"] = (
            compute_species_velocity_diagnostics(
                h=h,
                c=c,
                nu=nu,
                boxsize=boxsize,
            )
        )
    elif save_density_diagnostics:
        power_data["density_spectra"] = compute_density_diagnostics(
            h=h,
            c=c,
            nu=nu,
            boxsize=boxsize,
        )
    
    del h, c, nu

    
    return power_data


def compute_species_velocity_diagnostics(h, c, nu, boxsize):
    """
    Compute Pylians density, current, and velocity spectra for all species.

    For each available species a,

        J_a = (1+delta_a)V_a.

    ``PKL.XPk_vv`` returns

        P_{J_aJ_a}, P_{J_bJ_b}, P_{J_aJ_b},

    while the user-provided ``PKL.XPk_velvel`` routine returns

        P_{V_aV_a}, P_{V_bV_b}, P_{V_aV_b}.

    These are full vector spectra. They are consistency diagnostics and do
    not replace the scalar--divergence spectra used by the beta estimator.

    Notes
    -----
    The halo velocity is reconstructed as P_h/n_h and set to zero in empty
    cells. Current spectra involving halos are generally more robust than
    bulk-velocity spectra involving halos.
    """

    species = {"h": h, "c": c}
    if nu is not None:
        species["nu"] = nu

    validate_species_diagnostic_fields(species)

    axis = 0
    mas = "CIC"
    threads = 1

    out = {
        "available_species": tuple(species.keys()),
        "density_spectra": {},
        "current_spectra": {},
        "velocity_spectra": {},
        "current_definition": "J_a=(1+delta_a)V_a.",
        "current_note": (
            "PKL.XPk_vv constructs the momentum/current fields internally "
            "from delta_a and V_a."
        ),
        "velocity_note": (
            "PKL.XPk_velvel returns full gridded bulk-velocity spectra."
        ),
        "longitudinal_note": (
            "These full vector spectra contain longitudinal and transverse "
            "power. They are not equal to divergence spectra."
        ),
        "halo_velocity_note": (
            "V_h=P_h/n_h is set to zero in empty cells and should be treated "
            "as a grid-dependent diagnostic. J_h=P_h/<n_h> is better defined."
        ),
    }

    names = tuple(species.keys())

    for i, name_a in enumerate(names):
        for name_b in names[i:]:
            field_a = species[name_a]
            field_b = species[name_b]
            pair_key = f"{name_a}_x_{name_b}"

            delta_a = field_a["delta"]
            delta_b = field_b["delta"]
            velocity_a = tuple(
                field_a[f"V_{comp}"] for comp in _COMPONENTS
            )
            velocity_b = tuple(
                field_b[f"V_{comp}"] for comp in _COMPONENTS
            )

            # Density auto- and cross-spectrum.
            density_result = scalar_cross_spectrum(
                delta_a,
                delta_b,
                boxsize,
                MAS="CIC",
            )
            density_entry = {
                "k_h_per_Mpc": density_result[0],
                f"P_delta_{name_a}_delta_{name_a}": density_result[1],
                "Nmodes": density_result[4],
            }
            if name_a == name_b:
                # For a self-pair, P11=P22=P12 up to roundoff. Store once.
                density_entry[f"P_delta_{name_a}_delta_{name_a}"] = (
                    density_result[1]
                )
            else:
                density_entry.update(
                    {
                        f"P_delta_{name_b}_delta_{name_b}": (
                            density_result[2]
                        ),
                        f"P_delta_{name_a}_delta_{name_b}": (
                            density_result[3]
                        ),
                    }
                )
            out["density_spectra"][pair_key] = density_entry

            # Current/momentum auto- and cross-spectra.
            (
                k_current,
                p_ja_ja,
                p_jb_jb,
                p_ja_jb,
                nmodes_current,
            ) = PKL.XPk_vv(
                delta_a.copy(),
                velocity_a[0].copy(),
                velocity_a[1].copy(),
                velocity_a[2].copy(),
                delta_b.copy(),
                velocity_b[0].copy(),
                velocity_b[1].copy(),
                velocity_b[2].copy(),
                float(boxsize),
                axis,
                mas,
                threads,
            )

            current_entry = {
                "k_h_per_Mpc": np.asarray(k_current),
                f"P_J_{name_a}_J_{name_a}": np.asarray(p_ja_ja),
                "Nmodes": np.asarray(nmodes_current),
            }
            if name_a != name_b:
                current_entry.update(
                    {
                        f"P_J_{name_b}_J_{name_b}": np.asarray(p_jb_jb),
                        f"P_J_{name_a}_J_{name_b}": np.asarray(p_ja_jb),
                    }
                )
            out["current_spectra"][pair_key] = current_entry

            # Bulk-velocity auto- and cross-spectra.
            (
                k_velocity,
                p_va_va,
                p_vb_vb,
                p_va_vb,
                nmodes_velocity,
            ) = PKL.XPk_velvel(
                velocity_a[0].copy(),
                velocity_a[1].copy(),
                velocity_a[2].copy(),
                velocity_b[0].copy(),
                velocity_b[1].copy(),
                velocity_b[2].copy(),
                float(boxsize),
                axis,
                mas,
                threads,
            )

            velocity_entry = {
                "k_h_per_Mpc": np.asarray(k_velocity),
                f"P_V_{name_a}_V_{name_a}": np.asarray(p_va_va),
                "Nmodes": np.asarray(nmodes_velocity),
            }
            if name_a != name_b:
                velocity_entry.update(
                    {
                        f"P_V_{name_b}_V_{name_b}": np.asarray(p_vb_vb),
                        f"P_V_{name_a}_V_{name_b}": np.asarray(p_va_vb),
                    }
                )
            out["velocity_spectra"][pair_key] = velocity_entry

            assert_matching_k_and_modes(
                density_result[0],
                density_result[4],
                k_current,
                nmodes_current,
                label=f"density/current {pair_key}",
            )
            assert_matching_k_and_modes(
                density_result[0],
                density_result[4],
                k_velocity,
                nmodes_velocity,
                label=f"density/velocity {pair_key}",
            )

    return out


def validate_species_diagnostic_fields(species):
    """Validate fields required by the optional Pylians diagnostics."""

    reference_shape = next(iter(species.values()))["delta"].shape

    for name, field in species.items():
        if field["delta"].shape != reference_shape:
            raise ValueError(
                f"delta_{name} has shape {field['delta'].shape}; "
                f"expected {reference_shape}."
            )

        for comp in _COMPONENTS:
            key = f"V_{comp}"
            if key not in field:
                raise KeyError(
                    f"Missing {key} for species {name}. Construct the halo "
                    "velocity by enabling save_species_diagnostics."
                )
            if field[key].shape != reference_shape:
                raise ValueError(
                    f"{key} for species {name} has shape {field[key].shape}; "
                    f"expected {reference_shape}."
                )


def build_halo_fields(data, build_velocity_diagnostics=False):
    """
    Construct halo density and normalized-current fields.

    If ``build_velocity_diagnostics`` is true, also construct

        V_h = P_h/n_h,

    with V_h=0 in empty cells. The sparse halo velocity is only intended for
    optional Pylians diagnostics. The response estimator itself uses J_h.
    """

    sub = data["sub_box_data"]
    density = np.asarray(sub["density"], dtype=np.float32)
    mean_density = validate_mean_density(density, "halo")
    occupied = density > 0.0

    one_plus_delta = (density / mean_density).astype(np.float32)
    delta = (one_plus_delta - 1.0).astype(np.float32)

    fields = {
        "label": "halo",
        "one_plus_delta": one_plus_delta,
        "delta": delta,
        "summary": summarize_density(
            density,
            mean_density,
            "halo",
        ),
    }

    for comp in _COMPONENTS:
        p_comp = np.asarray(sub[f"P_{comp}"], dtype=np.float32)

        fields[f"J_{comp}"] = (
            p_comp / mean_density
        ).astype(np.float32)

        if build_velocity_diagnostics:
            v_comp = np.zeros_like(p_comp, dtype=np.float32)
            np.divide(
                p_comp,
                density,
                out=v_comp,
                where=occupied,
            )
            fields[f"V_{comp}"] = v_comp

    return fields


def build_matter_fields(data, field_label):
    """Construct density contrast and coarse-grained matter velocity."""

    sub = data["sub_box_data"]
    density = np.asarray(sub["density"], dtype=np.float32)
    mean_density = validate_mean_density(density, field_label)
    occupied = density > 0.0

    one_plus_delta = (density / mean_density).astype(np.float32)
    delta = (one_plus_delta - 1.0).astype(np.float32)

    fields = {
        "label": field_label,
        "one_plus_delta": one_plus_delta,
        "delta": delta,
        "summary": summarize_density(
            density,
            mean_density,
            field_label,
        ),
    }

    for comp in _COMPONENTS:
        p_comp = np.asarray(sub[f"P_{comp}"], dtype=np.float32)
        v_comp = np.zeros_like(p_comp, dtype=np.float32)
        np.divide(
            p_comp,
            density,
            out=v_comp,
            where=occupied,
        )
        fields[f"V_{comp}"] = v_comp

    return fields


def build_zero_neutrino_fields_like(reference_field):
    """
    Construct the LCDM convention

        delta_nu = 0,
        V_nu     = 0,
        J_nu     = 0.

    The same immutable zero grid is shared by all identically zero fields
    to avoid unnecessary memory allocation.
    """

    shape = reference_field["delta"].shape

    zero = np.zeros(shape, dtype=np.float32)
    one = np.ones(shape, dtype=np.float32)

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
        },
    }

    for comp in _COMPONENTS:
        fields[f"V_{comp}"] = zero
        fields[f"J_{comp}"] = zero

    return fields

def build_Y_components(h, c):
    """Return Y=(1+delta_h)(V_h-V_c)=J_h-(1+delta_h)V_c."""

    one_h = h["one_plus_delta"]
    return tuple(
        (
            h[f"J_{comp}"]
            - one_h * c[f"V_{comp}"]
        ).astype(np.float32)
        for comp in _COMPONENTS
    )


def build_X_components(h, c, nu):
    """
    Return X=(1+delta_h)(V_nu-V_c).

    For LCDM, the supplied synthetic neutrino field has V_nu=0, giving

        X_LCDM = -(1+delta_h)V_c.
    """

    if nu is None:
        raise ValueError(
            "A neutrino field or the synthetic LCDM zero-neutrino "
            "field is required to construct X."
        )

    one_h = h["one_plus_delta"]
    return tuple(
        (
            one_h
            * (nu[f"V_{comp}"] - c[f"V_{comp}"])
        ).astype(np.float32)
        for comp in _COMPONENTS
    )

def build_X_unweighted_components(c, nu):
    """
    Return the unweighted contribution

        X_unweighted = V_nu - V_c.

    For the LCDM zero-neutrino convention, V_nu=0 and therefore

        X_unweighted = -V_c.
    """

    if nu is None:
        raise ValueError(
            "A neutrino field or the synthetic LCDM zero-neutrino "
            "field is required."
        )

    return tuple(
        (
            nu[f"V_{comp}"] - c[f"V_{comp}"]
        ).astype(np.float32, copy=False)
        for comp in _COMPONENTS
    )

def build_Y_unweighted_components(h, c):
    """
    Return the unweighted contribution

        Y_unweighted = V_h - V_c,

    using the same halo bulk velocity definition as build_halo_fields:

        V_h = P_h / n_h
            = J_h / (1 + delta_h).

    In cells with no halos, np.divide leaves V_h equal to zero, exactly as
    in the existing halo-velocity diagnostic implementation.
    """

    one_plus_delta_h = h["one_plus_delta"]
    occupied = one_plus_delta_h > 0.0

    components = []

    for comp in _COMPONENTS:
        # This temporary grid becomes the output Y component, avoiding a
        # separate persistent V_h grid.
        component = np.zeros_like(
            h[f"J_{comp}"],
            dtype=np.float32,
        )

        # First reconstruct V_h directly into component.
        np.divide(
            h[f"J_{comp}"],
            one_plus_delta_h,
            out=component,
            where=occupied,
        )

        # Convert V_h to V_h - V_c in place.
        component -= c[f"V_{comp}"]

        components.append(component)

    return tuple(components)

def compute_one_sided_response_spectra(
    delta_c,
    div_Y,
    div_X,
    boxsize,
    physical_neutrino_driver_available,
    lcdm_zero_neutrino_convention,
):
    """
    Save the six spectra needed for beta_h and its Gaussian covariance.

    beta_h itself is intentionally not formed here.
    """

    delta_x_div_y = scalar_cross_spectrum(
        delta_c,
        div_Y,
        boxsize,
        MAS="None",
    )
    delta_x_div_x = scalar_cross_spectrum(
        delta_c,
        div_X,
        boxsize,
        MAS="None",
    )
    div_y_x_div_x = scalar_cross_spectrum(
        div_Y,
        div_X,
        boxsize,
        MAS="None",
    )

    assert_same_binning(
        delta_x_div_y,
        delta_x_div_x,
        div_y_x_div_x,
        labels=("delta-divY", "delta-divX", "divY-divX"),
    )

    return {
        "beta_estimator_available": True,
    
        "physical_neutrino_driver_available": bool(
            physical_neutrino_driver_available
        ),
    
        "lcdm_zero_neutrino_convention": bool(
            lcdm_zero_neutrino_convention
        ),
    
        "k_h_per_Mpc": delta_x_div_y[0],    
        "P_delta_c_delta_c": delta_x_div_y[1],
        "P_delta_c_divY": delta_x_div_y[3],
        "P_delta_c_divX": delta_x_div_x[3],
    
        "P_divY_divY": delta_x_div_y[2],
        "P_divX_divX": delta_x_div_x[2],
        "P_divY_divX": div_y_x_div_x[3],
    
        "Nmodes": delta_x_div_y[4],
    
        "postprocessing_estimator": (
            "beta_h(k)="
            "P_delta_c_divY(k)/P_delta_c_divX(k)"
        ),
    
        "driver_definition": (
            "Massive-neutrino runs: "
            "X=(1+delta_h)(V_nu-V_c). "
            "LCDM convention: V_nu=J_nu=0, hence "
            "X=-(1+delta_h)V_c."
        ),
    
        "normalization_note": (
            "div(Y) and div(X) are saved without -1/(aH); "
            "the common factor cancels in beta_h."
        ),
    
        "mas_note": (
            "No additional CIC deconvolution is applied to div(Y) "
            "or div(X), because Y and X are nonlinear products of "
            "separately gridded fields."
        ),
    }




def compute_density_diagnostics(h, c, nu, boxsize):
    """
    Save compact density spectra for bias and consistency tests.

    Primitive density fields are direct CIC assignments, so the usual CIC
    deconvolution is requested from Pylians.
    """

    h_x_c = scalar_cross_spectrum(
        h["delta"],
        c["delta"],
        boxsize,
        MAS="CIC",
    )

    out = {
        "k_h_per_Mpc": h_x_c[0],
        "P_hh": h_x_c[1],
        "P_cc": h_x_c[2],
        "P_hc": h_x_c[3],
        "Nmodes": h_x_c[4],
    }

    if nu is not None:
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
            labels=("h-c", "h-nu", "c-nu"),
        )

        out.update(
            {
                "P_nunu": h_x_nu[2],
                "P_hnu": h_x_nu[3],
                "P_cnu": c_x_nu[3],
            }
        )

    out["mas_note"] = (
        "These spectra are formed from directly CIC-deposited density "
        "fields, so the standard CIC correction is requested from Pylians."
    )
    return out


def fft_divergence(vec, boxsize):
    """
    Compute div(F)=i k.F using spectral derivatives and return a real grid.

    For an even grid, each Nyquist derivative component is set to zero. This
    preserves the Hermitian symmetry required for a real inverse transform.
    """

    fx, fy, fz = (
        np.asarray(component, dtype=np.float32)
        for component in vec
    )

    if fx.shape != fy.shape or fx.shape != fz.shape:
        raise ValueError("Vector components must have identical shapes.")
    if fx.ndim != 3:
        raise ValueError(f"Expected 3D fields; got shape {fx.shape}.")

    nx, ny, nz = fx.shape
    dx = float(boxsize) / nx
    dy = float(boxsize) / ny
    dz = float(boxsize) / nz

    kx_1d = (
        2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ).astype(np.float32)
    ky_1d = (
        2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    ).astype(np.float32)
    kz_1d = (
        2.0 * np.pi * np.fft.rfftfreq(nz, d=dz)
    ).astype(np.float32)

    if nx % 2 == 0:
        kx_1d[nx // 2] = 0.0
    if ny % 2 == 0:
        ky_1d[ny // 2] = 0.0
    if nz % 2 == 0:
        kz_1d[-1] = 0.0

    kx = kx_1d[:, None, None]
    ky = ky_1d[None, :, None]
    kz = kz_1d[None, None, :]

    fx_k = np.fft.rfftn(fx)
    div_k = 1j * kx * fx_k
    del fx_k

    fy_k = np.fft.rfftn(fy)
    div_k += 1j * ky * fy_k
    del fy_k

    fz_k = np.fft.rfftn(fz)
    div_k += 1j * kz * fz_k
    del fz_k

    div = np.fft.irfftn(
        div_k,
        s=fx.shape,
    ).real.astype(np.float32)
    del div_k

    return div


def scalar_cross_spectrum(
    field1,
    field2,
    boxsize,
    MAS="None",
    axis=0,
    threads=1,
):
    """Return [k, P11, P22, P12, Nmodes] for two scalar grids."""

    field1 = np.asarray(field1, dtype=np.float32)
    field2 = np.asarray(field2, dtype=np.float32)

    if field1.shape != field2.shape:
        raise ValueError(
            "Scalar fields have inconsistent shapes: "
            f"{field1.shape} and {field2.shape}."
        )

    pk = PKL.XPk(
        [field1.copy(), field2.copy()],
        float(boxsize),
        axis,
        [MAS, MAS],
        threads,
    )

    return [
        np.asarray(pk.k3D),
        np.asarray(pk.Pk[:, 0, 0]),
        np.asarray(pk.Pk[:, 0, 1]),
        np.asarray(pk.XPk[:, 0, 0]),
        np.asarray(pk.Nmodes3D),
    ]


def assert_same_binning(*spectra, labels=None):
    """Check that several scalar-spectrum outputs share bins and mode counts."""

    if len(spectra) < 2:
        return

    if labels is None:
        labels = tuple(
            f"spectrum-{index}" for index in range(len(spectra))
        )

    reference_k = np.asarray(spectra[0][0])
    reference_modes = np.asarray(spectra[0][4])

    for spectrum, label in zip(spectra[1:], labels[1:]):
        if not np.allclose(
            reference_k,
            spectrum[0],
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(f"Inconsistent k bins in {label}.")
        if not np.array_equal(reference_modes, spectrum[4]):
            raise RuntimeError(f"Inconsistent Nmodes in {label}.")


def assert_matching_k_and_modes(
    reference_k,
    reference_modes,
    test_k,
    test_modes,
    label,
):
    """Check binning consistency between different Pylians estimators."""

    if not np.allclose(
        np.asarray(reference_k),
        np.asarray(test_k),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(f"Inconsistent k bins for {label}.")

    if not np.array_equal(
        np.asarray(reference_modes),
        np.asarray(test_modes),
    ):
        raise RuntimeError(f"Inconsistent Nmodes for {label}.")


def validate_mean_density(density, field_label):
    """Return a validated mean CIC count."""

    mean_density = float(np.mean(density, dtype=np.float64))
    if not np.isfinite(mean_density) or mean_density <= 0.0:
        raise ValueError(
            f"{field_label} has non-positive or non-finite mean density: "
            f"{mean_density}"
        )
    return mean_density


def summarize_density(density, mean_density, field_label):
    """Return compact occupancy diagnostics for one gridded species."""

    return {
        "label": field_label,
        "mean_density": float(mean_density),
        "density_min": float(np.min(density)),
        "density_max": float(np.max(density)),
        "empty_cell_fraction": float(np.mean(density <= 0.0)),
    }


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
):
    """Build output metadata."""

    return {
        "analysis_type": "one_sided_cdm_halo_response_spectra",
        "spec": spec,
        "sim_type": sim,
        "ngrid": int(ngrid),
        "boxsize_Mpc_over_h": float(boxsize),
        "mass_selection": string,
        "mass_cut": float(mass_cut),
        "mass_width": float(mass_width),
        "lcdm_mode": bool(is_lcdm),
        "density_diagnostics_saved": bool(
            save_density_diagnostics or save_species_diagnostics
        ),
        "species_diagnostics_saved": bool(save_species_diagnostics),
        "primitive_fields": {
            "density": "n_a(x)=sum_p W_CIC(x-x_p)",
            "P_i": "P_a,i(x)=sum_p W_CIC(x-x_p)v_p,i",
            "V_i": "V_a,i=P_a,i/n_a, with V=0 in empty cells",
            "J_h_i": "J_h,i=P_h,i/<n_h>=(1+delta_h)V_h,i",
        },
        "one_sided_fields": {
            "Y": "Y=(1+delta_h)(V_h-V_c)=J_h-(1+delta_h)V_c",
            "X": ("X=(1+delta_h)(V_nu-V_c); for LCDM, V_nu=0, "
            "so X=-(1+delta_h)V_c" ),
        },
        "saved_response_spectra": (
            "For both massive-neutrino and LCDM runs: "
            "P_delta_c_delta_c, P_delta_c_divY, P_delta_c_divX, "
            "P_divY_divY, P_divX_divX, P_divY_divX, Nmodes. "
            "For LCDM, V_nu=J_nu=0 is imposed."
        ),
        "optional_species_diagnostics": (
            "All available density, current, and velocity auto- and "
            "cross-spectra from Pylians."
        ),
        "postprocessing_estimator": (
            "beta_h(k)=P_delta_c_divY(k)/P_delta_c_divX(k)"
        ),
    }


def validate_inputs(string, ngrid_step):
    """Validate user-facing options."""

    if string not in {"+", "-", "bin", "all"}:
        raise ValueError(
            "string must be one of '+', '-', 'bin', or 'all'; "
            f"got {string!r}"
        )
    if ngrid_step <= 0:
        raise ValueError(
            f"ngrid_step must be positive; got {ngrid_step}"
        )


def simulation_def(spec, sim_type, string, mass_cut, mass_width):
    """Return an output simulation tag."""

    if string == "+":
        halo_tag = f"halo_mass_{mass_cut:.1e}"
    elif string == "bin":
        halo_tag = (
            f"halo_mass_{mass_cut:.1e}_mass_bin_{mass_width}"
        )
    elif string == "-":
        halo_tag = f"halo_mass_le_{mass_cut:.1e}"
    elif string == "all":
        halo_tag = "halo_all"
    else:
        raise ValueError(
            f"Unknown halo selection string={string!r}"
        )

    return f"{sim_type}_{spec}_one_sided_{halo_tag}_cdm_nu"


def halo_directory_name(string, mass_cut, mass_width):
    """Return the halo input-directory name."""

    if string == "+":
        return f"halo_mass_{mass_cut:.0e}".replace("+", "")
    if string == "bin":
        return f"halo_mass_{mass_cut:.0e}".replace("+", "")
    if string == "-":
        return f"halo_mass_le_{mass_cut:.0e}".replace("+", "")
    if string == "all":
        return "halo_all"

    raise ValueError(
        f"Unknown halo selection string={string!r}"
    )


def halo_pickle_basename(
    ngrid,
    sim,
    spec,
    string,
    mass_cut,
    mass_width,
):
    """Return the halo pickle basename."""

    if string == "+":
        sim_name = f"{sim}_{spec}_halo_mass_{mass_cut:.1e}"
    elif string == "bin":
        sim_name = (
            f"{sim}_{spec}_halo_mass_{mass_cut:.1e}_"
            f"mass_bin_{mass_width}"
        )
    elif string == "-":
        sim_name = f"{sim}_{spec}_halo_mass_le_{mass_cut:.1e}"
    elif string == "all":
        sim_name = f"{sim}_{spec}_halo_all"
    else:
        raise ValueError(
            f"Unknown halo selection string={string!r}"
        )

    return f"data_ngrid_{ngrid}_sim_{sim_name}.pickle"


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
    """Load one gridded species pickle."""

    if bulk_species == "halo":
        halo_dir = halo_directory_name(
            string,
            mass_cut,
            mass_width,
        )
        basename = halo_pickle_basename(
            ngrid,
            sim,
            spec,
            string,
            mass_cut,
            mass_width,
        )
        path = os.path.join(
            file_path,
            spec,
            sim,
            halo_dir,
            "output",
            basename,
        )
    else:
        basename = (
            f"data_ngrid_{ngrid}_sim_{sim}_{spec}_"
            f"{bulk_species}.pickle"
        )
        path = os.path.join(
            file_path,
            spec,
            sim,
            bulk_species,
            "output",
            basename,
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing input pickle: {path}"
        )

    return load(path)


def save_and_print_usage(
    start_time,
    start_mem,
    simulation,
    ngrid,
    power_data,
    save_path,
):
    """Save one output and print resource use."""

    save_dataframe(
        power_data,
        simulation,
        ngrid,
        save_path,
    )
    print(f"{simulation}\n", flush=True)
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
    """Write one output pickle."""

    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(
        save_path,
        (
            f"One_sided_response_spectra_ngrid_{ngrid}_"
            f"sim_{simulation}.pickle"
        ),
    )

    with open(output_file, "wb") as handle:
        pickle.dump(
            power_data,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def print_usage(start_time, start_mem, message=""):
    """Print elapsed time and change in resident memory."""

    elapsed_time = time.time() - start_time
    elapsed_mem = (
        psutil.Process().memory_info().rss - start_mem
    )
    print(
        f"{message} - Time: {elapsed_time:.2f} s, "
        f"Memory change: {elapsed_mem / 1024**2:.2f} MB",
        flush=True,
    )
