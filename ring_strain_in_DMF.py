#!/usr/bin/env python
from __future__ import annotations

from typing import Dict, Tuple, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

from pyscf import gto, dft, solvent
import pyscf.lib as lib
from pyscf.geomopt.geometric_solver import optimize as geomopt_opt


# ===================== Global QC settings =====================

BASIS: str = "def2-svp"
DFT_XC: str = "wb97x"

MAX_CYC: int = 50

DEFAULT_CHARGE: int = 0
DEFAULT_SPIN: int = 0   # closed shell (2S = 0)

# DMF dielectric constant (approx)
DMF_EPS: float = 36.7

lib.num_threads(8)  # adjust for your machine


# ===================== Core set definition (CP1–CP5) =====================

CORE_SET = [
    {
        "name": "CP1",
        # cyclic: SMILES #1
        "cyc_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC([C@@H](NC([C@@H](NC(COCCOCCOCCNC2=O)=O)"
            "C(C)C)=O)C)=O)=O)=C(C(C)(C2)C)C(C)=C1"
        ),
        # reference: SMILES #11
        "ref_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC(CNC(CNC(COCCOCCOCCNC2=O)=O)=O)=O)=O)"
            "=C(C(C)(C2)C)C(C)=C1"
        ),
    },
    {
        "name": "CP2",
        # cyclic: SMILES #2
        "cyc_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC([C@@H](NC([C@@H](NC(COCCOCCOCCNC2=O)=O)"
            "CN=[N+]=[N-])=O)CN=[N+]=[N-])=O)=O)=C(C(C)(C2)C)C(C)=C1"
        ),
        # reference: SMILES #12
        "ref_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC([C@H](CN=[N+]=[N-])NC(CNC(COCCOCCOCCNC2=O)"
            "=O)=O)=O)=O)=C(C(C)(C2)C)C(C)=C1"
        ),
    },
    {
        "name": "CP3",
        # cyclic: SMILES #3
        "cyc_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC([C@@H](NC([C@@H](NC(COCCOCCOCCNC2=O)=O)"
            "CN3N=NC(OC[C@@H]4[C@H]([C@@H]([C@H]([C@H](O4)O)O)O)O[C@@H]"
            "([C@@H]([C@H]([C@H]5O)O)O)O[C@@H]5CO)=C3)=O)CN6N=NC(OC[C@@H]7"
            "[C@H]([C@@H]([C@H]([C@H](O7)O)O)O)O[C@@H]([C@@H]([C@H]([C@H]8O)"
            "O)O)O[C@@H]8CO)=C6)=O)=O)=C(C(C)(C2)C)C(C)=C1"
        ),
        # reference: SMILES #13
        "ref_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC([C@@H](NC(CNC(COCCOCCOCCNC2=O)=O)=O)"
            "CN3N=NC(OC[C@@H]4[C@H]([C@@H]([C@H]([C@H](O4)O)O)O)O[C@@H]"
            "([C@@H]([C@H]([C@H]5O)O)O)O[C@@H]5CO)=C3)=O)=O)=C(C(C)(C2)C)"
            "C(C)=C1"
        ),
    },
    {
        "name": "CP4",
        # cyclic: SMILES #4
        "cyc_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC([C@@H](NC([C@@H](NC(COCCOCCOCCNCC2)=O)"
            "CN=[N+]=[N-])=O)C)=O)=O)=C(C2(C)C)C(C)=C1"
        ),
        # reference: SMILES #14
        "ref_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC(CNC([C@H](CN=[N+]=[N-])NC(COCCOCCOCCNC2=O)"
            "=O)=O)=O)=O)=C(C(C)(C2)C)C(C)=C1"
        ),
    },
    {
        "name": "CP5",
        # cyclic: SMILES #5
        "cyc_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC([C@@H](NC([C@@H](NC(COCCOCCOCCNC2=O)=O)"
            "CN3N=NC(OC[C@@H]4[C@H]([C@@H]([C@H]([C@H](O4)O)O)O)O[C@@H]"
            "([C@@H]([C@H]([C@H]5O)O)O)O[C@@H]5CO)=C3)=O)C)=O)=O)=C(C(C)(C2)"
            "C)C(C)=C1"
        ),
        # reference: SMILES #15
        "ref_smiles": (
            "CC1=C([N+]([O-])=O)C(OC(CNC(CNC([C@@H](NC(COCCOCCOCCNC2=O)=O)"
            "CN3N=NC(OC[C@@H]4[C@H]([C@@H]([C@H]([C@H](O4)O)O)O)O[C@@H]"
            "([C@@H]([C@H]([C@H]5O)O)O)O[C@@H]5CO)=C3)=O)=O)=O)=C(C(C)(C2)C)"
            "C(C)=C1"
        ),
    },
]


# ===================== RDKit helpers =====================

def mol_from_smiles_3d(smiles: str, name: str) -> Chem.Mol:
    """
    Build an RDKit molecule with hydrogens and a 3D FF-minimized conformer.

    This version is robust to RDKit versions that don't support certain
    ETKDGv3 attributes (e.g. maxAttempts).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for {name}: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if hasattr(params, "useRandomCoords"):
        params.useRandomCoords = True

    # Simple retry loop instead of using params.maxAttempts
    cid = -1
    for _ in range(20):
        cid = AllChem.EmbedMolecule(mol, params)
        if cid >= 0:
            break

    if cid < 0:
        raise ValueError(f"3D embedding failed for {name} after 20 attempts.")

    try:
        res = AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
    except Exception:
        res = -1
    if res != 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
        except Exception as e:
            print(f"Warning: FF optimisation failed for {name}: {e}")

    mol.SetProp("_Name", name)
    return mol


# ===================== RDKit -> PySCF =====================

def rdkit_to_pyscf_mol(
    mol: Chem.Mol,
    basis: str = BASIS,
    charge: int = DEFAULT_CHARGE,
    spin: int = DEFAULT_SPIN,
) -> gto.Mole:
    conf = mol.GetConformer()
    atoms = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        atoms.append([atom.GetSymbol(), (pos.x, pos.y, pos.z)])

    mol_pyscf = gto.M(
        atom=atoms,
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        verbose=3,
    )
    mol_pyscf.build()
    return mol_pyscf


# ===================== Utilities =====================

def hartree_to_kcalmol(e_h: float) -> float:
    return e_h * 627.509474


def write_xyz_from_pyscf(mol: gto.Mole, path: str, comment: str = "") -> None:
    """Write a PySCF Mole geometry to an XYZ file (Angstrom)."""
    coords = mol.atom_coords(unit="Angstrom")
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    with open(path, "w") as f:
        f.write(f"{mol.natm}\n")
        f.write(comment.strip() + "\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s:2s}  {x:16.8f}  {y:16.8f}  {z:16.8f}\n")


# ===================== QC drivers =====================

def geomopt_dft_gas(
    mol_pyscf: gto.Mole,
    name: str,
) -> Tuple[gto.Mole, float]:
    """
    DFT geometry optimization (gas phase), then a clean DFT single point.

    Returns:
        (optimized_mol, E_dft_gas)
    """
    print(f"\n=== {name}: DFT geometry optimization ({DFT_XC}/{BASIS}) ===")
    mf0 = dft.RKS(mol_pyscf)
    mf0.xc = DFT_XC
    mf0.max_cycle = MAX_CYC

    mol_opt = geomopt_opt(mf0)
    print(f"{name}: optimization done; natm = {mol_opt.natm}")

    print(f"\n=== {name}: DFT single point on optimized geometry ===")
    mf = dft.RKS(mol_opt)
    mf.xc = DFT_XC
    mf.max_cycle = MAX_CYC
    e_dft = mf.kernel()
    print(f"{name} DFT gas-phase energy (opt geom): {e_dft:.10f} Ha")

    return mol_opt, e_dft


def single_point_dft_ddcosmo_dmf(
    mol_pyscf: gto.Mole,
    name: str,
) -> float:
    """
    DFT single-point with ddCOSMO (DMF-like dielectric) on a given geometry.
    """
    print(f"\n=== {name}: DFT + ddCOSMO (DMF, {DFT_XC}/{BASIS}) ===")
    mf = dft.RKS(mol_pyscf)
    mf.xc = DFT_XC
    mf.max_cycle = MAX_CYC

    # Attach ddCOSMO solvent model
    mf = solvent.ddCOSMO(mf)
    mf.with_solvent.eps = DMF_EPS

    e_cosmo = mf.kernel()
    print(f"{name} DFT+ddCOSMO(DMF) energy: {e_cosmo:.10f} Ha")
    return e_cosmo


# ===================== Core-set construction & strain =====================

def build_core_set_molecules(idx: int) -> Dict[str, Dict[str, gto.Mole]]:
    """Build PySCF molecules for a single system selected by idx (0–4)."""
    if idx < 0 or idx >= len(CORE_SET):
        raise IndexError(f"idx {idx} out of range for CORE_SET of length {len(CORE_SET)}.")

    entry = CORE_SET[idx]
    name = entry["name"]

    print(f"\n=== Building system {name} (idx={idx}) ===")
    cyc_rd = mol_from_smiles_3d(entry["cyc_smiles"], f"{name}_cyc")
    ref_rd = mol_from_smiles_3d(entry["ref_smiles"], f"{name}_ref")

    cyc_mol = rdkit_to_pyscf_mol(cyc_rd)
    ref_mol = rdkit_to_pyscf_mol(ref_rd)

    db: Dict[str, Dict[str, gto.Mole]] = {name: {"cyc": cyc_mol, "ref": ref_mol}}
    return db


def compute_strains_core_set(
    idx: int,
    out_dir: str = "results",
) -> pd.DataFrame:
    """
    Compute ring strain for a single core-set member (selected by idx 0–4)
    using DFT-only:

      1. DFT geometry optimization (gas) for cyc/ref.
      2. DFT single-point (gas) on optimized geometries.
      3. DFT single-point with ddCOSMO(DMF) on optimized geometries.

    Saves optimized XYZ geometries and returns a DataFrame with columns
        DFT_gas, DFT_COSMO_DMF, COSMO_impact
    (all in kcal/mol).
    """
    os.makedirs(out_dir, exist_ok=True)

    db = build_core_set_molecules(idx=idx)
    records = []

    for name, mols in db.items():
        print(f"\n################ {name} ################")
        cyc_0 = mols["cyc"]
        ref_0 = mols["ref"]

        # ----- DFT geomopt + DFT gas single point -----
        cyc_opt, e_cyc_dft_gas = geomopt_dft_gas(cyc_0, f"{name} (cyc)")
        ref_opt, e_ref_dft_gas = geomopt_dft_gas(ref_0, f"{name} (ref)")
        dE_dft_gas = e_cyc_dft_gas - e_ref_dft_gas

        # Save optimized geometries as XYZ
        cyc_xyz_path = os.path.join(out_dir, f"{name}_cyc_opt.xyz")
        ref_xyz_path = os.path.join(out_dir, f"{name}_ref_opt.xyz")
        write_xyz_from_pyscf(
            cyc_opt,
            cyc_xyz_path,
            comment=f"{name} cyclic optimized geometry ({DFT_XC}/{BASIS})",
        )
        write_xyz_from_pyscf(
            ref_opt,
            ref_xyz_path,
            comment=f"{name} reference optimized geometry ({DFT_XC}/{BASIS})",
        )
        print(f"Wrote XYZ: {cyc_xyz_path}")
        print(f"Wrote XYZ: {ref_xyz_path}")

        # ----- DFT + ddCOSMO(DMF) single points on optimized geometries -----
        e_cyc_cosmo = single_point_dft_ddcosmo_dmf(cyc_opt, f"{name} (cyc)")
        e_ref_cosmo = single_point_dft_ddcosmo_dmf(ref_opt, f"{name} (ref)")
        dE_dft_cosmo = e_cyc_cosmo - e_ref_cosmo

        # COSMO "impact" on ring strain: solvent-strain - gas-strain
        cosmo_impact = dE_dft_cosmo - dE_dft_gas

        record = {
            "system": name,
            "DFT_gas": hartree_to_kcalmol(dE_dft_gas),
            "DFT_COSMO_DMF": hartree_to_kcalmol(dE_dft_cosmo),
            "COSMO_impact": hartree_to_kcalmol(cosmo_impact),
        }

        print(f"\nRing strain for {name} (kcal/mol):")
        for k, v in record.items():
            if k != "system":
                print(f"  {k:14s}: {v:7.2f}")

        records.append(record)

    df = pd.DataFrame(records).set_index("system")

    # Save DF as CSV in out_dir
    csv_path = os.path.join(out_dir, "ring_strain_results.csv")
    df.to_csv(csv_path)
    print(f"\nSaved ring-strain DataFrame to: {csv_path}")

    return df


# ===================== Plotting helpers =====================

def plot_grouped_bars(df: pd.DataFrame) -> None:
    systems = df.index.tolist()
    methods = df.columns.tolist()
    strain = df.to_numpy()

    n_sys, n_methods = strain.shape
    x = np.arange(n_sys)
    width = 0.15

    _, ax = plt.subplots(figsize=(7, 4))

    for j, method in enumerate(methods):
        ax.bar(
            x + (j - (n_methods - 1) / 2) * width,
            strain[:, j],
            width=width,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylabel("Ring strain (kcal/mol)")
    ax.set_title("Ring strain by method and system")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_method_scatter(df: pd.DataFrame, method_x: str, method_y: str) -> None:
    if method_x not in df.columns or method_y not in df.columns:
        raise ValueError("Requested methods not found in DataFrame columns.")

    x_vals = df[method_x].to_numpy()
    y_vals = df[method_y].to_numpy()
    systems = df.index.tolist()

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(x_vals, y_vals)

    mn = min(x_vals.min(), y_vals.min())
    mx = max(x_vals.max(), y_vals.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--")

    for i, label in enumerate(systems):
        ax.annotate(
            label,
            (x_vals[i], y_vals[i]),
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel(f"{method_x} ring strain (kcal/mol)")
    ax.set_ylabel(f"{method_y} ring strain (kcal/mol)")
    ax.set_title(f"{method_x} vs {method_y} ring strain")

    plt.tight_layout()
    plt.show()


# ===================== Script entry point =====================

def run_dft(idx: int, out_dir: str = "results") -> pd.DataFrame:
    """
    Convenience wrapper:
    - DFT geomopt (gas) for cyc/ref
    - DFT gas single points
    - DFT + ddCOSMO(DMF) single points
    - Save optimized XYZs and ring-strain CSV in out_dir.
    """
    df = compute_strains_core_set(idx=idx, out_dir=out_dir)
    print("\n=== Ring strain summary (kcal/mol) ===")
    print(df)
    return df

