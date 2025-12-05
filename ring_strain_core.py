"""ring_strain_core.py

Build a small core set of cyclic vs linear peptides, compute ring strain
with PySCF (HF, DFT, MP2, optional CCSD), and provide simple plotting helpers.

Usage in a Jupyter notebook:

    import ring_strain_core as rs

    results_df = rs.compute_strains_core_set(do_ccsd=False)
    display(results_df)

    rs.plot_grouped_bars(results_df)
    rs.plot_method_scatter(results_df, "DFT", "MP2")
    rs.plot_method_scatter(results_df, "HF", "MP2")

Dependencies:
    pip install rdkit-pypi pyscf matplotlib pandas
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

from pyscf import gto, scf, dft, mp, cc
import pyscf.lib as lib


# ===================== Global QC settings =====================

# Basis and functional
BASIS: str = "def2-svp"
# NOTE: PySCF does NOT support 'wb97x-d' directly; 'wb97x' is supported via libxc.
DFT_XC: str = "wb97x"   # you can change this to e.g. "b3lyp" if you like

MAX_CYC: int = 50

DEFAULT_CHARGE: int = 0
DEFAULT_SPIN: int = 0   # closed shell (2S = 0)

# Threading
lib.num_threads(8)  # adjust if desired


# ===================== Core set definition =====================

CORE_SET = [
    {
        "name": "Gly4",
        "cyc_smiles": "N1CC(=O)NCC(=O)NCC(=O)NCC(=O)1",              # cyclo-[Gly-Gly-Gly-Gly]
        "ref_smiles": "CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NC",        # Ac-Gly-Gly-Gly-Gly-NHMe
    },
    {
        "name": "Ala4",
        # cyclo-[Ala-Ala-Ala-Ala]
        "cyc_smiles": "N1C(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)1",
        # Ac-Ala-Ala-Ala-Ala-NHMe
        "ref_smiles": "CC(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC",
    },
    {
        "name": "Val4",
        # cyclo-[Val-Val-Val-Val] (one reasonable pattern)
        "cyc_smiles": "N1C(C(C)C)C(=O)NC(C(C)C)C(=O)NC(C(C)C)C(=O)NC(C(C)C)C(=O)1",
        # Ac-Val-Val-Val-Val-NHMe
        "ref_smiles": (
            "CC(=O)NC(C(C)C)C(=O)NC(C(C)C)C(=O)NC(C(C)C)C(=O)"
            "NC(C(C)C)C(=O)NC"
        ),
    },
]


# ===================== RDKit helpers =====================

def mol_from_smiles_3d(smiles: str, name: str) -> Chem.Mol:
    """Build an RDKit molecule with hydrogens and a 3D FF-minimized conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for {name}: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 1
    AllChem.EmbedMolecule(mol, params)

    # Try MMFF, fall back to UFF if needed
    try:
        res = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        res = -1
    if res != 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
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
    """Convert an RDKit Mol (with conformer) to a PySCF gto.Mole."""
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


# ===================== QC drivers =====================

def hartree_to_kcalmol(e_h: float) -> float:
    return e_h * 627.509474


def single_point_dft(mol_pyscf: gto.Mole, name: str) -> float:
    """DFT single point (RKS)."""
    print(f"\n=== {name}: DFT ({DFT_XC}/{BASIS}) ===")
    mf = dft.RKS(mol_pyscf)
    mf.xc = DFT_XC
    mf.max_cycle = MAX_CYC
    e_dft = mf.kernel()
    print(f"{name} DFT energy: {e_dft:.10f} Ha")
    return e_dft


def single_point_hf_mp2_cc(
    mol_pyscf: gto.Mole,
    name: str,
    do_mp2: bool = False,
    do_ccsd: bool = False,
) -> Tuple[float, float, Optional[float]]:
    """Run HF, MP2, and optional CCSD single points.

    Returns:
        (E_HF, E_MP2, E_CCSD or None) in Hartree.
    """
    print(f"\n=== {name}: HF ({BASIS}) ===")
    mf_hf = scf.RHF(mol_pyscf)
    mf_hf.max_cycle = MAX_CYC
    e_hf = mf_hf.kernel()
    print(f"{name} HF energy: {e_hf:.10f} Ha")

    e_mp2: Optional[float] = None
    if do_mp2:
        print(f"\n=== {name}: MP2 ({BASIS}) ===")
        mp_calc = mp.MP2(mf_hf)
        e_mp2, _ = mp_calc.kernel()
        print(f"{name} MP2 energy: {e_mp2:.10f} Ha")

    e_ccsd: Optional[float] = None
    if do_ccsd:
        print(f"\n=== {name}: CCSD ({BASIS}) ===")
        cc_calc = cc.CCSD(mf_hf)
        e_ccsd, _ = cc_calc.kernel()
        print(f"{name} CCSD energy: {e_ccsd:.10f} Ha")

    return e_hf, e_mp2, e_ccsd


# ===================== Core-set construction & strain =====================

def build_core_set_molecules() -> Dict[str, Dict[str, gto.Mole]]:
    """Build PySCF molecules for each system in CORE_SET.

    Returns:
        db[system_name]["cyc"] and db[system_name]["ref"].
    """
    db: Dict[str, Dict[str, gto.Mole]] = {}
    for entry in CORE_SET:
        name = entry["name"]
        print(f"\n=== Building system {name} ===")
        cyc_rd = mol_from_smiles_3d(entry["cyc_smiles"], f"{name}_cyc")
        ref_rd = mol_from_smiles_3d(entry["ref_smiles"], f"{name}_ref")

        cyc_mol = rdkit_to_pyscf_mol(cyc_rd)
        ref_mol = rdkit_to_pyscf_mol(ref_rd)

        db[name] = {"cyc": cyc_mol, "ref": ref_mol}
    return db


def compute_strains_core_set(do_ccsd: bool = False) -> pd.DataFrame:
    """Compute ring strain for each system and method; return a DataFrame (kcal/mol)."""
    db = build_core_set_molecules()
    records = []

    for name, mols in db.items():
        print(f"\n################ {name} ################")
        cyc = mols["cyc"]
        ref = mols["ref"]

        # DFT
        e_cyc_dft = single_point_dft(cyc, f"{name} (cyc)")
        e_ref_dft = single_point_dft(ref, f"{name} (ref)")
        dE_dft = e_cyc_dft - e_ref_dft

        # HF / MP2 / (CCSD)
        e_cyc_hf, e_cyc_mp2, e_cyc_ccsd = single_point_hf_mp2_cc(
            cyc, f"{name} (cyc)", do_ccsd=do_ccsd
        )
        e_ref_hf, e_ref_mp2, e_ref_ccsd = single_point_hf_mp2_cc(
            ref, f"{name} (ref)", do_ccsd=do_ccsd
        )

        dE_hf = e_cyc_hf - e_ref_hf
        #dE_mp2 = e_cyc_mp2 - e_ref_mp2

        record = {
            "system": name,
            "HF": hartree_to_kcalmol(dE_hf),
            "DFT": hartree_to_kcalmol(dE_dft),
            #"MP2": hartree_to_kcalmol(dE_mp2),
        }

        if do_ccsd and (e_cyc_ccsd is not None) and (e_ref_ccsd is not None):
            dE_ccsd = e_cyc_ccsd - e_ref_ccsd
            record["CCSD"] = hartree_to_kcalmol(dE_ccsd)

        print(f"\nRing strain for {name} (kcal/mol):")
        for k, v in record.items():
            if k != "system":
                print(f"  {k:4s}: {v:7.2f}")

        records.append(record)

    df = pd.DataFrame(records).set_index("system")
    return df


# ===================== Plotting helpers =====================

def plot_grouped_bars(df: pd.DataFrame) -> None:
    """Grouped bar chart of ring strain (kcal/mol) by method and system."""
    systems = df.index.tolist()
    methods = df.columns.tolist()
    strain = df.to_numpy()  # shape (n_sys, n_methods)

    n_sys, n_methods = strain.shape
    x = np.arange(n_sys)
    width = 0.15

    fig, ax = plt.subplots(figsize=(7, 4))

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
    """Scatter plot comparing two methods' ring strain across systems."""
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


# ===================== Script entry point (optional) =====================

if __name__ == "__main__":
    df = compute_strains_core_set(do_ccsd=False)
    print("\n=== Ring strain summary (kcal/mol) ===")
    print(df)
