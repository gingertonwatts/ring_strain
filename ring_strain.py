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

BASIS: str = "def2-svp"
DFT_XC: str = "wb97x"

MAX_CYC: int = 50

DEFAULT_CHARGE: int = 0
DEFAULT_SPIN: int = 0   # closed shell (2S = 0)

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
    """Build an RDKit molecule with hydrogens and a 3D FF-minimized conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for {name}: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.useRandomCoords = True

    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise ValueError(f"3D embedding failed for {name} after {params.maxAttempts} attempts.")

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


# ===================== QC drivers =====================

def hartree_to_kcalmol(e_h: float) -> float:
    return e_h * 627.509474


def single_point_dft(mol_pyscf: gto.Mole, name: str) -> float:
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
) -> Tuple[float, Optional[float], Optional[float]]:
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


def compute_strains_core_set(idx: int, do_ccsd: bool = False) -> pd.DataFrame:
    """
    Compute ring strain for a single core-set member (selected by idx 0–4).

    Returns:
        DataFrame with one row (system) and columns HF, DFT, (optional CCSD),
        in kcal/mol.
    """
    db = build_core_set_molecules(idx=idx)
    records = []

    for name, mols in db.items():
        print(f"\n################ {name} ################")
        cyc = mols["cyc"]
        ref = mols["ref"]

        # DFT
        e_cyc_dft = single_point_dft(cyc, f"{name} (cyc)")
        e_ref_dft = single_point_dft(ref, f"{name} (ref)")
        dE_dft = e_cyc_dft - e_ref_dft

        # HF / (CCSD)
        e_cyc_hf, _, e_cyc_ccsd = single_point_hf_mp2_cc(
            cyc, f"{name} (cyc)", do_mp2=False, do_ccsd=do_ccsd
        )
        e_ref_hf, _, e_ref_ccsd = single_point_hf_mp2_cc(
            ref, f"{name} (ref)", do_mp2=False, do_ccsd=do_ccsd
        )

        dE_hf = e_cyc_hf - e_ref_hf
        record = {
            "system": name,
            "HF": hartree_to_kcalmol(dE_hf),
            "DFT": hartree_to_kcalmol(dE_dft),
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

def run_dft(idx: int):
    """
    Convenience wrapper:
    run DFT/HF ring strain for a single core-set member (idx 0–4),
    print the summary and return the DataFrame.
    """
    df = compute_strains_core_set(idx=idx, do_ccsd=False)
    print("\n=== Ring strain summary (kcal/mol) ===")
    print(df)
    return df
