#!/usr/bin/env python3
"""
ring_strain_graph_fix.py

Robust construction of:
  M (macrocycle), M_open (cut + cap), X (local reference fragment), X_open (cut + cap)

PLUS utilities:
  - list_candidate_cut_bonds / choose_subset_bonds_evenly
  - ETKDG conformers + MMFF ranking + XYZ saving
  - plotting graphs using RDKit connectivity -> NetworkX (no distance-based artifacts)

This file is intended to be imported as:
  import ring_strain_graph_fix as rs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import networkx as nx
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdDepictor


# ----------------------------
# Bond selection helpers
# ----------------------------

def is_cuttable_ring_bond(b: Chem.Bond,
                          allow_aromatic: bool = False,
                          allow_multiple: bool = False,
                          avoid_shared_ring_bonds: bool = True) -> bool:
    """Heuristic for bonds we allow to cut when opening a macrocycle."""
    if not b.IsInRing():
        return False
    if (not allow_aromatic) and b.GetIsAromatic():
        return False
    if (not allow_multiple) and b.GetBondType() != Chem.BondType.SINGLE:
        return False
    if avoid_shared_ring_bonds:
        ri = b.GetOwningMol().GetRingInfo()
        count = 0
        for ring in ri.BondRings():
            if b.GetIdx() in ring:
                count += 1
                if count > 1:
                    return False
    return True

def list_candidate_cut_bonds(mol: Chem.Mol, **kwargs) -> List[int]:
    """Return bond indices that pass is_cuttable_ring_bond."""
    return [b.GetIdx() for b in mol.GetBonds() if is_cuttable_ring_bond(b, **kwargs)]

def choose_subset_bonds_evenly(bond_indices: List[int], k: int) -> List[int]:
    """Pick ~evenly spaced indices from a list (deterministic)."""
    if k <= 0:
        return []
    if k >= len(bond_indices):
        return list(bond_indices)
    out = []
    for i in range(k):
        j = round(i * (len(bond_indices) - 1) / (k - 1))
        out.append(bond_indices[j])
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq


# ----------------------------
# Utilities: mapping + submol
# ----------------------------

def with_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy with atom map numbers = old_idx+1."""
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return m

def induced_submol(mol: Chem.Mol, atom_ids: Set[int]) -> Chem.Mol:
    """Induced submolecule on the specified atom set, preserving atom map numbers."""
    rw = Chem.RWMol()
    old2new = {}
    for old_idx in sorted(atom_ids):
        a = mol.GetAtomWithIdx(old_idx)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetIsAromatic(a.GetIsAromatic())
        na.SetChiralTag(a.GetChiralTag())
        na.SetAtomMapNum(a.GetAtomMapNum())
        new_idx = rw.AddAtom(na)
        old2new[old_idx] = new_idx
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in atom_ids and j in atom_ids:
            rw.AddBond(old2new[i], old2new[j], b.GetBondType())
            nb = rw.GetBondBetweenAtoms(old2new[i], old2new[j])
            nb.SetIsAromatic(b.GetIsAromatic())
    sub = rw.GetMol()
    Chem.SanitizeMol(sub)
    return sub

def bfs_atoms_within_radius(mol: Chem.Mol, centers: List[int], radius: int) -> Set[int]:
    """Atoms within graph distance <= radius from any center."""
    if radius < 0:
        return set()
    dist = {c: 0 for c in centers}
    frontier = list(centers)
    seen = set(centers)
    while frontier:
        u = frontier.pop(0)
        du = dist[u]
        if du == radius:
            continue
        for v in [n.GetIdx() for n in mol.GetAtomWithIdx(u).GetNeighbors()]:
            if v not in seen:
                seen.add(v)
                dist[v] = du + 1
                frontier.append(v)
    return seen

def find_bond_by_maps(mol: Chem.Mol, map1: int, map2: int) -> int:
    """Find a bond between two atoms identified by atom map numbers."""
    idx1 = idx2 = None
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == map1:
            idx1 = a.GetIdx()
        elif a.GetAtomMapNum() == map2:
            idx2 = a.GetIdx()
    if idx1 is None or idx2 is None:
        raise ValueError("Mapped atoms not found in molecule.")
    b = mol.GetBondBetweenAtoms(idx1, idx2)
    if b is None:
        raise ValueError("Mapped atoms are not directly bonded in this molecule.")
    return b.GetIdx()

def break_bond_and_cap(mol: Chem.Mol, bond_idx: int, cap_atomic_num: int = 1) -> Chem.Mol:
    """Break a bond and cap both ends (default H)."""
    frag = Chem.FragmentOnBonds(Chem.Mol(mol), [bond_idx], addDummies=True, dummyLabels=[(999, 999)])
    rw = Chem.RWMol(frag)
    dummy_idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_idxs) != 2:
        raise RuntimeError(f"Expected 2 dummy atoms after bond break, got {len(dummy_idxs)}.")
    for d in sorted(dummy_idxs, reverse=True):
        d_atom = rw.GetAtomWithIdx(d)
        nbrs = [n.GetIdx() for n in d_atom.GetNeighbors()]
        if len(nbrs) != 1:
            raise RuntimeError("Dummy atom should have exactly one neighbor.")
        nbr = nbrs[0]
        cap_idx = rw.AddAtom(Chem.Atom(cap_atomic_num))
        rw.AddBond(nbr, cap_idx, Chem.BondType.SINGLE)
        rw.RemoveAtom(d)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out

def mol_formula(m: Chem.Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(Chem.RemoveHs(m))


# ----------------------------
# Build M, M_open, X, X_open
# ----------------------------

def build_structures(smiles: str, bond_idx: int, ref_radius: int = 2, cap: str = "H") -> Dict[str, Chem.Mol]:
    capZ = {"H": 1, "C": 6, "N": 7, "O": 8}[cap]

    M_heavy = Chem.MolFromSmiles(smiles)
    if M_heavy is None:
        raise ValueError("Bad SMILES.")
    Chem.SanitizeMol(M_heavy)

    b = M_heavy.GetBondWithIdx(bond_idx)
    if not b.IsInRing():
        raise ValueError("bond_idx must be a ring bond in the macrocycle.")
    a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()

    M_map = with_atom_maps(M_heavy)
    map1 = M_map.GetAtomWithIdx(a1).GetAtomMapNum()
    map2 = M_map.GetAtomWithIdx(a2).GetAtomMapNum()

    M_open_heavy = break_bond_and_cap(M_heavy, bond_idx=bond_idx, cap_atomic_num=capZ)

    atoms_near = bfs_atoms_within_radius(M_map, [a1, a2], radius=ref_radius)
    atoms_near.add(a1); atoms_near.add(a2)
    X_map = induced_submol(M_map, atoms_near)

    bx = find_bond_by_maps(X_map, map1, map2)
    X_open_map = break_bond_and_cap(X_map, bond_idx=bx, cap_atomic_num=capZ)

    M = Chem.AddHs(M_heavy)
    M_open = Chem.AddHs(M_open_heavy)
    X = Chem.AddHs(X_map)
    X_open = Chem.AddHs(X_open_map)

    return {
        "M": M,
        "M_open": M_open,
        "X": X,
        "X_open": X_open,
        "cut_atoms_parent": (a1, a2),
        "cut_atom_maps": (map1, map2),
    }


# ----------------------------
# Conformers + XYZ saving
# ----------------------------

@dataclass
class ConfScore:
    conf_id: int
    mmff_energy: float

def embed_mmff_rank(mol: Chem.Mol,
                    num_confs: int = 60,
                    prune_rms: float = 0.25,
                    max_iters: int = 400,
                    seed: int = 0xC0FFEE) -> Tuple[Chem.Mol, List[ConfScore]]:
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    conf_ids = list(AllChem.EmbedMultipleConfs(m, numConfs=num_confs, params=params))
    if not conf_ids:
        raise RuntimeError("ETKDG failed to embed conformers.")
    props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
    if props is None:
        raise RuntimeError("MMFF typing failed (unsupported chemistry).")
    ranked: List[ConfScore] = []
    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=cid)
        if ff is None:
            continue
        ff.Minimize(maxIts=max_iters)
        ranked.append(ConfScore(cid, float(ff.CalcEnergy())))
    ranked.sort(key=lambda r: r.mmff_energy)
    if not ranked:
        raise RuntimeError("No MMFF energies computed.")
    return m, ranked

def choose_top_m(ranked: List[ConfScore], top_m: int = 3, window: float = 5.0) -> List[int]:
    e0 = ranked[0].mmff_energy
    keep = [r.conf_id for r in ranked if (r.mmff_energy - e0) <= window]
    return keep[:top_m]

def write_xyz_multiframe(mol3d: Chem.Mol, conf_ids: List[int], path: str, comment_prefix: str):
    syms = [a.GetSymbol() for a in mol3d.GetAtoms()]
    with open(path, "w") as f:
        for cid in conf_ids:
            conf = mol3d.GetConformer(cid)
            f.write(f"{mol3d.GetNumAtoms()}\n")
            f.write(f"{comment_prefix} conf={cid}\n")
            for i, s in enumerate(syms):
                p = conf.GetAtomPosition(i)
                f.write(f"{s:2s} {p.x: .10f} {p.y: .10f} {p.z: .10f}\n")


# ----------------------------
# Plotting: RDKit connectivity -> NetworkX
# ----------------------------

def rdkit_to_nx(mol: Chem.Mol) -> nx.Graph:
    G = nx.Graph()
    for a in mol.GetAtoms():
        G.add_node(a.GetIdx(), element=a.GetSymbol(), map=a.GetAtomMapNum())
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        G.add_edge(i, j, order=str(b.GetBondType()))
    return G

def nx_pos_from_rdkit_2d(mol: Chem.Mol):
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    conf = m.GetConformer()
    return {i: (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(m.GetNumAtoms())}

def plot_species_grid(species: Dict[str, Chem.Mol], title: str = ""):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    for ax, name in zip(axes, ["M", "M_open", "X", "X_open"]):
        mol = species[name]
        G = rdkit_to_nx(mol)
        pos = nx_pos_from_rdkit_2d(mol)
        labels = {i: G.nodes[i]["element"] for i in G.nodes}
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.2, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=220)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        ax.set_title(f"{name}  |  formula={mol_formula(mol)}")
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
