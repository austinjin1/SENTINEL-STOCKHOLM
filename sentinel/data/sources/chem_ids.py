# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Chemical identity: CAS -> DTXSID -> structure (SMILES / InChIKey / MW) with structure-class flags.

Sources, in order of preference:
  1. EPA CompTox Chemicals Dashboard backend search (keyless):
     ``https://comptox.epa.gov/dashboard-api/ccdapp1/search/chemical/equal-with-detail/<CAS>``
     — curated DSSTox record: DTXSID, preferred name, SMILES, InChIKey, formula, average mass,
     multicomponent flag, QC level. (The documented CTX API needs a key and is not used.)
  2. PubChem PUG-REST name search for CAS numbers CompTox does not resolve:
     ``https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/<CAS>/property/.../JSON``.
  3. Identifiers already carried by ECOTOX (``chemicals.txt``: CAS, name, DTXSID) and ADORE
     (``chem_dtxsid``, ``chem_inchikey``, ``chem_*_can_smiles``) are kept alongside for QC.
Both services are polled at <= 5 requests/s (the PubChem usage policy limit) with backoff on
HTTP 429/503. Requests contain only the CAS number.

Structure processing uses RDKit: canonical SMILES (RDKit flavour, isomeric), InChIKey and MolWt
are recomputed from the source SMILES so every table uses one SMILES dialect.
``structure_class`` is one of ``organic``, ``organic_multicomponent`` (salts, hydrates, ion pairs
containing carbon), ``organometallic``, ``inorganic`` (no carbon, or only inorganic carbon such
as carbonate/cyanide/CO/CO2), ``no_structure`` (record exists without a structure: mixtures,
UVCBs, polymers, petroleum products, e.g. diesel fuel) and ``unresolved`` (no record found).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

COMPTOX_SEARCH_URL = "https://comptox.epa.gov/dashboard-api/ccdapp1/search/chemical/equal-with-detail/"
PUBCHEM_PROPS = "SMILES,ConnectivitySMILES,InChIKey,MolecularFormula,MolecularWeight,IUPACName"
PUBCHEM_NAME_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property/" + PUBCHEM_PROPS + "/JSON"
)
MAX_REQ_PER_S = 5.0

_CAS_RE = re.compile(r"^0*(\d{2,7})-(\d{2})-(\d)$")
_DIGITS_RE = re.compile(r"^\d{5,10}$")

# --------------------------------------------------------------------------- CAS numbers


def cas_check_digit_ok(cas: str) -> bool:
    """CAS check digit: sum of digits (right to left, weights 1, 2, 3, ...) mod 10."""
    m = _CAS_RE.match(cas)
    if not m:
        return False
    body = m.group(1) + m.group(2)
    total = sum(int(d) * w for w, d in enumerate(reversed(body), start=1))
    return total % 10 == int(m.group(3))


def normalize_cas(value: Any, *, validate: bool = True) -> str | None:
    """Return ``NNNNNNN-NN-N`` (no leading zeros) or None.

    Accepts hyphenated (``000050-00-0``, ``50-00-0``) and ECOTOX digit-only (``50000``) forms.
    Placeholders (all zeros) and, when ``validate``, numbers with a wrong check digit -> None.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip().replace(" ", "")
    if s.endswith(".0"):
        s = s[:-2]
    if not s:
        return None
    if _DIGITS_RE.match(s):
        s = s.lstrip("0")
        if len(s) < 5:
            return None
        s = f"{s[:-3]}-{s[-3:-1]}-{s[-1]}"
    m = _CAS_RE.match(s)
    if not m:
        return None
    first = m.group(1).lstrip("0")
    if not first or len(first) < 2:
        return None
    cas = f"{first}-{m.group(2)}-{m.group(3)}"
    if int(first) == 0:
        return None
    if validate and not cas_check_digit_ok(cas):
        return None
    return cas


# --------------------------------------------------------------------------- service parsing


def comptox_url(cas: str) -> str:
    return COMPTOX_SEARCH_URL + quote(cas, safe="")


def parse_comptox_hits(hits: list[dict] | None, cas: str) -> dict:
    """Pick the record whose *current* CASRN equals the query (``searchMatch == 'CASRN'``).

    Falls back to a single non-CASRN match (e.g. 'Deleted CASRN', 'Alternate CASRN') with
    ``comptox_match`` recording which kind it was; ambiguous fallbacks are left unresolved.
    """
    out = {
        "comptox_n_hits": 0,
        "comptox_match": None,
        "dtxsid": None,
        "comptox_casrn": None,
        "comptox_name": None,
        "comptox_smiles": None,
        "comptox_inchikey": None,
        "comptox_formula": None,
        "comptox_avg_mass": None,
        "comptox_multicomponent": None,
        "comptox_qc_level": None,
    }
    if not hits:
        return out
    out["comptox_n_hits"] = len(hits)
    exact = [h for h in hits if h.get("searchMatch") == "CASRN" and h.get("casrn") == cas]
    pick = exact[0] if len(exact) == 1 else (hits[0] if len(hits) == 1 else None)
    if pick is None:
        out["comptox_match"] = "ambiguous"
        return out
    out.update(
        {
            "comptox_match": pick.get("searchMatch"),
            "dtxsid": pick.get("dtxsid"),
            "comptox_casrn": pick.get("casrn"),
            "comptox_name": pick.get("preferredName"),
            "comptox_smiles": pick.get("smiles") or None,
            "comptox_inchikey": pick.get("inchiKey") or None,
            "comptox_formula": pick.get("molFormula") or None,
            "comptox_avg_mass": pick.get("averageMass"),
            "comptox_multicomponent": pick.get("multicomponent"),
            "comptox_qc_level": pick.get("qcLevel"),
        }
    )
    return out


def pubchem_url(cas: str) -> str:
    return PUBCHEM_NAME_URL.format(q=quote(cas, safe=""))


def parse_pubchem(payload: dict | None) -> dict:
    """First PubChem record of a property table (CIDs are ranked by PubChem); count kept."""
    out = {
        "pubchem_n_cids": 0,
        "pubchem_cid": None,
        "pubchem_smiles": None,
        "pubchem_inchikey": None,
        "pubchem_formula": None,
        "pubchem_mw": None,
        "pubchem_iupac": None,
    }
    props = ((payload or {}).get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return out
    p = props[0]
    out.update(
        {
            "pubchem_n_cids": len(props),
            "pubchem_cid": str(p.get("CID")),
            "pubchem_smiles": p.get("SMILES")
            or p.get("IsomericSMILES")
            or p.get("ConnectivitySMILES")
            or p.get("CanonicalSMILES"),
            "pubchem_inchikey": p.get("InChIKey"),
            "pubchem_formula": p.get("MolecularFormula"),
            "pubchem_mw": float(p["MolecularWeight"]) if p.get("MolecularWeight") else None,
            "pubchem_iupac": p.get("IUPACName"),
        }
    )
    return out


# --------------------------------------------------------------------------- structures

_METALS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Ge",
    "Sb",
    "As",
    "Te",
    "Se",
    "B",
    "Si",
}
_TRUE_METALS = _METALS - {"B", "Si", "Ge", "As", "Sb", "Te", "Se"}
# carbon-containing fragments conventionally treated as inorganic
# (checked only on fragments with exactly one carbon atom; "[C;X0]" = an isolated carbon atom)
_INORGANIC_C_SMARTS = (
    "[CX3](=O)([O-,OH])[O-,OH]",
    "[C-]#N",
    "[CH1]#N",
    "[C;X1]#[O+,O]",
    "O=C=O",
    "[C-]#[O+]",
    "N=C=S",
    "[S-]C#N",
    "[C;X0]",
    "S=C=S",
    "[O-]C#N",
    "OC#N",
)


def rdkit_identity(smiles: str | None) -> dict:
    """RDKit canonical SMILES, InChIKey, MolWt, component count and structure class for a SMILES."""
    out = {
        "smiles_canonical": None,
        "inchikey_rdkit": None,
        "mw_rdkit": None,
        "n_components": None,
        "n_carbon": None,
        "has_metal": None,
        "structure_class": None,
        "rdkit_parse_ok": False,
    }
    if not smiles:
        return out
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return out
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:  # noqa: BLE001
            return out
    out["rdkit_parse_ok"] = True
    out["smiles_canonical"] = Chem.MolToSmiles(mol)
    try:
        out["inchikey_rdkit"] = Chem.MolToInchiKey(mol) or None
    except Exception:  # noqa: BLE001
        out["inchikey_rdkit"] = None
    try:
        out["mw_rdkit"] = float(Descriptors.MolWt(mol))
    except Exception:  # noqa: BLE001
        out["mw_rdkit"] = None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    out["n_components"] = len(frags)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    n_c = sum(1 for s in syms if s == "C")
    out["n_carbon"] = n_c
    out["has_metal"] = any(s in _TRUE_METALS for s in syms)
    out["structure_class"] = classify_structure(mol, frags, n_c, out["has_metal"])
    return out


def classify_structure(mol, frags, n_carbon: int, has_metal: bool) -> str:
    from rdkit import Chem

    if n_carbon == 0:
        return "inorganic"
    organic_frags = 0
    for f in frags:
        fc = sum(1 for a in f.GetAtoms() if a.GetSymbol() == "C")
        if fc == 0:
            continue
        inorganic_c = False
        if fc == 1:
            for sm in _INORGANIC_C_SMARTS:
                patt = Chem.MolFromSmarts(sm)
                if patt is not None and f.HasSubstructMatch(patt):
                    inorganic_c = True
                    break
        if not inorganic_c:
            organic_frags += 1
    if organic_frags == 0:
        return "inorganic"
    metal_c_bond = any(
        (b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() in _TRUE_METALS)
        or (b.GetEndAtom().GetSymbol() == "C" and b.GetBeginAtom().GetSymbol() in _TRUE_METALS)
        for b in mol.GetBonds()
    )
    if metal_c_bond:
        return "organometallic"
    if len(frags) > 1:
        return "organic_multicomponent"
    return "organic"


#: name keywords that explain why a record has no single structure (heuristic, for triage only)
_NO_STRUCTURE_HINTS = (
    (
        "petroleum_or_uvcb",
        re.compile(
            r"\b(oils?|petroleum|distillates?|naphtha|kerosene|gasoline|diesel|fuels?|"
            r"tars?|asphalt|bitumen|waxe?s?|paraffins?|creosote|residues?|residuum|"
            r"hydrocarbons?|extracts?|reaction mass|c\d+\s*-\s*c?\d+)\b",
            re.I,
        ),
    ),
    (
        "polymer",
        re.compile(
            r"(polymer|poly\(|resin|polyethylene|polyvinyl|polyoxy|ethoxylat|propoxylat|cellulose)", re.I
        ),
    ),
    (
        "mixture_or_uvcb",
        re.compile(
            r"(mixture|\bmixt\.|formulation|technical|commercial|blend|aroclor|surfactant|detergent|"
            r"emulsifiable|quaternary ammonium|\btallow|\bcoco\b|\balkyls?\b|\bcompounds\b|\bacids, )",
            re.I,
        ),
    ),
    ("variable_stoichiometry", re.compile(r"\(\s*(\d+|[nx?])\s*:\s*(\d+|[nx?])\s*\)|unspecified", re.I)),
    (
        "biological",
        re.compile(
            r"(bacillus|virus|toxin|venom|enzyme|protein|bacteri|fung|hormone|lectin|agglutinin)", re.I
        ),
    ),
)


def no_structure_hint(name: str | None, smiles: str | None = None) -> str | None:
    """Triage label for a record without a usable single structure (keyword heuristic)."""
    if smiles and "*" in smiles:
        return "polymer" if "Sg:n" in smiles else "isomer_or_markush"
    if not name:
        return None
    for label, rx in _NO_STRUCTURE_HINTS:
        if rx.search(str(name)):
            return label
    return "unspecified"


def resolve_identity(cas: str, comptox: dict, pubchem: dict | None, seeds: dict | None = None) -> dict:
    """Combine service records into one identity row (CompTox first, PubChem fallback)."""
    seeds = seeds or {}
    src_smiles, src = None, None
    if comptox.get("comptox_smiles"):
        src_smiles, src = comptox["comptox_smiles"], "comptox"
    elif pubchem and pubchem.get("pubchem_smiles"):
        src_smiles, src = pubchem["pubchem_smiles"], "pubchem"
    elif seeds.get("adore_smiles"):
        src_smiles, src = seeds["adore_smiles"], "adore"
    rd = rdkit_identity(src_smiles)
    row = {
        "cas": cas,
        **seeds,
        **comptox,
        **(pubchem or {}),
        **rd,
        "structure_source": src,
        "smiles_source": src_smiles,
    }
    if src_smiles is None:
        found = bool(comptox.get("dtxsid")) or bool(pubchem and pubchem.get("pubchem_cid"))
        row["structure_class"] = "no_structure" if found else "unresolved"
    row["dtxsid_final"] = comptox.get("dtxsid") or seeds.get("ecotox_dtxsid") or seeds.get("adore_dtxsid")
    row["dtxsid_conflict"] = bool(
        comptox.get("dtxsid")
        and seeds.get("ecotox_dtxsid")
        and comptox.get("dtxsid") != seeds.get("ecotox_dtxsid")
    )
    row["name"] = (
        comptox.get("comptox_name")
        or seeds.get("ecotox_name")
        or seeds.get("adore_name")
        or (pubchem or {}).get("pubchem_iupac")
    )
    mw = rd.get("mw_rdkit")
    if mw is None:
        mw = comptox.get("comptox_avg_mass") or (pubchem or {}).get("pubchem_mw")
    row["mw_g_mol"] = float(mw) if mw else None
    row["inchikey"] = (
        rd.get("inchikey_rdkit") or comptox.get("comptox_inchikey") or (pubchem or {}).get("pubchem_inchikey")
    )
    if src_smiles is not None and "*" in src_smiles:
        # CompTox CXSMILES with attachment points / R-groups ('Cl*.Clc1ccccc1 |m:...|', '[*]CC[*] |Sg:n|'):
        # positional-isomer mixtures, Markush classes and polymers have no single structure
        row["structure_class"] = "no_structure"
        row["mw_g_mol"] = comptox.get("comptox_avg_mass") or None
        row["smiles_canonical"] = None
        row["inchikey"] = comptox.get("comptox_inchikey") or None
    row["no_structure_hint"] = (
        no_structure_hint(row["name"], src_smiles)
        if row["structure_class"] in ("no_structure", "unresolved")
        else None
    )
    return row


# --------------------------------------------------------------------------- JSONL caches


class JsonlCache:
    """Append-only {key: payload} cache so long lookups are resumable and auditable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:  # a write interrupted mid-line: refetch that key
                            continue
                        self.data[rec["key"]] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, status: int, payload: Any, retrieved_utc: str) -> None:
        rec = {"key": key, "status": status, "payload": payload, "retrieved_utc": retrieved_utc}
        self.data[key] = rec
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def identity_frame(rows: list[dict]) -> pd.DataFrame:
    cols_first = [
        "cas",
        "dtxsid_final",
        "name",
        "smiles_canonical",
        "inchikey",
        "mw_g_mol",
        "structure_class",
        "structure_source",
        "n_components",
        "has_metal",
        "dtxsid_conflict",
    ]
    df = pd.DataFrame(rows)
    rest = [c for c in df.columns if c not in cols_first]
    return df[[c for c in cols_first if c in df.columns] + rest]
