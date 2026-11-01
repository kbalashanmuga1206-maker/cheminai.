"""
MolLab AI - SMILES molecule explorer (Streamlit + RDKit + py3Dmol)

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors

try:
    import py3Dmol
except ImportError:  # handled in UI
    py3Dmol = None

st.set_page_config(page_title="MolLab AI", page_icon="🧪", layout="wide")

EXAMPLES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "Glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
}

STYLES = {
    "Stick": {"stick": {"radius": 0.15}},
    "Ball and Stick": {"stick": {"radius": 0.12}, "sphere": {"scale": 0.28}},
    "Sphere (Space-filling)": {"sphere": {}},
    "Line (Wireframe)": {"line": {}},
}


# ----------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def parse_smiles(smiles: str):
    """Return (canonical_smiles, error)."""
    smiles = (smiles or "").strip()
    if not smiles:
        return None, "Please enter a SMILES string."
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES: RDKit could not parse this structure."
    return Chem.MolToSmiles(mol), None


def get_mol(smiles):
    return Chem.MolFromSmiles(smiles)


def draw_2d(mol, size=(450, 350)):
    return Draw.MolToImage(mol, size=size)


@st.cache_data(show_spinner=False)
def build_3d_molblock(smiles: str, seed: int = 42):
    """Embed + optimise a 3D conformer; returns (molblock, method, error)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    status = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
        if status != 0:
            return None, None, "3D embedding failed for this molecule."
    method = "none"
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            method = "MMFF94"
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
            method = "UFF"
    except Exception:
        pass
    return Chem.MolToMolBlock(mol), method, None


def render_3d(molblock, style_name, bg="white", spin=False, labels=False, height=450):
    if py3Dmol is None:
        st.error("py3Dmol is not installed. Run: pip install py3Dmol")
        return
    view = py3Dmol.view(width=600, height=height)
    view.addModel(molblock, "mol")
    view.setStyle(STYLES[style_name])
    if labels:
        view.addPropertyLabels("elem", "", {"fontSize": 10, "showBackground": False})
    view.setBackgroundColor(bg)
    if spin:
        view.spin(True)
    view.zoomTo()
    components.html(view._make_html(), height=height + 10)


def compute_properties(mol):
    props = {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact Mass": round(Descriptors.ExactMolWt(mol), 4),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Total Atoms (with H)": Chem.AddHs(mol).GetNumAtoms(),
        "LogP (Crippen)": round(Crippen.MolLogP(mol), 3),
        "Molar Refractivity": round(Crippen.MolMR(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "Formal Charge": Chem.GetFormalCharge(mol),
        "QED (drug-likeness)": round(QED.qed(mol), 3),
    }
    # Lipinski Rule of Five
    violations = sum(
        [
            Descriptors.MolWt(mol) > 500,
            Crippen.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        ]
    )
    props["Lipinski Violations"] = violations
    props["Lipinski Rule of 5"] = "Pass" if violations <= 1 else "Fail"
    return props


def build_report(name, smiles, canonical, props):
    lines = [
        "MOLLAB AI - MOLECULAR REPORT",
        "=" * 40,
        f"Generated : {datetime.now():%Y-%m-%d %H:%M}",
        f"Name      : {name or 'N/A'}",
        f"Input     : {smiles}",
        f"Canonical : {canonical}",
        "",
        "PROPERTIES",
        "-" * 40,
    ]
    lines += [f"{k:<28}{v}" for k, v in props.items()]
    return "\n".join(lines)


def mol_to_png_bytes(mol, size=(500, 400)):
    buf = io.BytesIO()
    draw_2d(mol, size).save(buf, format="PNG")
    return buf.getvalue()


def parse_multi(text):
    """Lines: 'SMILES' or 'name,SMILES' or 'name SMILES'."""
    items = []
    for i, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if "," in line:
            name, smi = [x.strip() for x in line.split(",", 1)]
        elif " " in line:
            name, smi = [x.strip() for x in line.split(None, 1)]
        else:
            name, smi = f"Molecule {i}", line
        items.append((name, smi))
    return items


def ai_explain(api_key, name, smiles, props):
    try:
        import anthropic
    except ImportError:
        return "Install the SDK first: pip install anthropic"
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are a medicinal chemistry assistant. Given the molecule below, give a concise "
        "assessment (under 250 words): likely functional groups, drug-likeness, solubility / "
        "permeability expectations, and any structural alerts. State clearly that this is a "
        "computational estimate, not experimental data.\n\n"
        f"Name: {name or 'unknown'}\nSMILES: {smiles}\nProperties: {json.dumps(props)}"
    )
    try:
        msg = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
    except Exception as e:
        return f"AI request failed: {e}"


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🧪 MolLab AI")
viz_style = st.sidebar.selectbox("3D style", list(STYLES.keys()))
bg_color = st.sidebar.selectbox("Background", ["white", "black", "lightgrey"])
spin = st.sidebar.checkbox("Auto-rotate", value=False)
atom_labels = st.sidebar.checkbox("Show atom labels", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("AI assistant (optional)")
api_key = st.sidebar.text_input(
    "Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", "")
)

st.title("MolLab AI – Molecular Structure Explorer")
tab_single, tab_compare, tab_about = st.tabs(["Single molecule", "Compare molecules", "About"])

# ----------------------------------------------------------------- single
with tab_single:
    c1, c2 = st.columns([3, 1])
    with c2:
        example = st.selectbox("Load example", ["(none)"] + list(EXAMPLES.keys()))
    default_smiles = EXAMPLES.get(example, "CC(=O)OC1=CC=CC=C1C(=O)O")
    with c1:
        smiles_in = st.text_input("Enter SMILES", value=default_smiles)
    mol_name = st.text_input("Molecule name (optional)", value="" if example == "(none)" else example)

    canonical, err = parse_smiles(smiles_in)
    if err:
        st.error(f"❌ {err}")
    else:
        mol = get_mol(canonical)
        st.success(f"✅ Valid structure — canonical SMILES: `{canonical}`")

        col2d, col3d = st.columns(2)
        with col2d:
            st.subheader("2D structure")
            st.image(draw_2d(mol), use_container_width=False)
        with col3d:
            st.subheader("Interactive 3D structure")
            molblock, method, err3d = build_3d_molblock(canonical)
            if err3d:
                st.warning(err3d)
            else:
                render_3d(molblock, viz_style, bg_color, spin, atom_labels)
                st.caption(f"Geometry optimised with {method}. Drag to rotate, scroll to zoom.")

        st.subheader("Molecular properties")
        props = compute_properties(mol)
        df = pd.DataFrame(props.items(), columns=["Property", "Value"]).astype({"Value": str})
        st.dataframe(df, use_container_width=True, hide_index=True)

        if props["Lipinski Rule of 5"] == "Pass":
            st.info("Lipinski's Rule of Five: no more than one violation (drug-like by this heuristic).")
        else:
            st.warning("Lipinski's Rule of Five: two or more violations.")

        st.subheader("Download")
        report = build_report(mol_name, smiles_in, canonical, props)
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("📄 Report (.txt)", report, "mollab_report.txt", "text/plain")
        d2.download_button("📊 Properties (.csv)", df.to_csv(index=False), "properties.csv", "text/csv")
        d3.download_button("🖼 2D image (.png)", mol_to_png_bytes(mol), "structure.png", "image/png")
        if not err3d:
            d4.download_button("🧬 3D (.mol)", molblock, "structure_3d.mol", "chemical/x-mdl-molfile")

        st.subheader("AI assessment")
        if st.button("Explain this molecule with AI"):
            if not api_key:
                st.warning("Add an Anthropic API key in the sidebar to use this feature.")
            else:
                with st.spinner("Thinking..."):
                    st.markdown(ai_explain(api_key, mol_name, canonical, props))

# ----------------------------------------------------------------- compare
with tab_compare:
    st.write("Enter one molecule per line as `name,SMILES` (or just SMILES).")
    multi_text = st.text_area(
        "Molecules",
        value="Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O\nIbuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O\nParacetamol,CC(=O)NC1=CC=C(O)C=C1",
        height=150,
    )
    if st.button("Compare", type="primary"):
        valid, invalid = [], []
        for name, smi in parse_multi(multi_text):
            can, e = parse_smiles(smi)
            (invalid if e else valid).append((name, smi) if e else (name, can))
        for name, smi in invalid:
            st.error(f"Invalid SMILES for **{name}**: `{smi}`")

        if len(valid) < 2:
            st.warning("Enter at least two valid molecules to compare.")
        else:
            mols = [get_mol(s) for _, s in valid]
            names = [n for n, _ in valid]

            st.subheader("Structures")
            grid = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 250), legends=names)
            st.image(grid)

            st.subheader("Property comparison")
            table = pd.DataFrame({n: compute_properties(m) for n, m in zip(names, mols)})
            st.dataframe(table.astype(str), use_container_width=True)
            st.download_button("⬇ Comparison (.csv)", table.to_csv(), "comparison.csv", "text/csv")

            st.subheader("Tanimoto similarity (Morgan, r=2)")
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
            sim = [[DataStructs.TanimotoSimilarity(a, b) for b in fps] for a in fps]
            sim_df = pd.DataFrame(sim, index=names, columns=names).round(3)
            st.dataframe(sim_df.style.background_gradient(cmap="Blues", vmin=0, vmax=1),
                         use_container_width=True)

            st.subheader("3D side by side")
            cols = st.columns(min(len(valid), 3))
            for i, (n, s) in enumerate(valid):
                mb, _, e3 = build_3d_molblock(s)
                with cols[i % len(cols)]:
                    st.markdown(f"**{n}**")
                    if e3:
                        st.warning(e3)
                    else:
                        render_3d(mb, viz_style, bg_color, spin, False, height=300)

# ----------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
**MolLab AI** features
- SMILES input with validation and canonicalisation
- 2D depiction (RDKit) and interactive 3D (py3Dmol, MMFF94/UFF optimised)
- Four 3D styles: stick, ball-and-stick, space-filling, wireframe
- Properties: formula, MW, LogP, TPSA, HBD/HBA, rotatable bonds, rings, QED, Lipinski Ro5
- Reports: TXT, CSV, PNG, MOL downloads
- Multi-molecule comparison with similarity matrix
- Optional AI explanation via the Anthropic API

*All values are computed estimates, not experimental measurements.*
"""
    )
