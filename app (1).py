
import streamlit as st
import numpy as np
import pandas as pd
from Bio import AlignIO
from io import StringIO
import matplotlib.pyplot as plt
import subprocess
import os
import re
import shutil

# ======================================================
# APP BRANDING
# ======================================================

st.set_page_config(page_title="BioMotifAI", layout="wide")

st.title("🧬 BioMotifAI – Motif Training & Annotation Platform")

st.markdown("""
### Founder: **Mohan K**  
**PhD Research Scholar – Vellore Institute of Technology (VIT)**  

Integrated DNA & Protein Motif Trainer + Knowledgebase + Pfam-style HMM
""")

# ======================================================
# MOTIF KNOWLEDGEBASE
# ======================================================

motif_db = [

("HExxH","CATGAA..CAT","Metalloprotease","Enzyme"),
("GxGxxG","GGAGGT","NAD/FAD Binding","Metabolism"),
("HGGXH","CATGGAGGC","Dehydrogenase","Metabolism"),
("DXH","GAT..CAT","Asp Protease","Enzyme"),
("GGXGG","GGAGGT","Dehydrogenase","Metabolism"),

("DFG","GATTTTGGT","Protein Kinase Loop","Signal"),
("HRD","CATCGTGAT","Kinase Catalytic","Signal"),
("GxGKT","GGAGGTAAGACT","P-loop NTP","ATP Binding"),
("NKXD","AACAAAGAT","GTP Binding","Signal"),

("HTH","CATACGCAT","Helix-Turn-Helix","DNA Binding"),
("C2H2","TGT..CAT..CAT","Zinc Finger","DNA Binding"),

("TMHELIX","ATGTTGTTG","Transmembrane","Membrane"),
("SIGNALP","ATGAAA","Signal Peptide","Secretion"),

("RdRp","GATGATGGT","RNA Polymerase","Virus"),
("CAPSID","ATGCGTGCT","Capsid Protein","Virus"),

("NxS","AAC..TCT","N-glycosylation","PTM"),
("YXXP","TAT..CCG","Phosphorylation","PTM"),
("RGD","AGGGGTGAT","Cell Adhesion","Binding")
]

motif_df = pd.DataFrame(motif_db,
columns=["AA_Motif","DNA_Motif","Function","Category"])

# Prepare regex columns
motif_df["AA_REGEX"] = motif_df["AA_Motif"]\
    .str.replace("x",".", regex=False)\
    .str.replace("2","{2}", regex=False)

motif_df["DNA_REGEX"] = motif_df["DNA_Motif"]

# ======================================================
# FILE INPUT
# ======================================================

uploaded_file = st.file_uploader("Upload ALIGNED FASTA (DNA or Protein)", type=["fasta","fa","txt"])

seq_type = st.selectbox("Sequence Type", ["Protein","DNA"])

# ======================================================
# SHANNON ENTROPY (Gap Safe)
# ======================================================

def shannon_entropy(column):

    column = column[column != '-']

    if len(column) == 0:
        return 0

    values, counts = np.unique(column, return_counts=True)
    probs = counts / counts.sum()

    return -np.sum(probs * np.log2(probs))

# ======================================================
# PIPELINE
# ======================================================

if uploaded_file:

    fasta_data = uploaded_file.read().decode("utf-8")

    try:
        alignment = AlignIO.read(StringIO(fasta_data), "fasta")
    except:
        st.error("Invalid FASTA alignment format")
        st.stop()

    msa = np.array([list(rec.seq) for rec in alignment])

    n_seq, aln_len = msa.shape

    st.success(f"Alignment Loaded → {n_seq} sequences | Length {aln_len}")

    # ================= PWM =================

    alphabet = list("ACDEFGHIKLMNPQRSTVWY") if seq_type=="Protein" else list("ATGC")

    pwm = []

    for col in msa.T:

        valid_col = col[col != '-']

        freq = {}

        if len(valid_col) == 0:
            for sym in alphabet:
                freq[sym] = 0
        else:
            for sym in alphabet:
                freq[sym] = np.sum(valid_col == sym) / len(valid_col)

        pwm.append(freq)

    pwm_df = pd.DataFrame(pwm)

    st.subheader("📊 PWM Matrix")
    st.dataframe(pwm_df)

    # ================= ENTROPY =================

    entropy_scores = [shannon_entropy(col) for col in msa.T]

    entropy_df = pd.DataFrame({
        "Position": np.arange(1, aln_len+1),
        "Entropy": entropy_scores
    })

    st.subheader("📈 Conservation Profile")
    st.dataframe(entropy_df)

    # ================= CONSENSUS =================

    consensus = ""

    for i in range(len(pwm_df)):
        row = pwm_df.iloc[i]
        consensus += row.idxmax() if row.sum()>0 else "X"

    st.subheader("🧬 Consensus Sequence")
    st.code(consensus)

    # ================= MOTIF SCAN =================

    st.subheader("📚 Motif Annotation Results")

    hits = []

    for _,row in motif_df.iterrows():

        if seq_type == "Protein":
            match = re.search(row["AA_REGEX"], consensus)
        else:
            match = re.search(row["DNA_REGEX"], consensus)

        if match:
            hits.append((
                row["AA_Motif"],
                row["DNA_Motif"],
                row["Function"],
                row["Category"],
                match.group(),
                match.start()+1
            ))

    if hits:

        hit_df = pd.DataFrame(hits, columns=[
            "AA_Motif","DNA_Motif","Function",
            "Category","Matched","Position"
        ])

        st.dataframe(hit_df)

    else:
        st.warning("No motif matched")

    # ================= HMM PROFILE =================

    st.subheader("🧬 HMM Profile Training")

    with open("train.fasta","w") as f:
        f.write(fasta_data)

    if shutil.which("hmmbuild"):

        subprocess.call("hmmbuild trained_profile.hmm train.fasta", shell=True)

        if os.path.exists("trained_profile.hmm"):

            st.success("HMM Profile Generated")

            with open("trained_profile.hmm","rb") as f:
                st.download_button("Download HMM Profile", f, "trained_profile.hmm")

    else:
        st.warning("HMMER not installed")

    # ================= DOWNLOAD =================

    st.subheader("⬇ Download Reports")

    st.download_button(
        "Download PWM CSV",
        pwm_df.to_csv(index=False).encode(),
        "trained_pwm.csv"
    )

    st.download_button(
        "Download Entropy CSV",
        entropy_df.to_csv(index=False).encode(),
        "entropy_profile.csv"
    )

    st.download_button(
        "Download Motif Database",
        motif_df.to_csv(index=False).encode(),
        "motif_reference_database.csv"
    )
