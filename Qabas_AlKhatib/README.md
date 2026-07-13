# ML Approximation of Hessian Components for CASSCF
**CyberTraining 2026 Capstone Project**

**Author:** Qabas Mohammad Al Khatib  
**University:** Case Western Reserve University, Department of Chemistry  
**Advisor:** Dr. Shane M. Parker  

---

## What is this project about?

My PhD dissertation focuses on making the Resonating Hartree-Fock (ResHF) method 
converge reliably. One of the biggest bottlenecks is computing an expensive Hessian 
matrix at every optimization step. This project asks: can we use machine learning 
to predict this Hessian from cheap molecular features, instead of computing it 
explicitly every time?

As a first step, I used CASSCF as a starting point (since ResHF is still being 
implemented), trained a neural network on 16 INVEST molecules, and got 3% prediction 
error on molecules the model had never seen before.

---

## What did I actually do?

**Step 1 — Generate data**  
I ran CASSCF(6,6)/def2-SVP calculations on 23 INVEST molecules from Pollice et al. 
2021 using PySCF on the UB CCR HPC cluster. 16 out of 23 molecules converged.

**Step 2 — Extract features**  
Each molecule has variable-sized arrays (different number of orbitals). I summarized 
them into 12 fixed-size features: gradient norms, energy, CI coefficient statistics, 
and Hessian diagonal statistics.

**Step 3 — Train a neural network**  
I used PyTorch to train a feedforward network (12 → 64 → 64 → 400) to predict the 
CI Hessian diagonal from the 12 features. Trained on 14 molecules, tested on 2.

**Step 4 — Results**  
3.00% relative error on the 2 held-out test molecules. The model generalizes!

---

## Results

| | Value |
|--|-------|
| Training molecules | 14 |
| Test molecules | 2 |
| Relative error (S1000) | 2.34% |
| Relative error (S1010) | 3.69% |
| Overall error | **3.00%** |

---

## Honest limitations

- I only predicted the CI Hessian diagonal — not the full coupled orbital-CI Hessian
- CASSCF is not the same as ResHF (orthogonal vs nonorthogonal orbitals)
- Only 16 molecules — this is a proof of concept, not a production model

---

## What comes next?

Once the ResHF optimizer is implemented in my dissertation, I will retrain this 
pipeline on ResHF-specific Hessian data, extending to the full coupled Hessian 
including the orbital-CI cross terms.

---

## How to reproduce

```bash
git clone https://github.com/Qabas96/ml-hessian-casscf.git
cd ml-hessian-casscf
pip install pyscf torch numpy matplotlib scikit-learn
sbatch scripts/submit.slurm  # generates CASSCF data on HPC
jupyter notebook notebooks/02_feature_engineering.ipynb
```

---

## Tools used

PySCF 2.13.1, PyTorch 2.8.0, ChemML 1.3, UB CCR HPC (Slurm)

## Reference

Pollice et al., Matter 4, 1654–1682 (2021)
