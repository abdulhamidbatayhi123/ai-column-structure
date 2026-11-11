# Decoder-gate diagnostics on the pilot data (pre-freeze evidence; no arm trained or evaluated)

Kaggle CPU kernels on the reference kit (kit commit in each JSON). Exploratory: they inform the
author's decision in docs/freeze_decisions_3B.md section 1b and are not part of the registered design.

| file | kernel | what |
|---|---|---|
| decoder_diag.json | decoder_diag_kernel.py | MLP and conv decoders, d 16/32/64, 60 and 300 epochs, all three keys |
| decoder_diag2.json | decoder_diag2_kernel.py | MOF-303 learning curve (8/16/32 trajectories), per-snapshot balance errors, MSE + balance-term loss (lambda 0.1, 1) |
| decoder_diag3.json | decoder_diag3_kernel.py | lambda 3/10/30, the lambda = 1 learning curve, d = 64 (pending) |
