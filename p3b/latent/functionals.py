"""E^, N^, S^, the port fluxes and their latent gradients for a latent state (formulation 5.2).

E^(z) = E(D(z)) and likewise N^ and S^: the physical functionals of the decoded, unstandardised
fields. Their latent gradients are vector-Jacobian products through the decoder. Shared by
U+F (inputs), P (penalties) and the structured arms. System B has no water terms.
"""
from __future__ import annotations

import torch


class LatentFunctionals:
    def __init__(self, decode, std, phys, system: str, n_cells: int):
        self.decode, self.std, self.phys, self.system, self.n_cells = decode, std, phys, system, n_cells
        self.n_ch = 3 if system == "A" else 2
        owner = getattr(decode, "__self__", None)          # the frozen autoencoder (PREREG 3)
        if isinstance(owner, torch.nn.Module):
            owner.requires_grad_(False)

    def split(self, X):
        k = self.n_cells * self.n_ch
        return X[..., :k].reshape(*X.shape[:-1], self.n_cells, self.n_ch), X[..., k:]

    def _values(self, z, raw):
        ch, out = self.split(self.std.decode(self.decode(z)))
        v = {"E": self.phys.E(ch), "S": self.phys.S(ch)}
        if self.system == "A":
            v["N"] = self.phys.N(ch)
            v["P_N"], v["P_E"], v["Phi_S"] = self.phys.ports(ch, out, raw)
        else:
            v["P_E"], v["Phi_S"] = self.phys.ports(ch, out, raw)
        return v

    def evaluate(self, z, raw, grads: bool = True, create_graph: bool | None = None) -> dict:
        """Values (batch,) and, if `grads`, the latent gradients dE, dN, dS (batch, d). The results
        stay in the graph when gradients are enabled (create_graph=None follows torch.is_grad_enabled()),
        so U+F's inputs, P's penalties and S's terms keep their dependence on z in training."""
        if create_graph is None:
            create_graph = torch.is_grad_enabled()
        if not grads:
            v = self._values(z, raw)
            return v if create_graph else {k: t.detach() for k, t in v.items()}
        with torch.enable_grad():
            zz = z if z.requires_grad else z.detach().requires_grad_(True)
            v = self._values(zz, raw)
            names = ["E", "S"] + (["N"] if self.system == "A" else [])
            for name in names:
                v["d" + name] = torch.autograd.grad(v[name].sum(), zz, create_graph=create_graph,
                                                    retain_graph=True)[0]
        if not create_graph:
            v = {k: t.detach() for k, t in v.items()}
        return v
