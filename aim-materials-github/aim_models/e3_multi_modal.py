from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from composition_constraints import (
    composition_diversity_loss,
    noble_gas_penalty,
    radioactive_penalty,
    NOBLE_GASES,
    RADIOACTIVE,
    RADIOACTIVE_SEVERITY
)

from aim_models.masks_voigt import symmetry_mask_21, equality_penalties, unpack_voigt_21_sym, pack_voigt_6x6_sym
from utils.geometry import lat6_to_matrix, soft_min_distance

def structure_factor_magnitude(frac, lattice, qgrid):
    cart = frac @ lattice
    cart = torch.nan_to_num(cart)
    phase = (qgrid @ cart.T)
    Sreal = torch.cos(phase).sum(dim=-1)
    Simag = torch.sin(phase).sum(dim=-1)
    S = torch.sqrt(torch.clamp(Sreal**2 + Simag**2, min=0.0))
    return torch.nan_to_num(S)

def safe_min_distance(frac, lattice):
    if frac.numel() == 0 or frac.size(0) < 2:
        return torch.tensor(2.0, device=frac.device)
    cart = frac @ lattice
    cart = torch.nan_to_num(cart)
    diff = cart.unsqueeze(1) - cart.unsqueeze(0)
    d = torch.linalg.norm(diff + 1e-12, dim=-1) + torch.eye(frac.size(0), device=frac.device)*1e6
    d = torch.nan_to_num(d, nan=1e6, posinf=1e6, neginf=1e6)
    return d.min()

try:
    from egnn_pytorch import EGNN_Sparse
    HAS_EGNN = True
except Exception:
    EGNN_Sparse = None
    HAS_EGNN = False

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, hidden)
        self.to_shift = nn.Linear(cond_dim, hidden)

    def forward(self, h: torch.Tensor, cond: torch.Tensor, reps: Optional[List[int]] = None):
        if cond is None:
            return h
        if cond.dim() == 2 and reps is not None:
            cond = torch.repeat_interleave(cond, torch.tensor(reps, device=cond.device), dim=0)
        s = self.to_scale(cond); t = self.to_shift(cond)
        return h * (1 + s) + t

class EquivariantBackbone(nn.Module):
    def __init__(self, in_node: int, in_edge: int, hidden: int, n_layers: int,
                 use_egnn: bool = True, cond_dim: Optional[int]=None):
        super().__init__()
        # FIX: respect the flag
        self.use_egnn = bool(use_egnn) and HAS_EGNN

        self.node_embed = nn.Linear(in_node, hidden)
        self.edge_embed = nn.Linear(in_edge, hidden)
        self.layers = nn.ModuleList()
        self.cond = FiLM(cond_dim, hidden) if cond_dim is not None else None

        if self.use_egnn:
            for _ in range(n_layers):
                self.layers.append(
                    EGNN_Sparse(
                        feats_dim=hidden,
                        pos_dim=3,
                        edge_attr_dim=hidden,
                        m_dim=16,
                        fourier_features=0,
                        aggr="add",
                    )
                )

        else:
            for _ in range(n_layers):
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden*2, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden)
                ))

    def forward(self, x, pos, edge_index, edge_attr, graph_idx: List[torch.LongTensor], global_cond=None):
        x = torch.nan_to_num(x)
        pos = torch.nan_to_num(pos)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        pos = pos.clamp(-20.0, 20.0)

        edge_attr = torch.nan_to_num(edge_attr)

        h = self.node_embed(x)              # (N,H)
        e = self.edge_embed(edge_attr)      # (E,H)

        h   = torch.nan_to_num(h, nan=0.0)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1.0, neginf=-1.0)
        e   = torch.nan_to_num(e, nan=0.0)
        
        if self.cond is not None and global_cond is not None:
            reps = [len(idx) for idx in graph_idx]
            h = self.cond(h, global_cond, reps)

        if self.use_egnn:
            # build PyG batch vector from graph_idx (list of node indices per graph)
            N = h.size(0)
            batch = torch.empty((N,), dtype=torch.long, device=h.device)
            for bi, idx in enumerate(graph_idx):
                batch[idx] = bi

            # EGNN_Sparse packs coords+feats into one tensor and returns the same format
            xcat = torch.cat([pos, h], dim=-1)
            pos0 = xcat[:, :3].clone()

            for layer in self.layers:
                xcat = layer(xcat, edge_index, edge_attr=e, batch=batch)
                xcat = torch.nan_to_num(xcat, nan=0.0, posinf=0.0, neginf=0.0)
                xcat[:, :3] = pos0   # <-- force coords not to drift


            pos = xcat[:, :3]
            h   = xcat[:, 3:]
            return h, pos


        # invariant fallback
        i, j = edge_index
        for layer in self.layers:
            m = torch.cat([h[i], h[j]], dim=-1)  # (E,2H)
            m = layer(m)
            agg = torch.zeros_like(h).index_add(0, i, m)
            h = F.silu(h + agg)
        return h, pos

class MotifPooling(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.att = nn.Linear(hidden, 1)

    def forward(self, h, batch_idx: List[torch.LongTensor]):
        outs = []
        for idx in batch_idx:
            hi = h[idx]
            mean = hi.mean(0, keepdim=True)
            mx   = hi.max(0, keepdim=True).values
            w    = torch.softmax(self.att(hi).squeeze(-1), dim=0).unsqueeze(-1)
            att  = (w * hi).sum(0, keepdim=True)
            outs.append(torch.cat([mean, mx, att], dim=-1))
        return torch.cat(outs, dim=0)

class PredictionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h: int,
        out_scalars: int,
        out_voigt: int = 21,
        n_hidden=2,
        out_classes: int = 0,
        enforce_spd_voigt: bool = True,
        spd_eps: float = 1e-4,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, h), nn.SiLU()]
        for _ in range(n_hidden-1):
            layers += [nn.Linear(h, h), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out_scalar = nn.Linear(h, out_scalars)
        self.out_voigt  = nn.Linear(h, out_voigt)
        self.enforce_spd_voigt = bool(enforce_spd_voigt)
        self.spd_eps = float(spd_eps)
        tril = torch.tril_indices(6, 6, offset=0)
        self.register_buffer("tril_i", tril[0])
        self.register_buffer("tril_j", tril[1])
        self.use_class  = out_classes > 0
        if self.use_class:
            self.out_class = nn.Linear(h, out_classes)

    def _raw21_to_spd_voigt21(self, raw21: torch.Tensor) -> torch.Tensor:
        """
        Map unconstrained 21 parameters to SPD 6x6 elastic tensor via Cholesky-like factorization:
          raw -> lower-triangular L (diag through softplus) -> C = L L^T -> Voigt-21.
        """
        bsz = raw21.size(0)
        L = torch.zeros((bsz, 6, 6), device=raw21.device, dtype=raw21.dtype)
        L[:, self.tril_i, self.tril_j] = raw21

        d = torch.diagonal(L, dim1=-2, dim2=-1)
        d_pos = F.softplus(d) + self.spd_eps
        L = L - torch.diag_embed(d) + torch.diag_embed(d_pos)

        C = L @ L.transpose(-1, -2)
        return pack_voigt_6x6_sym(C)

    def forward(self, z):
        h = self.mlp(z)
        raw_voigt = self.out_voigt(h)
        voigt = self._raw21_to_spd_voigt21(raw_voigt) if self.enforce_spd_voigt else raw_voigt
        out = {"scalars": self.out_scalar(h), "voigt": voigt, "voigt_raw": raw_voigt}
        if self.use_class:
            out["class"] = self.out_class(h)
        return out

class VAEHead(nn.Module):
    """VAE with optional conditioning on target number of elements"""
    def __init__(self, in_dim: int, latent: int, n_species: int, max_atoms: int=64, 
                 conditional: bool=True, max_num_elements: int=10):
        super().__init__()
        self.mu     = nn.Linear(in_dim, latent)
        self.logvar = nn.Linear(in_dim, latent)
        self.max_atoms = max_atoms
        self.n_species = n_species
        self.conditional = conditional
        
        # Conditioning: embed target_num_elements
        if conditional:
            self.num_elements_embed = nn.Embedding(max_num_elements + 1, 16)  # +1 for safety
            decoder_input_dim = latent + 16  # Concatenate embedding to latent
        else:
            self.num_elements_embed = None
            decoder_input_dim = latent
        
        # Decoders take concatenated [z, num_elements_embedding]
        self.dec_lattice = nn.Sequential(nn.Linear(decoder_input_dim, 128), nn.SiLU(), nn.Linear(128, 6))
        self.dec_natoms  = nn.Sequential(nn.Linear(decoder_input_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.dec_coords  = nn.Sequential(nn.Linear(decoder_input_dim, 256), nn.SiLU(), nn.Linear(256, 3*max_atoms))
        self.dec_species = nn.Sequential(nn.Linear(decoder_input_dim, 256), nn.SiLU(), nn.Linear(256, max_atoms*n_species))

    def encode(self, z):
        mu = self.mu(z)
        logvar = self.logvar(z)

        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        logvar = torch.nan_to_num(logvar, nan=0.0, posinf=0.0, neginf=0.0)
        logvar = torch.clamp(logvar, -8.0, 8.0)   # prevents exp overflow

        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, 1e-6, 1e2)  # avoid 0 or inf
        eps = torch.randn_like(std)
        z = mu + eps * std
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


    def decode(self, latent, target_num_elements=None):
        """
        Decode latent code to structure, optionally conditioning on target_num_elements.
        
        Args:
            latent: (batch_size, latent_dim) latent code
            target_num_elements: (batch_size,) target number of unique elements (1, 2, 3, ...)
                                If None and conditional=True, uses default value of 3
        
        Returns:
            dict with lattice6, nat, coords_frac, species_logits
        """
        if self.conditional:
            if target_num_elements is None:
                # Default to 3 elements (ternary) if not specified
                target_num_elements = torch.full((latent.size(0),), 3, 
                                                device=latent.device, dtype=torch.long)
            else:
                # Ensure it's on correct device and dtype
                if not isinstance(target_num_elements, torch.Tensor):
                    target_num_elements = torch.tensor(target_num_elements, 
                                                       device=latent.device, dtype=torch.long)
                else:
                    target_num_elements = target_num_elements.to(latent.device).long()
                    
                # Ensure it's 1D
                if target_num_elements.dim() == 0:
                    target_num_elements = target_num_elements.unsqueeze(0)
                    
            # Clamp to valid range
            target_num_elements = target_num_elements.clamp(1, 10)
            
            # Embed and concatenate
            num_elem_emb = self.num_elements_embed(target_num_elements)  # (batch, 16)
            latent = torch.cat([latent, num_elem_emb], dim=-1)  # (batch, latent+16)
        
        lat6 = self.dec_lattice(latent)
        nat  = torch.clamp(self.dec_natoms(latent).abs(), 1, self.max_atoms)
        coords = torch.sigmoid(self.dec_coords(latent)).view(-1, self.max_atoms, 3)
        species_logits = self.dec_species(latent).view(-1, self.max_atoms, self.n_species)
        return {"lattice6": lat6, "nat": nat, "coords_frac": coords, "species_logits": species_logits}

    def forward(self, crystal_embed, target_num_elements=None):
        """
        Full VAE forward pass
        
        Args:
            crystal_embed: Pooled crystal embedding
            target_num_elements: Optional target composition for conditional generation
        """
        mu, logvar = self.encode(crystal_embed)
        z_lat = self.reparameterize(mu, logvar)
        dec = self.decode(z_lat, target_num_elements)
        return dec, mu, logvar, z_lat

class AimMultiModalModel(nn.Module):
    def __init__(self,
                 node_dim: int, edge_dim: int,
                 hidden: int=128, n_layers: int=4,
                 cond_dim: Optional[int]=None,
                 out_scalars: int=8, out_voigt: int=21, out_classes: int=0,
                 latent: int=64, n_species: int=92, max_atoms: int=64,
                 use_egnn: bool=True, use_motif_pool: bool=True,
                 enforce_spd_voigt: bool=True, spd_eps: float=1e-4, hard_symmetry_mask: bool=False,
                 max_elements_loss: int=4,
                 scalar_mean: Optional[List[float]] = None,
                 scalar_std: Optional[List[float]] = None,
                 voigt_mean: Optional[List[float]] = None,
                 voigt_std: Optional[List[float]] = None):
        super().__init__()
        self.backbone = EquivariantBackbone(node_dim, edge_dim, hidden, n_layers, use_egnn, cond_dim)
        self.pool = MotifPooling(hidden) if use_motif_pool else None
        pooled_dim = hidden*3 if use_motif_pool else hidden
        self.max_elements_loss = int(max_elements_loss)
        self.hard_symmetry_mask = bool(hard_symmetry_mask)

        self.vae  = VAEHead(pooled_dim, latent, n_species, max_atoms)

        # IMPORTANT: predict from latent (so generation can be predicted directly)
        self.pred = PredictionHead(
            latent, hidden, out_scalars, out_voigt, n_hidden=2, out_classes=out_classes,
            enforce_spd_voigt=enforce_spd_voigt, spd_eps=spd_eps,
        )

        self.register_buffer("qgrid", self._make_qgrid(64, 6.0))
        self.register_buffer("_scalar_mean_buf", torch.empty(0))
        self.register_buffer("_scalar_std_buf", torch.empty(0))
        self.register_buffer("_voigt_mean_buf", torch.empty(0))
        self.register_buffer("_voigt_std_buf", torch.empty(0))

        if scalar_mean is not None and scalar_std is not None:
            sm = torch.tensor(scalar_mean, dtype=torch.float32).view(1, -1)
            ss = torch.tensor(scalar_std, dtype=torch.float32).view(1, -1)
            if sm.numel() > 0 and sm.shape == ss.shape:
                self._scalar_mean_buf = sm
                self._scalar_std_buf = ss
        if voigt_mean is not None and voigt_std is not None:
            vm = torch.tensor(voigt_mean, dtype=torch.float32).view(1, -1)
            vs = torch.tensor(voigt_std, dtype=torch.float32).view(1, -1)
            if vm.numel() > 0 and vm.shape == vs.shape:
                self._voigt_mean_buf = vm
                self._voigt_std_buf = vs

    @staticmethod
    def _make_qgrid(nq=64, qmax=6.0):
        q = torch.randn(nq, 3)
        q = q / (q.norm(dim=-1, keepdim=True)+1e-9)
        r = torch.rand(nq,1)*qmax
        return q*r

    @staticmethod
    def _lat6_to_matrix(lat6: torch.Tensor) -> torch.Tensor:
        a,b,c,alpha,beta,gamma = [lat6[:,i] for i in range(6)]
        a = torch.clamp(a, 1e-2, 50.0); b = torch.clamp(b, 1e-2, 50.0); c = torch.clamp(c, 1e-2, 50.0)
        alpha = torch.clamp(alpha, 5.0, 175.0); beta = torch.clamp(beta, 5.0, 175.0); gamma = torch.clamp(gamma, 5.0, 175.0)

        alpha = torch.deg2rad(alpha); beta = torch.deg2rad(beta); gamma = torch.deg2rad(gamma)
        va = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
        vb = torch.stack([b*torch.cos(gamma), b*torch.sin(gamma), torch.zeros_like(b)], dim=-1)

        cx = c*torch.cos(beta)
        denom = torch.sin(gamma) + 1e-6
        cy = c*(torch.cos(alpha) - torch.cos(beta)*torch.cos(gamma)) / denom
        cz_sq = (c**2 - cx**2 - cy**2)
        cz = torch.sqrt(torch.clamp(cz_sq, min=1e-6))
        vc = torch.stack([cx, cy, cz], dim=-1)

        L = torch.stack([va, vb, vc], dim=1)
        return torch.nan_to_num(L)

    @staticmethod
    def _safe_std(std: torch.Tensor) -> torch.Tensor:
        return torch.where(std.abs() < 1e-12, torch.ones_like(std), std)

    def _denorm_scalars(self, scalars: torch.Tensor) -> Optional[torch.Tensor]:
        if self._scalar_mean_buf.numel() == 0 or self._scalar_std_buf.numel() == 0:
            return None
        mean = self._scalar_mean_buf.to(device=scalars.device, dtype=scalars.dtype)
        std = self._safe_std(self._scalar_std_buf.to(device=scalars.device, dtype=scalars.dtype))
        if scalars.dim() != 2 or mean.shape[-1] != scalars.shape[-1]:
            return None
        return scalars * std + mean

    def _denorm_voigt(self, voigt: torch.Tensor) -> Optional[torch.Tensor]:
        if self._voigt_mean_buf.numel() == 0 or self._voigt_std_buf.numel() == 0:
            return None
        mean = self._voigt_mean_buf.to(device=voigt.device, dtype=voigt.dtype)
        std = self._safe_std(self._voigt_std_buf.to(device=voigt.device, dtype=voigt.dtype))
        if voigt.dim() != 2 or mean.shape[-1] != voigt.shape[-1]:
            return None
        return voigt * std + mean

    def forward(
            self,
            atom_fea: torch.Tensor,
            pos: torch.Tensor,
            edge_index: torch.LongTensor,
            edge_attr: torch.Tensor,
            crystal_atom_idx,
            global_cond: Optional[torch.Tensor] = None,
            y_scalars: Optional[torch.Tensor] = None,
            y_voigt: Optional[torch.Tensor] = None,
            y_classes: Optional[torch.Tensor] = None,
            y_frac: Optional[List[torch.Tensor]] = None,
            y_lattice: Optional[List[torch.Tensor]] = None,
            y_species: Optional[List[torch.Tensor]] = None,
            crystal_systems: Optional[List[str]] = None,
            batch_nat: Optional[torch.Tensor] = None,
            target_num_elements: Optional[torch.Tensor] = None
            ):
        h, pos_out = self.backbone(atom_fea, pos, edge_index, edge_attr, crystal_atom_idx, global_cond)

        if self.pool is None:
            z_crys = torch.stack([h[idx].mean(0) for idx in crystal_atom_idx], dim=0)
        else:
            z_crys = self.pool(h, crystal_atom_idx)
        
                # --- sanitize BEFORE VAE ---
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        z_crys = torch.nan_to_num(z_crys, nan=0.0, posinf=0.0, neginf=0.0)

        # optional: clamp to prevent exploding activations
        z_crys = torch.clamp(z_crys, -10.0, 10.0)

        if target_num_elements is not None:
            target_num_elements = target_num_elements.to(z_crys.device).long().view(-1)
            if target_num_elements.numel() == 1 and z_crys.size(0) > 1:
                target_num_elements = target_num_elements.repeat(z_crys.size(0))
            elif target_num_elements.numel() != z_crys.size(0):
                target_num_elements = target_num_elements[:z_crys.size(0)]


        dec, mu, logvar, z_lat = self.vae(z_crys, target_num_elements)

                # --- sanitize mu BEFORE prediction ---
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        mu = torch.clamp(mu, -10.0, 10.0)
        
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        z_crys = torch.nan_to_num(z_crys, nan=0.0, posinf=0.0, neginf=0.0)


        # PREDICT FROM LATENT (mu is stable for inference)
        pred = self.pred(mu)
        pred_sample = self.pred(z_lat)

        cs_batch = None
        if crystal_systems is not None:
            try:
                cs_list = [str(x) for x in crystal_systems]
            except Exception:
                cs_list = [str(crystal_systems)]
            if len(cs_list) > 0:
                cs_batch = [cs_list[b] if b < len(cs_list) else cs_list[-1] for b in range(pred["voigt"].size(0))]

        if self.hard_symmetry_mask and cs_batch is not None:
            for b in range(pred["voigt"].size(0)):
                mask = symmetry_mask_21(cs_batch[b], device=pred["voigt"].device)
                pred["voigt"][b] = pred["voigt"][b] * mask.float()
                pred_sample["voigt"][b] = pred_sample["voigt"][b] * mask.float()

        # --- sanitize predictions so loss never NaNs ---
        pred["scalars"] = torch.nan_to_num(pred["scalars"], nan=0.0, posinf=0.0, neginf=0.0)
        pred["voigt"]   = torch.nan_to_num(pred["voigt"],   nan=0.0, posinf=0.0, neginf=0.0)
        pred_sample["scalars"] = torch.nan_to_num(pred_sample["scalars"], nan=0.0, posinf=0.0, neginf=0.0)
        pred_sample["voigt"]   = torch.nan_to_num(pred_sample["voigt"],   nan=0.0, posinf=0.0, neginf=0.0)


        out = {"pred": pred, "pred_sample": pred_sample, "gen": dec, "mu": mu, "logvar": logvar, "z_lat": z_lat, "z": z_crys}

        losses = {}
        mu_s = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        lv_s = torch.nan_to_num(logvar, nan=0.0, posinf=0.0, neginf=0.0)
        lv_s = torch.clamp(lv_s, -8.0, 8.0)

        kld = -0.5 * (1.0 + lv_s - mu_s.pow(2) - torch.exp(lv_s))
        kld = torch.nan_to_num(kld, nan=0.0, posinf=0.0, neginf=0.0)
        losses["L_kld"] = kld.mean()


        latt = self._lat6_to_matrix(dec["lattice6"])
        B = dec["coords_frac"].size(0)
        mind = []
        for b in range(B):
            mind.append(safe_min_distance(dec["coords_frac"][b], latt[b]))
        mind = torch.stack(mind)
        target_min = torch.tensor(1.2, device=mind.device)
        losses["L_minDist"] = F.relu(target_min - mind).mean()

        S_mag = structure_factor_magnitude(dec["coords_frac"].view(-1,3), latt.mean(dim=0), self.qgrid)
        losses["L_recip"] = -S_mag.mean() * 1e-3

        # L_nat: Number of atoms prediction loss
        if batch_nat is not None and "nat" in dec:
            nat_pred = dec["nat"].squeeze(-1)  # (B,)
            nat_true = batch_nat.to(device=nat_pred.device, dtype=nat_pred.dtype)
            losses["L_nat"] = torch.abs(nat_pred - nat_true).mean()
        else:
            losses["L_nat"] = torch.tensor(0.0, device=dec["lattice6"].device)       

        # Couple deterministic (mu) and sampled (z) property predictions used by generation.
        losses["L_consistency"] = (
            torch.abs(pred["scalars"] - pred_sample["scalars"]).mean()
            + torch.abs(pred["voigt"] - pred_sample["voigt"]).mean()
        )

        if y_scalars is not None:
            B, D = pred["scalars"].shape

            if y_scalars.dim() == 1 and y_scalars.numel() == B * D:
                y_scalars = y_scalars.view(B, D)

            if y_scalars.dim() == 2 and y_scalars.shape == (B, D):
                mask = torch.isfinite(y_scalars) & torch.isfinite(pred["scalars"])
                if mask.any():
                    losses["L_pred_scalar"] = torch.abs(pred["scalars"][mask] - y_scalars[mask]).mean()
                else:
                    losses["L_pred_scalar"] = torch.tensor(0.0, device=pred["scalars"].device)


        losses["L_voigt"] = torch.tensor(0.0, device=pred["voigt"].device)
        if y_voigt is not None:
            Bv, Dv = pred["voigt"].shape  # (B, 21)

            if y_voigt.dim() == 1 and y_voigt.numel() == Bv * Dv:
                y_voigt = y_voigt.view(Bv, Dv)

            if y_voigt.dim() == 2 and y_voigt.shape == (Bv, Dv):
                mask = torch.isfinite(y_voigt) & torch.isfinite(pred["voigt"])
                if mask.any():
                    losses["L_voigt"] = torch.abs(pred["voigt"][mask] - y_voigt[mask]).mean()
                else:
                    losses["L_voigt"] = torch.tensor(0.0, device=pred["voigt"].device)

        # Physics consistency in physical units (if normalization stats are available):
        # 1) Predicted scalar mechanics should be non-negative.
        # 2) Scalar bulk/shear should agree with Voigt-derived bulk/shear from predicted C_ij.
        losses["L_scalar_pos"] = torch.tensor(0.0, device=pred["scalars"].device)
        losses["L_tensor_scalar_cons"] = torch.tensor(0.0, device=pred["scalars"].device)
        try:
            scalars_phys = self._denorm_scalars(pred["scalars"])
            voigt_phys = self._denorm_voigt(pred["voigt"])
            if scalars_phys is not None and voigt_phys is not None and scalars_phys.size(1) >= 3 and voigt_phys.size(1) == 21:
                bulk = scalars_phys[:, 0]
                shear = scalars_phys[:, 1]
                young = scalars_phys[:, 2]

                # Non-negativity prior for mechanically meaningful scalar properties.
                pos_terms = [bulk, shear, young]
                if scalars_phys.size(1) >= 8:
                    pos_terms.extend(
                        [
                            scalars_phys[:, 3],
                            scalars_phys[:, 4],
                            scalars_phys[:, 5],
                            scalars_phys[:, 6],
                            scalars_phys[:, 7],
                        ]
                    )
                losses["L_scalar_pos"] = torch.stack([F.relu(-t) for t in pos_terms], dim=0).mean()

                c6 = unpack_voigt_21_sym(voigt_phys)
                c11, c22, c33 = c6[:, 0, 0], c6[:, 1, 1], c6[:, 2, 2]
                c12, c13, c23 = c6[:, 0, 1], c6[:, 0, 2], c6[:, 1, 2]
                c44, c55, c66 = c6[:, 3, 3], c6[:, 4, 4], c6[:, 5, 5]

                b_voigt = (c11 + c22 + c33 + 2.0 * (c12 + c13 + c23)) / 9.0
                g_voigt = (c11 + c22 + c33 - (c12 + c13 + c23) + 3.0 * (c44 + c55 + c66)) / 15.0
                y_from_bg = (9.0 * bulk * shear) / torch.clamp(3.0 * bulk + shear, min=1e-6)

                def _rel_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return torch.abs(a - b) / torch.clamp(torch.abs(b), min=1.0)

                losses["L_tensor_scalar_cons"] = (
                    _rel_l1(bulk, b_voigt)
                    + _rel_l1(shear, g_voigt)
                    + _rel_l1(young, y_from_bg)
                ).mean()
        except Exception:
            losses["L_scalar_pos"] = torch.tensor(0.0, device=pred["scalars"].device)
            losses["L_tensor_scalar_cons"] = torch.tensor(0.0, device=pred["scalars"].device)

        # PD regularization: keep all eigenvalues positive.
        try:
            C_pred = unpack_voigt_21_sym(pred["voigt"])
            eigvals = torch.linalg.eigvalsh(C_pred)
            losses["L_voigt_pd"] = F.relu(1e-6 - eigvals).mean()
        except Exception:
            losses["L_voigt_pd"] = torch.tensor(0.0, device=pred["voigt"].device)

        # Crystal-system symmetry regularization (soft zero mask + equality constraints).
        if cs_batch is not None:
            try:
                masks = torch.stack(
                    [symmetry_mask_21(cs, device=pred["voigt"].device) for cs in cs_batch],
                    dim=0,
                )
                forbidden = (~masks).float()
                denom = forbidden.sum().clamp(min=1.0)
                losses["L_voigt_sym"] = (pred["voigt"].abs() * forbidden).sum() / denom
            except Exception:
                losses["L_voigt_sym"] = torch.tensor(0.0, device=pred["voigt"].device)

            try:
                losses["L_voigt_eq"] = equality_penalties(pred["voigt"], cs_batch)
            except Exception:
                losses["L_voigt_eq"] = torch.tensor(0.0, device=pred["voigt"].device)
        else:
            losses["L_voigt_sym"] = torch.tensor(0.0, device=pred["voigt"].device)
            losses["L_voigt_eq"] = torch.tensor(0.0, device=pred["voigt"].device)


        if y_classes is not None and "class" in pred:
            losses["L_class"] = F.cross_entropy(pred["class"], y_classes)
        
        if dec is not None and "species_logits" in dec:
            # Loss 1: Limit element diversity (max 8 elements per structure)
            losses["L_composition_diversity"] = composition_diversity_loss(
                dec["species_logits"],
                dec["nat"],
                max_elements=self.max_elements_loss,
                temperature=0.1
            )
            
            # Loss 2: Strongly penalize noble gases (He, Ne, Ar, Kr, Xe, Rn)
            # These don't form compounds!
            losses["L_noble_gas"] = noble_gas_penalty(
                dec["species_logits"],
                NOBLE_GASES
            )
            
            # Loss 3: Penalize radioactive elements (especially short-lived)
            losses["L_radioactive"] = radioactive_penalty(
                dec["species_logits"],
                RADIOACTIVE,
                RADIOACTIVE_SEVERITY.to(dec["species_logits"].device)
            )
        else:
            # During inference without generation, set to zero
            losses["L_composition_diversity"] = torch.tensor(0.0, device=pred["scalars"].device)
            losses["L_noble_gas"] = torch.tensor(0.0, device=pred["scalars"].device)
            losses["L_radioactive"] = torch.tensor(0.0, device=pred["scalars"].device)

        out["losses"] = losses
        # Only crash on supervised losses. For unsupervised ones, auto-zero if unstable.
        safe = {}
        for k, v in losses.items():
            # Convert to tensor if it's a Python float/int (BUGFIX)
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            
            if torch.isnan(v).any() or torch.isinf(v).any():
                if k in ["L_pred_scalar", "L_voigt", "L_class"]:
                    raise RuntimeError(f"Loss {k} became NaN/Inf")
                else:
                    safe[k] = torch.tensor(0.0, device=v.device)
            else:
                safe[k] = v
        out["losses"] = safe


        return out
