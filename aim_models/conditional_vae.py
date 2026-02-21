# aim_models/conditional_vae.py
"""
Conditional VAE for composition-controlled materials generation.

Key Idea:
- Encoder: Takes structure → latent z
- Decoder: Takes (latent z + composition) → structure
- Generation: Specify composition → model generates structures with that composition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_mean, scatter_add
import numpy as np

from .e3_multi_modal import (
    E3NN_Encoder, 
    PropertyPredictor, 
    VoigtElasticPredictor,
    NatPredictor
)


class CompositionEncoder(nn.Module):
    """
    Encode composition vector (element counts) into embedding.
    Input: [batch_size, 92] (counts for each element)
    Output: [batch_size, comp_embed_dim]
    """
    def __init__(self, n_species=92, comp_embed_dim=64):
        super().__init__()
        self.n_species = n_species
        self.comp_embed_dim = comp_embed_dim
        
        # Normalize counts to fractions
        self.normalize = True
        
        # MLP to embed composition
        self.encoder = nn.Sequential(
            nn.Linear(n_species, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, comp_embed_dim)
        )
    
    def forward(self, composition_vector):
        """
        Args:
            composition_vector: [batch_size, 92] element counts
        Returns:
            comp_embedding: [batch_size, comp_embed_dim]
        """
        # Normalize to fractions
        if self.normalize:
            total = composition_vector.sum(dim=-1, keepdim=True).clamp(min=1.0)
            composition_vector = composition_vector / total
        
        return self.encoder(composition_vector)


class ConditionalDecoder(nn.Module):
    """
    Decoder that takes latent z + composition embedding → structure.
    """
    def __init__(self, latent_dim, comp_embed_dim, hidden_dim, n_species, max_atoms=200):
        super().__init__()
        self.latent_dim = latent_dim
        self.comp_embed_dim = comp_embed_dim
        self.hidden_dim = hidden_dim
        self.n_species = n_species
        self.max_atoms = max_atoms
        
        # Combined latent + composition → per-atom features
        combined_dim = latent_dim + comp_embed_dim
        
        # Predict number of atoms (optional, can be fixed)
        self.nat_predictor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        # Expand to per-atom features
        self.atom_expander = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * max_atoms),
            nn.SiLU()
        )
        
        # Refine per-atom features
        self.atom_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Predict positions
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Predict species (conditioned on composition)
        # This uses composition embedding to bias predictions
        self.species_head = nn.Sequential(
            nn.Linear(hidden_dim + comp_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_species)
        )
    
    def forward(self, z, comp_embedding, batch_size, n_atoms=64):
        """
        Args:
            z: [batch_size, latent_dim]
            comp_embedding: [batch_size, comp_embed_dim]
            batch_size: int
            n_atoms: int (fixed atom count)
        
        Returns:
            positions: [batch_size * n_atoms, 3]
            species_logits: [batch_size * n_atoms, n_species]
        """
        # Combine latent + composition
        combined = torch.cat([z, comp_embedding], dim=-1)  # [B, latent+comp]
        
        # Expand to per-atom features
        atom_features = self.atom_expander(combined)  # [B, hidden*max_atoms]
        atom_features = atom_features.view(batch_size, self.max_atoms, self.hidden_dim)
        atom_features = atom_features[:, :n_atoms, :]  # [B, n_atoms, hidden]
        
        # Flatten for processing
        atom_features_flat = atom_features.view(-1, self.hidden_dim)  # [B*n_atoms, hidden]
        
        # Refine
        atom_features_flat = self.atom_refiner(atom_features_flat)
        
        # Predict positions
        positions = self.position_head(atom_features_flat)  # [B*n_atoms, 3]
        
        # Predict species (with composition conditioning)
        # Expand comp_embedding to per-atom
        comp_per_atom = comp_embedding.unsqueeze(1).expand(-1, n_atoms, -1)  # [B, n_atoms, comp_dim]
        comp_per_atom_flat = comp_per_atom.reshape(-1, self.comp_embed_dim)  # [B*n_atoms, comp_dim]
        
        # Concatenate and predict
        species_input = torch.cat([atom_features_flat, comp_per_atom_flat], dim=-1)
        species_logits = self.species_head(species_input)  # [B*n_atoms, n_species]
        
        return positions, species_logits


class ConditionalMaterialVAE(nn.Module):
    """
    Conditional VAE for materials with composition control.
    """
    def __init__(
        self,
        n_species=92,
        hidden_dim=128,
        latent_dim=256,
        comp_embed_dim=64,
        edge_dim=6,
        angle_dim=1,
        global_dim=105,
        use_voigt_symmetry=True,
        max_atoms=200
    ):
        super().__init__()
        
        self.n_species = n_species
        self.latent_dim = latent_dim
        self.comp_embed_dim = comp_embed_dim
        self.use_voigt_symmetry = use_voigt_symmetry
        self.max_atoms = max_atoms
        
        # Composition encoder (for conditioning)
        self.comp_encoder = CompositionEncoder(n_species, comp_embed_dim)
        
        # Structure encoder (same as before)
        self.encoder = E3NN_Encoder(
            n_species=n_species,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            angle_dim=angle_dim
        )
        
        # VAE latent layers
        encoder_out_dim = hidden_dim
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)
        
        # Conditional decoder
        self.decoder = ConditionalDecoder(
            latent_dim=latent_dim,
            comp_embed_dim=comp_embed_dim,
            hidden_dim=hidden_dim,
            n_species=n_species,
            max_atoms=max_atoms
        )
        
        # Property predictors (same as before)
        self.property_predictor = PropertyPredictor(
            latent_dim=latent_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            n_scalar_targets=8
        )
        
        self.voigt_predictor = VoigtElasticPredictor(
            latent_dim=latent_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            use_voigt_symmetry=use_voigt_symmetry
        )
        
        self.nat_predictor = NatPredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
    
    def extract_composition_vector(self, batch_data):
        """
        Extract composition vector from batch.
        Returns: [batch_size, 92] with element counts
        """
        species = batch_data.x[:, 0]  # Assuming first feature is species index
        batch_indices = batch_data.batch
        batch_size = batch_indices.max().item() + 1
        
        # Initialize composition vectors
        comp_vectors = torch.zeros(batch_size, self.n_species, 
                                   device=batch_data.x.device)
        
        # Count species per batch
        for b in range(batch_size):
            mask = (batch_indices == b)
            batch_species = species[mask].long()
            # Count occurrences
            for sp_idx in batch_species:
                if 0 <= sp_idx < self.n_species:
                    comp_vectors[b, sp_idx] += 1
        
        return comp_vectors
    
    def encode(self, batch_data):
        """Encode structure to latent distribution."""
        # Encode structure
        node_emb, graph_emb = self.encoder(batch_data)
        
        # Latent distribution
        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, composition_vector, n_atoms=64):
        """
        Decode latent + composition to structure.
        
        Args:
            z: [batch_size, latent_dim]
            composition_vector: [batch_size, 92] element counts
            n_atoms: int (fixed atom count)
        """
        batch_size = z.size(0)
        
        # Encode composition
        comp_embedding = self.comp_encoder(composition_vector)
        
        # Decode to structure
        positions, species_logits = self.decoder(
            z, comp_embedding, batch_size, n_atoms
        )
        
        return {
            'pos': positions,
            'species_logits': species_logits,
            'comp_embedding': comp_embedding
        }
    
    def forward(self, batch_data, n_atoms=64):
        """Full forward pass."""
        # Extract composition from input
        comp_vector = self.extract_composition_vector(batch_data)
        
        # Encode
        mu, logvar = self.encode(batch_data)
        z = self.reparameterize(mu, logvar)
        
        # Decode (conditioned on composition)
        recon = self.decode(z, comp_vector, n_atoms)
        
        # Predict properties
        global_features = batch_data.global_features
        pred_scalars = self.property_predictor(z, global_features)
        pred_voigt = self.voigt_predictor(z, global_features)
        pred_nat = self.nat_predictor(z)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'recon_pos': recon['pos'],
            'recon_species_logits': recon['species_logits'],
            'pred_scalars': pred_scalars,
            'pred_voigt': pred_voigt,
            'pred_nat': pred_nat,
            'comp_vector': comp_vector
        }
    
    def generate(self, composition_dict, n_structures=1, device='cuda'):
        """
        Generate structures with specified composition.
        
        Args:
            composition_dict: dict, e.g., {'Ti': 1, 'O': 2} for TiO2
            n_structures: int, number to generate
        
        Returns:
            list of generated structures
        """
        self.eval()
        
        # Convert composition dict to vector
        comp_vector = torch.zeros(1, self.n_species, device=device)
        total_atoms = 0
        for element, count in composition_dict.items():
            if hasattr(self, 'species_to_idx'):
                sp_idx = self.species_to_idx[element]
            else:
                # Assume alphabetical ordering
                sp_idx = sorted(list(range(self.n_species)))[
                    ord(element[0]) - ord('A')
                ]
            comp_vector[0, sp_idx] = count
            total_atoms += count
        
        # Repeat for batch
        comp_vector = comp_vector.repeat(n_structures, 1)
        
        generated = []
        with torch.no_grad():
            # Sample latent vectors
            z = torch.randn(n_structures, self.latent_dim, device=device)
            
            # Decode
            recon = self.decode(z, comp_vector, n_atoms=total_atoms)
            
            # Extract structures
            positions = recon['pos'].view(n_structures, total_atoms, 3)
            species_logits = recon['species_logits'].view(
                n_structures, total_atoms, self.n_species
            )
            species = torch.argmax(species_logits, dim=-1)
            
            for i in range(n_structures):
                generated.append({
                    'positions': positions[i].cpu().numpy(),
                    'species': species[i].cpu().numpy(),
                    'composition': composition_dict
                })
        
        return generated