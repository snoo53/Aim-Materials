"""
Physics-informed composition constraints for materials generation.
Prevents unrealistic multi-element structures.
"""
import torch
import torch.nn.functional as F


def composition_diversity_loss(
    species_logits: torch.Tensor,
    nat: torch.Tensor,
    max_elements: int = 4,  # Changed from 8 to 4! 🎯
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Penalizes structures with too many unique element types.
    
    Args:
        species_logits: [B, max_atoms, n_species] - raw logits from decoder
        nat: [B, 1] - predicted number of atoms per structure
        max_elements: Maximum allowed unique elements (default: 8)
        temperature: Softmax temperature for soft counting
        
    Returns:
        loss: Scalar tensor penalizing diversity
    """
    B, max_atoms, n_species = species_logits.shape
    
    # Soft element presence using temperature-scaled softmax
    # Lower temperature = sharper, closer to hard argmax
    probs = F.softmax(species_logits / temperature, dim=-1)  # [B, max_atoms, n_species]
    
    # Sum probabilities across all atomic sites to get "element importance"
    element_scores = probs.sum(dim=1)  # [B, n_species]
    
    # Soft count: how many elements have significant presence?
    # Use sigmoid to get soft "is this element present?" indicator
    element_presence = torch.sigmoid(10 * (element_scores - 0.5))  # [B, n_species]
    n_elements = element_presence.sum(dim=-1)  # [B]
    
    # Penalize exceeding max_elements
    penalty = F.relu(n_elements - max_elements)
    
    return penalty.mean()


def charge_neutrality_loss(species_logits, nat, element_charges):
    """
    Enforces charge neutrality constraint.
    
    Args:
        species_logits: [B, max_atoms, n_species]
        nat: [B, 1] - number of atoms
        element_charges: [n_species] - typical oxidation states (e.g., O=-2, Fe=+3)
        
    Returns:
        loss: Penalizes non-neutral total charge
    """
    B, max_atoms, n_species = species_logits.shape
    
    # Get soft species assignment
    probs = F.softmax(species_logits, dim=-1)  # [B, max_atoms, n_species]
    
    # Compute expected charge per atom: sum over species dimension
    charges_per_atom = (probs * element_charges.view(1, 1, -1)).sum(dim=-1)  # [B, max_atoms]
    
    # Total charge per structure (mask out atoms beyond nat)
    nat_idx = torch.arange(max_atoms, device=species_logits.device).view(1, -1)  # [1, max_atoms]
    nat_mask = (nat_idx < nat).float()  # [B, max_atoms]
    
    total_charge = (charges_per_atom * nat_mask).sum(dim=-1)  # [B]
    
    # Penalize deviation from neutrality
    return (total_charge ** 2).mean()


def element_cooccurrence_loss(species_logits, cooccurrence_matrix, temperature=0.1):
    """
    Encourages chemically realistic element combinations.
    
    Args:
        species_logits: [B, max_atoms, n_species]
        cooccurrence_matrix: [n_species, n_species] - learned or pre-computed
                            High values = elements often appear together
                            Low values = rare/impossible combinations
        temperature: Softmax temperature
        
    Returns:
        loss: Encourages likely element pairs
    """
    B, max_atoms, n_species = species_logits.shape
    
    # Get soft species probabilities
    probs = F.softmax(species_logits / temperature, dim=-1)  # [B, max_atoms, n_species]
    
    # Compute element presence vector per structure
    element_presence = probs.sum(dim=1)  # [B, n_species]
    element_presence = element_presence / (element_presence.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize
    
    # Compute expected co-occurrence score
    # For each structure, how compatible are its elements?
    # element_presence [B, n_species] @ cooccurrence_matrix [n_species, n_species] @ element_presence.T [n_species, B]
    compatibility = torch.einsum('bi,ij,bj->b', element_presence, cooccurrence_matrix, element_presence)
    
    # Maximize compatibility (minimize negative)
    return -compatibility.mean()


def noble_gas_penalty(species_logits, noble_gas_indices):
    """
    Strongly penalizes noble gas inclusion (He, Ne, Ar, Kr, Xe, Rn).
    
    Args:
        species_logits: [B, max_atoms, n_species]
        noble_gas_indices: List of indices for noble gases
        
    Returns:
        loss: Penalizes noble gas presence
    """
    # Sum logits for noble gases
    noble_logits = species_logits[:, :, noble_gas_indices]  # [B, max_atoms, n_noble]
    
    # Apply penalty to any positive logits (model wants to use noble gas)
    penalty = F.relu(noble_logits).sum()
    
    return penalty


def radioactive_penalty(species_logits, radioactive_indices, severity_weights):
    """
    Penalizes radioactive elements (especially short-lived ones).
    
    Args:
        species_logits: [B, max_atoms, n_species]
        radioactive_indices: List of indices
        severity_weights: [n_radioactive] - higher for more unstable
        
    Returns:
        loss: Penalizes radioactive element use
    """
    radioactive_logits = species_logits[:, :, radioactive_indices]  # [B, max_atoms, n_radioactive]
    
    # Weight by severity
    weighted_penalty = F.relu(radioactive_logits) * severity_weights.view(1, 1, -1)
    
    return weighted_penalty.sum()


def get_typical_oxidation_states(n_species=92):
    """
    Returns typical oxidation states for elements 1-92.
    Simplified - you should customize based on your materials.
    """
    charges = torch.zeros(n_species)
    
    # Group 1: +1 (Li, Na, K, Rb, Cs, Fr)
    charges[[2, 10, 18, 36, 54, 86]] = 1.0
    
    # Group 2: +2 (Be, Mg, Ca, Sr, Ba, Ra)
    charges[[3, 11, 19, 37, 55, 87]] = 2.0
    
    # Group 13: +3 (Al, Ga, In, Tl)
    charges[[12, 30, 48, 80]] = 3.0
    
    # Transition metals: variable (use +2 as default)
    charges[21:30] = 2.0  # Sc-Zn
    charges[39:48] = 2.0  # Y-Cd
    
    # Oxygen: -2
    charges[7] = -2.0
    
    # Halogens: -1 (F, Cl, Br, I, At)
    charges[[8, 16, 34, 52, 84]] = -1.0
    
    return charges


def build_cooccurrence_matrix(dataset, n_species=92, smoothing=0.01):
    """
    Build element co-occurrence matrix from training data.
    
    Args:
        dataset: List of structures with species information
        n_species: Number of elements
        smoothing: Laplace smoothing to avoid zeros
        
    Returns:
        matrix: [n_species, n_species] symmetric matrix
    """
    import numpy as np
    
    matrix = np.zeros((n_species, n_species))
    
    for structure in dataset:
        # Get unique element indices in this structure
        elements = np.unique(structure['species'])  # or however you access species
        
        # Increment co-occurrence for all pairs
        for i in elements:
            for j in elements:
                matrix[i, j] += 1.0
    
    # Normalize by row sums (how often does element i appear with element j given i is present?)
    row_sums = matrix.sum(axis=1, keepdims=True) + smoothing
    matrix = (matrix + smoothing) / row_sums
    
    # Symmetrize
    matrix = (matrix + matrix.T) / 2.0
    
    return torch.FloatTensor(matrix)


# Element indices for constraints
NOBLE_GASES = [1, 9, 17, 35, 53, 85]  # He, Ne, Ar, Kr, Xe, Rn (0-indexed)
RADIOACTIVE = [60, 88, 89, 90, 91]     # Pm, Ac, Th, Pa, U (simplified)
RADIOACTIVE_SEVERITY = torch.tensor([5.0, 10.0, 3.0, 8.0, 3.0])  # Pm very unstable, Th less so