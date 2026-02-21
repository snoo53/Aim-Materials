# process_filtered_mp.py
"""
Process filtered Materials Project data into graph format for Phase 2 training.
Modified from Data Extraction.ipynb to use filtered data.
"""

import os
import json
import numpy as np
import torch
import pandas as pd
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.distance import cosine
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class ElementDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.filtered_columns = None
        self.one_hot_encoders = {}
        self.oxidation_state_range = list(range(-3,8))
        self.feature_means = None
        self.feature_stds = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        
        for prop in ['StandardState', 'GroupBlock']:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.df[[prop]] = self.df[[prop]].astype(str)
            encoder.fit(self.df[[prop]])
            self.one_hot_encoders[prop] = encoder

    def columns(self):
        self.filtered_columns = self.df[['AtomicNumber', 'Symbol', 'AtomicMass', 'Electronegativity', 
                                'AtomicRadius', 'IonizationEnergy', 'ElectronAffinity', 
                                'OxidationStates', 'StandardState', 'MeltingPoint', 'BoilingPoint', 'Density',
                                'GroupBlock','YoungModulus','BulkModulus','ShearModulus','PoissonRatio']
                                ].set_index('Symbol')
        
        self.filtered_columns['has_young_modulus'] = self.filtered_columns['YoungModulus'].notna().astype(float)
        self.filtered_columns['has_bulk_modulus'] = self.filtered_columns['BulkModulus'].notna().astype(float)
        self.filtered_columns['has_shear_modulus'] = self.filtered_columns['ShearModulus'].notna().astype(float)
        self.filtered_columns['has_poisson_ratio'] = self.filtered_columns['PoissonRatio'].notna().astype(float)
        
        gas_rows = self.filtered_columns['StandardState'] == 'gas'
        self.filtered_columns.loc[gas_rows] = self.filtered_columns.loc[gas_rows].fillna(0)
        
        solid_rows = self.filtered_columns['StandardState'] == 'solid'
        
        for col in self.filtered_columns.columns:
            if col in ['YoungModulus','BulkModulus','ShearModulus','PoissonRatio']:
                mean_value = self.filtered_columns.loc[solid_rows, col].mean()
                self.filtered_columns.loc[solid_rows, col] = self.filtered_columns.loc[solid_rows, col].fillna(mean_value)

    def means_stds(self):
        num_columns = ['AtomicMass', 'Electronegativity', 'AtomicRadius', 'IonizationEnergy', 
                            'ElectronAffinity', 'MeltingPoint', 'BoilingPoint', 'Density', 
                            'YoungModulus', 'BulkModulus', 'ShearModulus', 'PoissonRatio']
        
        self.feature_means = self.filtered_columns[num_columns].mean()
        self.feature_stds = self.filtered_columns[num_columns].std()

    def normalize_node_features(self, feature_vector):
        num_features = feature_vector[:12]
        normalized_num_features = [
            (value - self.feature_means[col]) / self.feature_stds[col] if self.feature_stds[col] != 0 else 0
            for value, col in zip(num_features, self.feature_means.index)
        ]
        return normalized_num_features + feature_vector[12:]

    def encode_oxidation_states(self, oxidation_states):
        vector = [0] * len(self.oxidation_state_range)
        for state in oxidation_states.split(', '):
            state = state.strip()
            try:
                idx = self.oxidation_state_range.index(int(state))
                vector[idx] = 1
            except (ValueError, IndexError):
                continue
        return vector

    def get_element_properties(self, element):
        if element in self.filtered_columns.index:
            properties = self.filtered_columns.loc[element].to_dict()

            for key in properties:
                if pd.isna(properties[key]):
                    properties[key] = 0

            state_encoded = self.one_hot_encoders['StandardState'].transform([[str(properties.get('StandardState', 'Unknown'))]])[0]
            group_encoded = self.one_hot_encoders['GroupBlock'].transform([[str(properties.get('GroupBlock', 'Unknown'))]])[0]
            oxi_encoded = self.encode_oxidation_states(properties.get('OxidationStates', ''))
            
            return {
                'float_properties': [
                    properties['AtomicMass'], properties['Electronegativity'], properties['AtomicRadius'],
                    properties['IonizationEnergy'], properties['ElectronAffinity'], properties['MeltingPoint'],
                    properties['BoilingPoint'], properties['Density'], properties['YoungModulus'], 
                    properties['BulkModulus'], properties['ShearModulus'], properties['PoissonRatio']
                ],
                'state_encoded': state_encoded.tolist(),
                'oxi_encoded': oxi_encoded,
                'group_block_encoded': group_encoded.tolist(),
                'modulus_indicators': [
                    properties['has_young_modulus'],
                    properties['has_bulk_modulus'],
                    properties['has_shear_modulus'],
                    properties['has_poisson_ratio']
                ]
            }
        else:
            return None

    def get_electronegativity(self, element):
        if element in self.filtered_columns.index:
            properties = self.filtered_columns.loc[element].to_dict()
            en = properties['Electronegativity']
        return en

class MaterialDataset:
    def __init__(self, element_data_processor, summary_filepath, elasticity_filepath):
        self.nn_finder = MinimumDistanceNN()
        self.element_data_processor = element_data_processor
        self.summary_filepath = summary_filepath
        self.elasticity_filepath = elasticity_filepath
        self.summary_dataset = None
        self.elasticity_dataset = None
        
        self.structure = None
        
        self.crystal_system_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.point_group_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.cs_len = 0
        self.pg_len = 0
        
        self.equilibrium_reaction_energy_mean = -0.14790047661889588
        self.sound_velocity_transverse_mean = 2627.382382167975
        self.sound_velocity_longitudinal_mean = 4829.860788248521
        self.thermal_conductivity_clarke_mean = 0.8098828291077165
        self.thermal_conductivity_cahill_mean = 0.9043483163173393
        self.debye_temperature_mean = 347.1797212469588

        self.global_features_mean = None
        self.global_features_std = None
        self.tensor_feature_len = None
        self.n_stat_global = None

        self.bond_length_mean = None
        self.bond_length_std = None
        self.en_difference_mean = None
        self.en_difference_std = None

        self.max_bonds = None
        self.max_angles = None
        
        self.feature_mapping = []

    def flatten(self, nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(self.flatten(item))
            else:
                flat_list.append(item)
        return flat_list

    def replace_none_with_mean(self, features):
        features_array = np.array(features, dtype=object)
        
        if features_array.shape[0] == 0 or len(features_array.shape) < 2:
            raise ValueError("Features array is empty or has incorrect dimensions.")
        
        column_means = []
        for col in range(features_array.shape[1]):
            col_values = [x for x in features_array[:, col] if x is not None]
            col_mean = np.mean(col_values) if col_values else 0
            column_means.append(col_mean)
        
        for row_idx, row in enumerate(features_array):
            for col_idx, value in enumerate(row):
                if value is None:
                    features_array[row_idx, col_idx] = column_means[col_idx]
        
        return features_array.astype(float)

    def fetch_dataset(self):
        with open(self.summary_filepath, "r") as file:
            self.summary_dataset = json.load(file)
        with open(self.elasticity_filepath, "r") as file:
            self.elasticity_dataset = json.load(file)

        crystal_system_values = [[material['symmetry'].get('crystal_system', '')] for material in self.summary_dataset]
        point_group_values = [[material['symmetry'].get('point_group', '')] for material in self.summary_dataset]

        self.crystal_system_encoder.fit(crystal_system_values)
        self.point_group_encoder.fit(point_group_values)

        self.cs_len = len(self.crystal_system_encoder.categories_[0]) if self.crystal_system_encoder.categories_ else 0
        self.pg_len = len(self.point_group_encoder.categories_[0]) if self.point_group_encoder.categories_ else 0

        self.max_bonds, self.max_angles = self.max_bonds_and_angles()

    def disp_pbc(self, x_i, x_j, image, lattice):
        return (x_j - x_i) + (np.asarray(image, dtype = float) @ lattice)

    def unique_neighbors(self, structure, idx):
        infos = self.nn_finder.get_nn_info(structure, idx)
        best = {}
        for n in infos:
            j = int(n['site_index'])
            d = float(n['weight'])
            if (j not in best) or (d < best[j]['weight']):
                best[j] = {'site_index': j, 'weight': d, 'image': tuple(int(x) for x in n['image'])}

        best.pop(idx, None)
        neighbors = list(best.values())
        neighbors.sort(key = lambda n: n['weight'])
        return neighbors

    def max_bonds_and_angles(self):
        max_bonds = 0
        max_angles = 0

        for material_data in self.summary_dataset:
            structure = Structure.from_dict(material_data['structure'])

            for idx, _ in enumerate(structure):
                neighbors = self.unique_neighbors(structure, idx)
                deg = len(neighbors)
                max_bonds = max(max_bonds, deg)
                max_angles = max(max_angles, deg * (deg - 1) // 2)

        print(f"Maximum bonds per atom: {max_bonds}, Maximum angles per atom: {max_angles}")
        return max_bonds, max_angles

    def edge_features_stats(self):
        all_bond_lengths = []
        all_en_diffs = []

        for material_data in self.summary_dataset:
            structure = Structure.from_dict(material_data['structure'])
            lattice = structure.lattice.matrix
            coords = np.array([site.coords for site in structure], dtype = float)

            for idx, site in enumerate(structure):
                element_i = str(site.specie)
                en_i = self.element_data_processor.get_electronegativity(element_i)

                for n in self.unique_neighbors(structure, idx):
                    j = n['site_index']
                    element_j = str(structure[j].specie)
                    en_j = self.element_data_processor.get_electronegativity(element_j)

                    d_ij = self.disp_pbc(coords[idx], coords[j], n['image'], lattice)
                    r_ij = float(np.linalg.norm(d_ij))
                    en_diff = float(abs(en_i-en_j))
                    
                    all_bond_lengths.append(r_ij)
                    all_en_diffs.append(en_diff)

        self.bond_length_mean = float(np.mean(all_bond_lengths))
        self.bond_length_std = float(np.std(all_bond_lengths))
        self.en_difference_mean = float(np.mean(all_en_diffs))
        self.en_difference_std = float(np.std(all_en_diffs))

        print(f'Bond length mean: {self.bond_length_mean:.4f}, std: {self.bond_length_std:.4f}')
        print(f'EN difference mean: {self.en_difference_mean:.4f}, std: {self.en_difference_std:.4f}')

    def global_feature_stats(self):
        all_features = []
        for idx in range(len(self.summary_dataset)):
            try:
                features = self.extract_global_features(idx)
                all_features.append(features)
            except Exception as e:
                print(f"Error at idx {idx}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid global features extracted.")

        all_features = self.replace_none_with_mean(all_features)

        self.global_features_mean = np.mean(all_features, axis=0).astype(float)
        self.global_features_std = np.std(all_features, axis=0).astype(float)

        base_cont = 8 + 1 + 2 + 8 
        if self.tensor_feature_len is None:
            raise ValueError("tensor_feature_len is None")
        n_stat_layout = base_cont + self.tensor_feature_len

        cs_len = int(self.cs_len)
        pg_len = int(self.pg_len)
        tail_len = 2 + cs_len + pg_len + 1

        total_len = int(all_features.shape[1])
        n_stat_sub = total_len - tail_len

        self.n_stat_global = max(0, min(n_stat_sub, total_len))
        print(f"Global features: total={total_len}, normalize_first={self.n_stat_global}")

    def normalize_global_features(self, features):
        if self.global_features_mean is None or self.global_features_std is None:
            raise ValueError("Global feature statistics not computed.")
        if self.n_stat_global is None:
            raise ValueError("n_stat_global is not set.")
        
        normalized = list(features)
        for i in range(self.n_stat_global):
            mean = float(self.global_features_mean[i])
            std = float(self.global_features_std[i])
            val = float(features[i])
            normalized[i] = (val - mean) / std if std != 0.0 else 0.0

        normalized = [float(f) for f in normalized]
        return normalized

    def extract_global_features_final(self, index):
        raw_features = self.extract_global_features(index)
        norm_features = self.normalize_global_features(raw_features)
        return norm_features

    def extract_global_features(self, index):
        material_data_summary = self.summary_dataset[index]
        material_data_elasticity = self.elasticity_dataset[index]
        
        global_features_summary_keys = [
            'nsites', 'nelements', 'volume', 'density', 'density_atomic', 'energy_per_atom', 
            'formation_energy_per_atom','energy_above_hull'
        ]
        
        global_features = [material_data_summary.get(key, 0) if material_data_summary.get(key) is not None 
                                else 0 for key in global_features_summary_keys]
        
        equilibrium_reaction_energy_per_atom = (material_data_summary.get('equilibrium_reaction_energy_per_atom') 
                                        if material_data_summary.get('equilibrium_reaction_energy_per_atom') is not None 
                                        else self.equilibrium_reaction_energy_mean)
        global_features.extend([equilibrium_reaction_energy_per_atom])

        global_features_elasticity_keys = ['order', 'universal_anisotropy']
        for key in global_features_elasticity_keys:
            value = material_data_elasticity.get(key, 0)
            global_features.append(value)
        
        bulk_modulus = material_data_elasticity.get('bulk_modulus', {}).get('vrh')
        shear_modulus = material_data_elasticity.get('shear_modulus', {}).get('vrh')
        bulk_modulus = float(bulk_modulus) if bulk_modulus is not None else 0.0
        shear_modulus = float(shear_modulus) if shear_modulus is not None else 0.0
        denom = 3.0 * bulk_modulus + shear_modulus
        young_modulus = (9.0 * bulk_modulus * shear_modulus / denom) if denom != 0.0 else 0.0

        sound_velocity = material_data_elasticity.get('sound_velocity', {})
        if isinstance(sound_velocity, dict):
            sound_velocity_transverse = float(sound_velocity.get('transverse', self.sound_velocity_transverse_mean))
            sound_velocity_longitudinal = float(sound_velocity.get('longitudinal', self.sound_velocity_longitudinal_mean))
        else:
            sound_velocity_transverse = float(self.sound_velocity_transverse_mean)
            sound_velocity_longitudinal = float(self.sound_velocity_longitudinal_mean)

        thermal_conductivity_clarke = material_data_elasticity.get('thermal_conductivity', {}).get('clarke', self.thermal_conductivity_clarke_mean)
        thermal_conductivity_cahill = material_data_elasticity.get('thermal_conductivity', {}).get('cahill', self.thermal_conductivity_cahill_mean)
        debye_temperature = material_data_elasticity.get('debye_temperature', self.debye_temperature_mean)

        global_features.extend([
            bulk_modulus, shear_modulus, young_modulus, float(sound_velocity_transverse), float(sound_velocity_longitudinal), 
            float(thermal_conductivity_clarke), float(thermal_conductivity_cahill), float(debye_temperature)
        ])

        tensors, feature_mapping_1 = self.global_tensor_features(index)
        global_features.extend(tensors)
        if self.tensor_feature_len is None:
            self.tensor_feature_len = len(tensors)
            self.feature_mapping = feature_mapping_1

        is_stable = 1 if material_data_summary.get('is_stable') else 0
        is_metal = 1 if material_data_summary.get('is_metal') else 0
        global_features.extend([is_stable, is_metal])

        symmetry_data = material_data_summary.get('symmetry', {})
        crystal_system_encoded = self.crystal_system_encoder.transform([[symmetry_data.get('crystal_system', '')]])[0]
        point_group_encoded = self.point_group_encoder.transform([[symmetry_data.get('point_group', '')]])[0]
        global_features.extend(crystal_system_encoded)
        global_features.extend(point_group_encoded)

        has_equilibrium_reaction_energy = 1 if (material_data_summary.get('equilibrium_reaction_energy_per_atom') is not None) else 0
        global_features.extend([has_equilibrium_reaction_energy])

        return global_features
                
    def global_tensor_features(self, index):
        material_data_elasticity = self.elasticity_dataset[index]
        
        fitting_data_tensors = ["cauchy_stresses", "deformations", "strains"]
        features = []
        feature_mapping_1 = []

        for tensor_name_1 in fitting_data_tensors:    
            eigenvalues_list_fit = []
            frobenius_norms_fit = []
            determinants_fit = []
            trace_strains = []

            fitting_data_tensor_set = material_data_elasticity.get('fitting_data').get(tensor_name_1, [])
            
            for tensor in fitting_data_tensor_set:
                tensor_3x3 = np.array(tensor,dtype=float)

                eigenvalues_list_fit.extend(np.real(np.linalg.eigvals(tensor_3x3)))
                frobenius_norms_fit.append(np.linalg.norm(tensor_3x3, ord='fro'))
                
                if tensor_name_1 == 'strains':
                    trace_strains.append(np.trace(tensor_3x3))
                else:
                    determinants_fit.append(np.linalg.det(tensor_3x3))

            features.extend([
                np.mean(eigenvalues_list_fit) if eigenvalues_list_fit else 0.0,
                np.max(eigenvalues_list_fit) if eigenvalues_list_fit else 0.0,
                np.min(eigenvalues_list_fit) if eigenvalues_list_fit else 0.0,
                np.mean(frobenius_norms_fit) if frobenius_norms_fit else 0.0,
                np.max(frobenius_norms_fit) if frobenius_norms_fit else 0.0,
                np.min(frobenius_norms_fit) if frobenius_norms_fit else 0.0])
            
            feature_mapping_1.extend([
            f"{tensor_name_1}_eigenvalues_mean",
            f"{tensor_name_1}_eigenvalues_max",
            f"{tensor_name_1}_eigenvalues_min",
            f"{tensor_name_1}_frobenius_mean",
            f"{tensor_name_1}_frobenius_max",
            f"{tensor_name_1}_frobenius_min"
            ])

            if tensor_name_1 == 'strains':
                features.extend([
                    np.mean(trace_strains) if trace_strains else 0.0,
                    np.max(trace_strains) if trace_strains else 0.0,
                    np.min(trace_strains) if trace_strains else 0.0
                ])
                feature_mapping_1.extend([
                    f"{tensor_name_1}_trace_mean",
                    f"{tensor_name_1}_trace_max",
                    f"{tensor_name_1}_trace_min"
                ])
            else:
                features.extend([
                    np.mean(determinants_fit) if determinants_fit else 0.0,
                    np.max(determinants_fit) if determinants_fit else 0.0,
                    np.min(determinants_fit) if determinants_fit else 0.0
                ])
                feature_mapping_1.extend([
                    f"{tensor_name_1}_determinant_mean",
                    f"{tensor_name_1}_determinant_max",
                    f"{tensor_name_1}_determinant_min"
                ])

        tensors_to_extract = {
        'elastic_tensor': 'raw',
        'compliance_tensor': 'raw'
        }
        for tensor_key, tensor_name_2 in tensors_to_extract.items():

            eigenvalues_list_ela = []
            frobenius_norms_ela = []
            determinants_ela = []
            tensor = np.array(material_data_elasticity.get(tensor_key).get('raw'), dtype=float)

            eigenvalues_list_ela.extend(np.real(np.linalg.eigvals(tensor)))
            frobenius_norms_ela.append(np.linalg.norm(tensor, ord='fro'))
            determinants_ela.append(np.linalg.det(tensor))

            features.extend([
                np.mean(eigenvalues_list_ela) if eigenvalues_list_ela else 0.0,
                np.max(eigenvalues_list_ela) if eigenvalues_list_ela else 0.0,
                np.min(eigenvalues_list_ela) if eigenvalues_list_ela else 0.0,
                np.mean(frobenius_norms_ela) if frobenius_norms_ela else 0.0,
                np.max(frobenius_norms_ela) if frobenius_norms_ela else 0.0,
                np.min(frobenius_norms_ela) if frobenius_norms_ela else 0.0,
                np.mean(determinants_ela) if determinants_ela else 0.0,
                np.max(determinants_ela) if determinants_ela else 0.0,
                np.min(determinants_ela) if determinants_ela else 0.0,
            ])

            feature_mapping_1.extend([
                    f"{tensor_name_2}_eigenvalues_mean",
                    f"{tensor_name_2}_eigenvalues_max",
                    f"{tensor_name_2}_eigenvalues_min",
                    f"{tensor_name_2}_frobenius_mean",
                    f"{tensor_name_2}_frobenius_max",
                    f"{tensor_name_2}_frobenius_min",
                    f"{tensor_name_2}_determinant_mean",
                    f"{tensor_name_2}_determinant_max",
                    f"{tensor_name_2}_determinant_min",
                ])

        features = self.flatten(features)
        return features, feature_mapping_1
    
    def _voigt21_from_raw(self, elastic_raw):
        if elastic_raw is None:
            return [0.0] * 21
        arr = np.array(elastic_raw, dtype=float)
        if arr.shape != (6, 6):
            flat = arr.reshape(-1).tolist()
            if len(flat) >= 21:
                return flat[:21]
            return flat + [0.0] * (21 - len(flat))
        
        idxs = [
            (0,0), (0,1), (0,2), (0,3), (0,4), (0,5),
            (1,1), (1,2), (1,3), (1,4), (1,5),
            (2,2), (2,3), (2,4), (2,5),
            (3,3), (3,4), (3,5),
            (4,4), (4,5),
            (5,5)
        ]
        return [float(arr[i,j]) for (i,j) in idxs]
    
    def _get_scalar_targets(self, material_data_summary, material_data_elasticity):
        bulk_modulus = material_data_elasticity.get('bulk_modulus', {}).get('vrh')
        shear_modulus = material_data_elasticity.get('shear_modulus', {}).get('vrh')
        bulk_modulus = float(bulk_modulus) if bulk_modulus is not None else 0.0
        shear_modulus = float(shear_modulus) if shear_modulus is not None else 0.0
        denom = 3.0 * bulk_modulus + shear_modulus
        young_modulus = (9.0 * bulk_modulus * shear_modulus / denom) if denom != 0.0 else 0.0

        sound_velocity = material_data_elasticity.get('sound_velocity', {})
        if isinstance(sound_velocity, dict):
            sound_velocity_transverse = float(sound_velocity.get('transverse', self.sound_velocity_transverse_mean))
            sound_velocity_longitudinal = float(sound_velocity.get('longitudinal', self.sound_velocity_longitudinal_mean))
        else:
            sound_velocity_transverse = float(self.sound_velocity_transverse_mean)
            sound_velocity_longitudinal = float(self.sound_velocity_longitudinal_mean)

        thermal_conductivity_clarke = material_data_elasticity.get('thermal_conductivity', {}).get('clarke', self.thermal_conductivity_clarke_mean)
        thermal_conductivity_cahill = material_data_elasticity.get('thermal_conductivity', {}).get('cahill', self.thermal_conductivity_cahill_mean)
        debye_temperature = material_data_elasticity.get('debye_temperature', self.debye_temperature_mean)

        return [
            bulk_modulus,
            shear_modulus,
            young_modulus,
            sound_velocity_transverse,
            sound_velocity_longitudinal,
            float(thermal_conductivity_clarke),
            float(thermal_conductivity_cahill),
            float(debye_temperature)
        ]
    
    def analyze_structure(self, index):
        node_features = []
        angle_attr = []
        angle_triplets = []

        edges_map = {}

        bond_type_one_hot = {
            "metallic": [1, 0, 0, 0],
            "ionic": [0, 1, 0, 0],
            "polar covalent": [0, 0, 1, 0],
            "nonpolar covalent": [0, 0, 0, 1],
        }

        material_data = self.summary_dataset[index]
        self.structure = Structure.from_dict(material_data['structure'])
        is_metal = bool(material_data.get('is_metal', False))

        lattice = self.structure.lattice.matrix
        coords = np.array([site.coords for site in self.structure], dtype=float)

        for site in self.structure:
            element = str(site.specie)
            props = self.element_data_processor.get_element_properties(element)
            if props:
                raw = (
                    props['float_properties']
                    + props['state_encoded']
                    + props['oxi_encoded']
                    + props['group_block_encoded']
                    + props['modulus_indicators']
                )
                node_features.append(self.element_data_processor.normalize_node_features(raw))

        for i, _ in enumerate(self.structure):
            neighbors = self.unique_neighbors(self.structure, i)

            disp_by_j = {}
            for n in neighbors:
                j = int(n['site_index'])
                d_ij = self.disp_pbc(coords[i], coords[j], n['image'], lattice)
                disp_by_j[j] = (d_ij, float(np.linalg.norm(d_ij)))

                u, v = (i, j) if i < j else (j, i)
                if u != v and (u, v) not in edges_map:
                    en_i = self.element_data_processor.get_electronegativity(str(self.structure[u].specie))
                    en_v = self.element_data_processor.get_electronegativity(str(self.structure[v].specie))
                    en_diff = abs(en_i - en_v)

                    bond_type = (
                        "metallic" if is_metal
                        else "ionic" if en_diff >= 1.7
                        else "polar covalent" if en_diff >= 0.4
                        else "nonpolar covalent"
                    )
                    edges_map[(u, v)] = {
                        "length": float(np.linalg.norm(d_ij)),
                        "en_diff": float(en_diff),
                        "onehot": bond_type_one_hot[bond_type],
                    }

            nbr_ids = list(disp_by_j.keys())
            for a in range(len(nbr_ids)):
                j = nbr_ids[a]
                d_ij, rij = disp_by_j[j]
                if rij == 0.0:
                    continue
                for b in range(a + 1, len(nbr_ids)):
                    k = nbr_ids[b]
                    d_ik, rik = disp_by_j[k]
                    if rik == 0.0:
                        continue
                    cos_th = float(np.dot(d_ij, d_ik) / (rij * rik))
                    theta = float(np.arccos(np.clip(cos_th, -1.0, 1.0)))
                    angle_triplets.append([i, j, k])
                    angle_attr.append(theta)

        edge_index = []
        edge_attr = []

        bl_mu, bl_std = self.bond_length_mean, self.bond_length_std
        ed_mu, ed_std = self.en_difference_mean, self.en_difference_std

        for (u, v), info in edges_map.items():
            r = info["length"]
            en = info["en_diff"]
            onehot = info["onehot"]

            norm_len = (r - bl_mu) / bl_std if (bl_std and bl_std != 0.0) else 0.0
            norm_en  = (en - ed_mu) / ed_std if (ed_std and ed_std != 0.0) else 0.0
            ef = onehot + [float(norm_len), float(norm_en)]

            edge_index.append([u, v]); edge_attr.append(ef)
            edge_index.append([v, u]); edge_attr.append(ef)

        x                   = torch.tensor(node_features, dtype=torch.float)
        pos                 = torch.tensor(coords, dtype=torch.float)
        edge_index          = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr       = torch.tensor(edge_attr, dtype=torch.float)
        angle_attr     = torch.tensor(angle_attr, dtype=torch.float)
        angle_triplets = torch.tensor(angle_triplets, dtype=torch.long)

        return Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            angle_attr=angle_attr,
            angle_triplets=angle_triplets
        )

    def process_all_materials(self, output_dir = "processed_data"):
        os.makedirs(output_dir, exist_ok = True)

        all_materials_data = []

        for index in range(len(self.summary_dataset)):
            try:
                if (index + 1) % 100 == 0:
                    print(f"Processing material {index+1} / {len(self.summary_dataset)}")
                
                material_data_summary = self.summary_dataset[index]
                material_data_elasticity = self.elasticity_dataset[index]

                global_features = self.extract_global_features_final(index)

                graph_data = self.analyze_structure(index)

                targets_scalars = self._get_scalar_targets(material_data_summary, material_data_elasticity)

                elastic_tensor_raw = material_data_elasticity.get('elastic_tensor', {}).get('raw', None)
                targets_voigt21 = self._voigt21_from_raw(elastic_tensor_raw)

                crystal_system = material_data_summary.get('symmetry', {}).get('crystal_system', "")
                target_num_elements = int(material_data_summary.get('nelements', 0) or 0)

                material_id = material_data_summary.get('material_id', None)
                is_stable = 1 if material_data_summary.get('is_stable') else 0
                    
                material_data = {
                    "node_features": graph_data.x.tolist(),
                    "positions": graph_data.pos.tolist(),  
                    "edge_attr": graph_data.edge_attr.tolist(),  
                    "edge_index": graph_data.edge_index.tolist(),
                    "angle_attr": graph_data.angle_attr.tolist(),
                    "angle_triplets": graph_data.angle_triplets.tolist(),
                    "global_features": global_features,
                    
                    "num_nodes": int(graph_data.x.size(0)),
                    "num_edges": int(graph_data.edge_index.size(1)),
                    "num_angles": int(graph_data.angle_attr.size(0)),
                    "targets_scalars": targets_scalars,
                    "targets_voigt21": targets_voigt21,
                    "crystal_system": crystal_system,
                    "target_num_elements": target_num_elements,
                    "nelements": target_num_elements,
                    "is_stable": is_stable
                }
                if material_id is not None:
                    material_data["material_id"] = material_id
                    
                all_materials_data.append(material_data)

            except Exception as e:
                print(f"Error processing material at index {index}: {e}")

        with open(os.path.join(output_dir, "all_materials_data.json"), "w") as f:
            json.dump(all_materials_data, f, indent=4)

        print(f"\n✓ Processing complete! Saved {len(all_materials_data)} materials")
        print(f"✓ Output: {output_dir}/all_materials_data.json")


if __name__ == '__main__':
    print("="*80)
    print("PROCESSING FILTERED MATERIALS PROJECT DATA FOR PHASE 2")
    print("="*80)
    print()
    
    # Initialize element processor
    print("Loading element data...")
    processor = ElementDataProcessor("datasets/PubChemElements_all.csv")
    processor.load_data()
    processor.columns()
    processor.means_stds()
    print("✓ Element data loaded\n")

    # Initialize material dataset with FILTERED files
    print("Loading filtered MP data...")
    mpr = MaterialDataset(
        processor, 
        "datasets/mp_summary_filtered.json",
        "datasets/mp_elasticity_filtered.json"
    )
    
    print("Fetching datasets...")
    mpr.fetch_dataset()
    print(f"✓ Loaded {len(mpr.summary_dataset)} filtered materials\n")
    
    print("Computing statistics...")
    mpr.edge_features_stats()
    print()
    mpr.global_feature_stats()
    print()
    
    print("Processing all materials into graph format...")
    print("(This will take 10-15 minutes)\n")
    mpr.process_all_materials(output_dir="processed_data_filtered")
    
    print()
    print("="*80)
    print("SUCCESS! PHASE 2 DATA READY")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Normalize: python normalize_data.py")
    print("     (modify to use processed_data_filtered/all_materials_data.json)")
    print()
    print("  2. Train Phase 2:")
    print("     python train.py --data processed_data_filtered/all_materials_data_normalized.json")
