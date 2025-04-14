#!/usr/bin/env python3
"""
Protein Binding Site Prediction

This script:
    1. Reads one or more protein structures in PDB format
    2. Extracts features using Feature_Extraction_01.py module
    3. Applies a pre-trained random forest model
    4. Identifies predicted active site residues
    5. Outputs visualization files for chimera and PyMOL, 
    a summary of the predicted binding sites,
    and a PDB file containing only the predicted binding sites' residues.

Supports processing:
    - Single PDB file
    - Multiple PDB files
    - A directory containing PDB files
    - A list of PDB files from a text file
"""

import sys
import os
import numpy as np
import pandas as pd
import glob
import Feature_Extraction_01 as fe
import Categorical_Data_Transformation_02_2 as transf
from Bio.PDB import PDBParser, PDBIO, Select
from argparse import ArgumentParser
from joblib import load
from sklearn.preprocessing import StandardScaler

class PredictedSiteSelector(Select):
    """Select the residues identified as active sites in the prediction."""
    
    def __init__(self, predicted_residues):
        """Args: predicted_residues: list of tuples (chain_id, residue_id) of predicted active site residues."""
        self.predicted_residues = predicted_residues
        
    def accept_residue(self, residue):
        """Return True if the residue is in the predicted active sites list."""
        chain_id = residue.get_parent().id
        res_id = residue.id[1]
        return (chain_id, res_id) in self.predicted_residues


def extract_features_from_structure(structure, pdb_path):
    """Extract features from a pdb using Feature_Extraction_01 module functions;
    returns a dataframe with the extracted features and a list of dictionaries with metadata for each residue."""

    print("Extracting secondary structure and ASA...")
    sec_struct_dict, asa_dict = fe.extract_secondary_structure(pdb_path, structure)
    
    print("Calculating curvature...")
    curvature_dict = fe.calculate_curvature(structure)
    
    print("Finding structural neighbors...")
    neighbors_dict, contact_freq_dict = fe.get_structural_neighbors(structure)
    
    print("Calculating B-factors and polar density...")
    bfactor_dict, polar_density_dict = fe.calculate_bfactor_and_polar_density(structure)
    
    print("Checking for MSMS and calculating residue depths...")
    try:
        depth_dict = fe.calculate_residue_depths(structure, pdb_path)
    except Exception as e:
        print(f"Warning: Couldn't calculate residue depth: {e}")
        print("Continuing without depth information...")
        depth_dict = {}
    
    residues = fe.parse_pdb(pdb_path)
    
    if not residues:
        raise ValueError(f"No residues found in the structure.")
    
    feature_rows = []
    features_metadata = []
    
    for position, residue, chain in residues:
        properties = fe.get_residue_properties(residue)
        
        sec_struct = "Unknown"
        if chain in sec_struct_dict and position in sec_struct_dict[chain]:
            sec_struct = sec_struct_dict[chain][position]
        
        asa = None
        if chain in asa_dict and position in asa_dict[chain]:
            asa = asa_dict[chain][position]
        
        depth = None
        if chain in depth_dict and position in depth_dict[chain]:
            depth = depth_dict[chain][position]
        
        curvature = None
        if chain in curvature_dict and position in curvature_dict[chain]:
            curvature = curvature_dict[chain][position]
        
        neighbors = ""
        if chain in neighbors_dict and position in neighbors_dict[chain]:
            neighbors = neighbors_dict[chain][position]
        
        contact_freq = None
        if chain in contact_freq_dict and position in contact_freq_dict[chain]:
            contact_freq = contact_freq_dict[chain][position]
        
        bfactor = None
        if chain in bfactor_dict and position in bfactor_dict[chain]:
            bfactor = bfactor_dict[chain][position]
        
        polar_density = None
        if chain in polar_density_dict and position in polar_density_dict[chain]:
            polar_density = polar_density_dict[chain][position]
        
        feature_row = {
            "Hydrophobicity": properties.get("Hydrophobicity", "NA"),
            "NormalizedVDWV": properties.get("NormalizedVDWV", "NA"),
            "Polarity": properties.get("Polarity", "NA"),
            "Charge": properties.get("Charge", "NA"),
            "SecondaryStr": properties.get("SecondaryStr", "NA"),
            "SolventAccessibility": properties.get("SolventAccessibility", "NA"),
            "Polarizability": properties.get("Polarizability", "NA"),
            "H-Bond_Propensity": properties.get("H-Bond_Propensity", "NA"),
            "Secondary_Structure": sec_struct,
            "ASA": asa if asa is not None else 0.0,
            "Surface_Curvature": curvature if curvature is not None else 0.0,
            "Contact_Frequency": contact_freq if contact_freq is not None else 0,
            "Bfactor": bfactor if bfactor is not None else 0.0,
            "Neighbors": neighbors,
            "Polar_Density": polar_density if polar_density is not None else 0.0,
            "Residue_Depth": depth if depth is not None else 0.0,
        }
        
        metadata = {
            "chain_id": chain,
            "residue_id": position,
            "residue_type": residue
        }
        
        feature_rows.append(feature_row)
        features_metadata.append(metadata)
    
    features_df = pd.DataFrame(feature_rows)
    
    categorical_columns = [
        "Hydrophobicity", "NormalizedVDWV", "Polarity", "Charge", 
        "SecondaryStr", "SolventAccessibility", "Polarizability", 
        "H-Bond_Propensity", "Secondary_Structure"
    ]
    
    cat_cols_exist = [col for col in categorical_columns if col in features_df.columns]
    
    features_df = pd.get_dummies(features_df, columns=cat_cols_exist, drop_first=False)
    
    return features_df, features_metadata

def get_pdb_files(input_paths):
    """Return a list of absolute paths to PDB files from various input types:
    list of files or directory paths or a file containing a list of paths."""

    pdb_files = []
    
    for path in input_paths:
        if os.path.isdir(path):
            pdb_files.extend(glob.glob(os.path.join(path, "*.pdb")))
            pdb_files.extend(glob.glob(os.path.join(path, "*.ent")))
        
        elif (path.lower().endswith(".pdb") or path.lower().endswith(".ent")) and os.path.isfile(path):
            pdb_files.append(os.path.abspath(path))
        
        elif os.path.isfile(path) and not (path.lower().endswith(".pdb") or path.lower().endswith(".ent")):
            try:
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and os.path.isfile(line) and (line.lower().endswith(".pdb") or line.lower().endswith(".ent")):
                            pdb_files.append(os.path.abspath(line))
            except Exception as e:
                print(f"Warning: Error reading file list from {path}: {e}")
    
    return pdb_files

def process_pdb_file(pdb_file, cleaned_pdb, model_path, skip_depth):
    """Process a single PDB file and return the csv with the extracted features.
    Arguments:
        pdb_file: original path to the PDB file.
        cleaned_pdb: path to the cleaned PDB file.
        model_path: path to the trained model.
        skip_depth: to skip residue depth calculation."""
    
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    output_prefix = base_name
    
    print(f"\n{'='*80}")
    print(f"Processing PDB file: {pdb_file}")
    print(f"Using cleaned file: {cleaned_pdb}")
    print(f"Output prefix: {output_prefix}")
    print(f"{'='*80}\n")
    
    print(f"Reading protein structure...")
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', cleaned_pdb)
    except FileNotFoundError:
        print(f"Error: PDB file '{cleaned_pdb}' not found.")
        return
    except Exception as e:
        print(f"Error reading PDB structure: {e}")
        return
    
    print("Extracting features...")
    try:
        features_df, features_metadata = extract_features_from_structure(structure, cleaned_pdb)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return
    
    #### Dealing with non-numerical values from Secondary Structure column
    Second_strct_values = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
    Second_strct_dict = {value: index for index, value in enumerate(Second_strct_values)}

    if 'Secondary_Structure' in features_df.columns:
        features_df['Secondary_Structure'] = features_df['Secondary_Structure'].map(Second_strct_dict)
        features_df['Secondary_Structure'] = features_df['Secondary_Structure'].fillna(7)       

    #### Dealing with non-numerical values from Neighbor column
    result_df = transf.extract_single_neighbor_feature(features_df)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(result_df)  
    X_scaled_df = pd.DataFrame(X_scaled, columns=result_df.columns)
    
    print(f"Loading model from {model_path}...")
    try:
        model_data = load(model_path)
        if isinstance(model_data, tuple):
            model = model_data[0]  
        else:
            model = model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Making predictions...")
    try:
        expected_features = ['ASA', 'Surface_Curvature', 'Contact_Frequency', 'Bfactor',
                            'Polar_Density', 'Residue_Depth', 'Hydrophobicity_1', 'Hydrophobicity_2', 
                            'Hydrophobicity_3', 'NormalizedVDWV_1', 'NormalizedVDWV_2', 'NormalizedVDWV_3', 
                            'Polarity_1', 'Polarity_2', 'Polarity_3', 'Charge_1']

        X_scaled_df = X_scaled_df[expected_features]

        predictions = model.predict_proba(X_scaled_df)
        predicted_indices = np.where(predictions[:, 1] > 0.5)[0]
    
    except Exception as e:
        print(f"Error making predictions: {e}")
        return
    
    predicted_residues = []
    for idx in predicted_indices:
        chain_id = features_metadata[idx]['chain_id']
        res_id = features_metadata[idx]['residue_id']
        predicted_residues.append((chain_id, res_id))
    
    print(f"Identified {len(predicted_residues)} residues in predicted sites")
    
    print(f"Generating output files with prefix {output_prefix}...")
    
    io = PDBIO()
    io.set_structure(structure)
    selector = PredictedSiteSelector(predicted_residues)
    io.save(f"{output_prefix}_sites.pdb", selector)
    
    # Create PyMOL script
    with open(f"{output_prefix}.pml", 'w') as f:
        f.write(f"load {output_prefix}_sites.pdb, predicted_sites\n")
        f.write(f"load {cleaned_pdb}, original\n")
        f.write("show cartoon, original\n")
        f.write("color gray, original\n")
        f.write("show sticks, predicted_sites\n")
        f.write("color red, predicted_sites\n")
        f.write("zoom predicted_sites\n")
    
    # Create Chimera script
    with open(f"{output_prefix}.cmd", 'w') as f:
        f.write(f"open {cleaned_pdb}\n")
        f.write(f"open {output_prefix}_sites.pdb\n")
        f.write("style #0 cartoon\n")
        f.write("color #0 gray\n")
        f.write("style #1 stick\n")
        f.write("color #1 red\n")
        f.write("focus\n")
    
    # Generate summary text file
    residue_names = {}
    for chain_id, res_id in predicted_residues:
        for chain in structure.get_chains():
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[1] == res_id:
                        aa_name = residue.get_resname()
                        position = f"{chain_id}:{res_id}"
                        residue_names[position] = aa_name
    
    with open(f"{output_prefix}_summary.txt", 'w') as f:
        f.write("Predicted Active Site Residues:\n")
        f.write("------------------------------\n")
        for position, aa_name in sorted(residue_names.items()):
            f.write(f"{position} - {aa_name}\n")
        
        f.write("\nPredicted Sites (grouped by chain):\n")
        f.write("--------------------------------\n")
        
        by_chain = {}
        for chain_id, res_id in predicted_residues:
            if chain_id not in by_chain:
                by_chain[chain_id] = []
            by_chain[chain_id].append(res_id)
        
        site_num = 1
        for chain_id, residues in by_chain.items():
            f.write(f"Site {site_num} (Chain {chain_id}): ")
            residue_strings = [f"{residue_names.get(f'{chain_id}:{r}')} {r}" for r in sorted(residues)]
            f.write(", ".join(residue_strings))
            f.write("\n")
            site_num += 1
    
    print(f"Analysis complete. Results saved with prefix '{output_prefix}'")


def main():
    """Main function to run the protein analysis pipeline."""

    parser = ArgumentParser(description="Predict active sites in protein structure")
    parser.add_argument("input", nargs='+', help="Input PDB file(s), directory of PDB files, or file containing list of PDB files")
    parser.add_argument("--model", default="model.pkl", help="Trained random forest model file")
    parser.add_argument("--skip-depth", action="store_true", help="Skip residue depth calculation")
    parser.add_argument("--clean-dir", default="cleaned_pdbs", help="Directory to store cleaned PDB files")
    args = parser.parse_args()
    
    pdb_files = get_pdb_files(args.input)
    
    if not pdb_files:
        print("Error: No valid PDB files found.")
        sys.exit(1)
    
    print(f"Found {len(pdb_files)} PDB files.")
    
    print("Cleaning all PDB files...")
    os.makedirs(args.clean_dir, exist_ok=True)
    
    cleaned_pdbs = {}
    
    unique_dirs = set(os.path.dirname(pdb_file) or "." for pdb_file in pdb_files)
  
    for dir_path in unique_dirs:
        try:
            cleaned_files = fe.clean_pdb_files(dir_path, args.clean_dir)
            for orig_file in pdb_files:
                if os.path.dirname(orig_file) == dir_path:
                    base_name = os.path.basename(orig_file)
                    for cleaned_file in cleaned_files:
                        if os.path.basename(cleaned_file) == base_name:
                            cleaned_pdbs[orig_file] = cleaned_file
                            break
                    if orig_file not in cleaned_pdbs:
                        cleaned_pdbs[orig_file] = orig_file
        except Exception as e:
            print(f"Warning: Error cleaning PDB files in directory {dir_path}: {e}")
            for orig_file in pdb_files:
                if os.path.dirname(orig_file) == dir_path and orig_file not in cleaned_pdbs:
                    cleaned_pdbs[orig_file] = orig_file
    
    for i, pdb_file in enumerate(pdb_files):
        print(f"\nProcessing file {i+1} of {len(pdb_files)}: {pdb_file}")
        cleaned_pdb = cleaned_pdbs.get(pdb_file, pdb_file)
        process_pdb_file(pdb_file, cleaned_pdb, args.model, args.skip_depth)
    
    print(f"\nAll processing complete. {len(pdb_files)} PDB files processed.")
    print(f"\nTo visualize the predicted binding sites, run the following command in your terminal: \nPyMOL: pymol <output_prefix>.pml \nChimera: chimera <output_prefix>.cmd")

if __name__ == "__main__":
    main()