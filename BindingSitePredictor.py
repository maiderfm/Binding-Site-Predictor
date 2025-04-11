#!/usr/bin/env python3
"""
Protein Binding Site Prediction

This script:
1. Reads one or more protein structures in PDB format
2. Extracts features using feature_extraction.py module
3. Applies a pre-trained logistic regression model
4. Identifies predicted residue sites
5. Outputs visualization files for Chimera/PyMOL, 
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
import pickle
import numpy as np
import pandas as pd
import glob
import feature_extraction as fe
from Bio.PDB import PDBParser, PDBIO, Select
from argparse import ArgumentParser
from joblib import load
from sklearn.preprocessing import StandardScaler

class PredictedSiteSelector(Select):
    """Selector for residues identified as part of a predicted site."""
    
    def __init__(self, predicted_residues):
        """
        Args:
            predicted_residues: List of tuples (chain_id, residue_id) of predicted active residues
        """
        self.predicted_residues = predicted_residues
        
    def accept_residue(self, residue):
        """Return True if the residue is in the predicted sites list."""
        chain_id = residue.get_parent().id
        res_id = residue.id[1]
        return (chain_id, res_id) in self.predicted_residues


def extract_features_from_structure(structure, pdb_path):
    """
    Extract features from a protein structure using feature_extraction module functions.
    
    Args:
        structure: Bio.PDB Structure object
        pdb_path: Path to the PDB file
        
    Returns:
        features_df: DataFrame containing extracted features
        features_metadata: List of dictionaries with metadata for each residue
    """
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
    
    # Parse residues from PDB
    residues = fe.parse_pdb(pdb_path)
    
    if not residues:
        raise ValueError(f"No residues found in structure")
    
    # Collect feature data
    feature_rows = []
    features_metadata = []
    
    for position, residue, chain in residues:
        # Get amino acid properties
        properties = fe.get_residue_properties(residue)
        
        # Get secondary structure
        sec_struct = "Unknown"
        if chain in sec_struct_dict and position in sec_struct_dict[chain]:
            sec_struct = sec_struct_dict[chain][position]
        
        # Get ASA
        asa = None
        if chain in asa_dict and position in asa_dict[chain]:
            asa = asa_dict[chain][position]
        
        # Get residue depth
        depth = None
        if chain in depth_dict and position in depth_dict[chain]:
            depth = depth_dict[chain][position]
        
        # Get curvature
        curvature = None
        if chain in curvature_dict and position in curvature_dict[chain]:
            curvature = curvature_dict[chain][position]
        
        # Get neighbors
        neighbors = ""
        if chain in neighbors_dict and position in neighbors_dict[chain]:
            neighbors = neighbors_dict[chain][position]
        
        # Get contact frequency
        contact_freq = None
        if chain in contact_freq_dict and position in contact_freq_dict[chain]:
            contact_freq = contact_freq_dict[chain][position]
        
        # Get B-factor
        bfactor = None
        if chain in bfactor_dict and position in bfactor_dict[chain]:
            bfactor = bfactor_dict[chain][position]
        
        # Get polar density
        polar_density = None
        if chain in polar_density_dict and position in polar_density_dict[chain]:
            polar_density = polar_density_dict[chain][position]
        
        # Create feature row
        feature_row = {
            "Hydrophobicity": properties.get("Hydrophobicity", "NA"),
            "NormalizedVDWV": properties.get("NormalizedVDWV", "NA"),
            "Polarity": properties.get("Polarity", "NA"),
            "Charge": properties.get("Charge", "NA"),
            "SecondaryStr": properties.get("SecondaryStr", "NA"),
            "SolventAccessibility": properties.get("SolventAccessibility", "NA"),
            "Polarizability": properties.get("Polarizability", "NA"),
            "H-Bond Propensity": properties.get("H-Bond Propensity", "NA"),
            "Secondary_Structure": sec_struct,
            "ASA": asa if asa is not None else 0.0,
            "Surface_Curvature": curvature if curvature is not None else 0.0,
            "Contact_Frequency": contact_freq if contact_freq is not None else 0,
            "Bfactor": bfactor if bfactor is not None else 0.0,
            "Neighbors": neighbors,
            "Polar_Density": polar_density if polar_density is not None else 0.0,
            "Residue_Depth": depth if depth is not None else 0.0,
        }
        
        # Create metadata to map back to structure
        metadata = {
            "chain_id": chain,
            "residue_id": position,
            "residue_type": residue
        }
        
        feature_rows.append(feature_row)
        features_metadata.append(metadata)
    
    # Convert categorical features to one-hot encoding
    features_df = pd.DataFrame(feature_rows)
    
    # Convert categorical columns to one-hot encoding
    categorical_columns = [
        "Hydrophobicity", "NormalizedVDWV", "Polarity", "Charge", 
        "SecondaryStr", "SolventAccessibility", "Polarizability", 
        "H-Bond Propensity", "Secondary_Structure"
    ]
    
    # Only include columns that exist in the dataframe
    cat_cols_exist = [col for col in categorical_columns if col in features_df.columns]
    
    # Create dummy variables for categorical features
    features_df = pd.get_dummies(features_df, columns=cat_cols_exist, drop_first=False)
    
    return features_df, features_metadata

def get_pdb_files(input_paths):
    """
    Get a list of PDB files from various input types.
    
    Args:
        input_paths: List of file/directory paths or a file containing a list of paths
        
    Returns:
        List of absolute paths to PDB files
    """
    pdb_files = []
    
    for path in input_paths:
        # Check if the path is a directory
        if os.path.isdir(path):
            # Get all .pdb files in the directory
            pdb_files.extend(glob.glob(os.path.join(path, "*.pdb")))
            pdb_files.extend(glob.glob(os.path.join(path, "*.ent")))
        
        # Check if the path is a file with .pdb or .ent extension
        elif (path.lower().endswith(".pdb") or path.lower().endswith(".ent")) and os.path.isfile(path):
            pdb_files.append(os.path.abspath(path))
        
        # Check if the path is a file containing a list of PDB files
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
    """
    Process a single PDB file and generate output.
    
    Args:
        pdb_file: Original path to the PDB file (for naming purposes)
        cleaned_pdb: Path to the already cleaned PDB file
        model_path: Path to the trained model file
        skip_depth: Whether to skip residue depth calculation
    """
    # Get output base name from the input file name
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    output_prefix = base_name
    
    print(f"\n{'='*80}")
    print(f"Processing PDB file: {pdb_file}")
    print(f"Using cleaned file: {cleaned_pdb}")
    print(f"Output prefix: {output_prefix}")
    print(f"{'='*80}\n")
    
    # Step 2: Read the protein structure
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
    
    # Step 3: Extract features
    print("Extracting features...")
    try:
        features_df, features_metadata = extract_features_from_structure(structure, cleaned_pdb)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return
    
    # Process features for prediction
    # List of possible values in the 'Secondary_structure' column
    Second_strct_values = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
    # Create a dictionary with the values as keys and their index as the value
    Second_strct_dict = {value: index for index, value in enumerate(Second_strct_values)}

    if 'Secondary_Structure' in features_df.columns:
        features_df['Secondary_Structure'] = features_df['Secondary_Structure'].map(Second_strct_dict)
        features_df['Secondary_Structure'] = features_df['Secondary_Structure'].fillna(7)   # Replace NaN with 7    

    #### Dealing with non-numerical values from Neighbor column
    def extract_single_neighbor_feature(features_df):
        """
        Extract a single numerical feature from the 'Neighbors' column 
        that captures the most important information about protein binding sites.
    
        This function creates a weighted neighbor score that considers:
        1. Number of neighbors
        2. Chain diversity (with higher weight for different chain interactions)
        3. Proximity of neighbors (closer neighbors have higher influence)
    
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing a 'Neighbors' column with strings like "C:68,A:2,A:4,A:5,A:3"
        
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with a new 'Neighbor_Score' column and 'Neighbors' column removed
        """
        # Create a copy to avoid modifying the original dataframe
        result_df = features_df.copy()
    
        def calculate_neighbor_score(row):
            neighbor_str = row['Neighbors']
            # Return 0 if Neighbors is null or empty
            if not pd.notnull(neighbor_str) or not str(neighbor_str).strip():
                return 0
            
            # Parse the neighbor string
            neighbors = str(neighbor_str).split(',')
        
            # Early return if no neighbors
            if not neighbors or neighbors[0] == '':
                return 0
            
            # Initialize counters
            num_neighbors = len(neighbors)
            chains = set()
            same_chain_count = 0
            diff_chain_count = 0
        
            # Get the current chain if available
            current_chain = row.get('Chain', None)
        
            # Process each neighbor
            for item in neighbors:
                if ':' in item:
                    parts = item.split(':')
                    if len(parts) == 2:
                        chain = parts[0]
                        chains.add(chain)
                    
                        # Count same vs different chain interactions
                        if current_chain is not None:
                            if chain == current_chain:
                                same_chain_count += 1
                            else:
                                diff_chain_count += 1
        
            # Calculate chain diversity factor (higher weight for different chain interactions)
            chain_diversity = len(chains) / max(1, num_neighbors)
        
            # Calculate weighted score
            # Different chain interactions are weighted 3 times more than same chain
            weighted_interactions = (same_chain_count + 3 * diff_chain_count) / max(1, num_neighbors)
        
            # Final score combines number of neighbors and their diversity
            neighbor_score = num_neighbors * (0.3 + 0.7 * (chain_diversity * weighted_interactions))
        
            return neighbor_score
    
        # Calculate the neighbor score for each row
        result_df['Neighbor_Score'] = result_df.apply(calculate_neighbor_score, axis=1)
    
        # Drop the original Neighbors column - THIS IS IMPORTANT
        result_df = result_df.drop(columns=['Neighbors'])
    
        return result_df
    
    result_df = extract_single_neighbor_feature(features_df)
    
    # Scale the features to improve model performance (especially important for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(result_df)  # Fit and transform on the training data
    # Convert the numpy array back to a pandas DataFrame to handle it properly
    X_scaled_df = pd.DataFrame(X_scaled, columns=result_df.columns)
    
    # Step 4: Load and apply the trained model
    print(f"Loading model from {model_path}...")
    try:
        model_data = load(model_path)
        # Check if it's a tuple and extract the model   
        if isinstance(model_data, tuple):
            model = model_data[0]  # Assuming the model is the first element
        else:
            model = model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Making predictions...")
    try:
        # List of features expected by the model (you need to know this list based on training data)
        expected_features = ['ASA', 'Surface_Curvature', 'Contact_Frequency', 'Bfactor',
                            'Polar_Density', 'Residue_Depth', 'Hydrophobicity_1', 'Hydrophobicity_2', 
                            'Hydrophobicity_3', 'NormalizedVDWV_1', 'NormalizedVDWV_2', 'NormalizedVDWV_3', 
                            'Polarity_1', 'Polarity_2', 'Polarity_3', 'Charge_1']

        # Ensure the feature dataframe has exactly these columns
        X_scaled_df = X_scaled_df[expected_features]

        # Now make predictions with the correct feature alignment
        predictions = model.predict_proba(X_scaled_df)
        # Get indices where prediction exceeds threshold (e.g., 0.5)
        # Second column contains probability of positive class
        predicted_indices = np.where(predictions[:, 1] > 0.5)[0]
    
    except Exception as e:
        print(f"Error making predictions: {e}")
        return
    
    # Step 5: Map predictions back to protein residues
    predicted_residues = []
    for idx in predicted_indices:
        # Get mapping from feature index to residue
        chain_id = features_metadata[idx]['chain_id']
        res_id = features_metadata[idx]['residue_id']
        predicted_residues.append((chain_id, res_id))
    
    print(f"Identified {len(predicted_residues)} residues in predicted sites")
    
    # Step 6: Generate visualization files
    print(f"Generating output files with prefix {output_prefix}...")
    
    # Create PDB file with only predicted sites
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
        
        # Group residues by chain
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
    # Parse command line arguments
    parser = ArgumentParser(description="Predict active sites in protein structure")
    parser.add_argument("input", nargs='+', help="Input PDB file(s), directory of PDB files, or file containing list of PDB files")
    parser.add_argument("--model", default="model.pkl", help="Trained logistic regression model file")
    parser.add_argument("--skip-depth", action="store_true", help="Skip residue depth calculation")
    parser.add_argument("--clean-dir", default="cleaned_pdbs", help="Directory to store cleaned PDB files")
    args = parser.parse_args()
    
    # Get list of PDB files to process
    pdb_files = get_pdb_files(args.input)
    
    if not pdb_files:
        print("Error: No valid PDB files found to process.")
        sys.exit(1)
    
    print(f"Found {len(pdb_files)} PDB files to process.")
    
    # Step 1: Clean the PDB files
    print("Cleaning all PDB files...")
    # Create clean directory if it doesn't exist
    os.makedirs(args.clean_dir, exist_ok=True)
    
    # Create a mapping from original PDB files to cleaned PDB files
    cleaned_pdbs = {}
    
    # Get unique directories containing PDB files to clean
    unique_dirs = set(os.path.dirname(pdb_file) or "." for pdb_file in pdb_files)
  
    # Clean PDB files in each directory
    for dir_path in unique_dirs:
        try:
            cleaned_files = fe.clean_pdb_files(dir_path, args.clean_dir)
            # Map original files to cleaned files
            for orig_file in pdb_files:
                if os.path.dirname(orig_file) == dir_path:
                    base_name = os.path.basename(orig_file)
                    for cleaned_file in cleaned_files:
                        if os.path.basename(cleaned_file) == base_name:
                            cleaned_pdbs[orig_file] = cleaned_file
                            break
                    # If no matching cleaned file was found, use the original
                    if orig_file not in cleaned_pdbs:
                        cleaned_pdbs[orig_file] = orig_file
        except Exception as e:
            print(f"Warning: Error cleaning PDB files in directory {dir_path}: {e}")
            # Use original files in this directory
            for orig_file in pdb_files:
                if os.path.dirname(orig_file) == dir_path and orig_file not in cleaned_pdbs:
                    cleaned_pdbs[orig_file] = orig_file
    
    # Process each PDB file
    for i, pdb_file in enumerate(pdb_files):
        print(f"\nProcessing file {i+1} of {len(pdb_files)}: {pdb_file}")
        # Get the cleaned PDB file or use the original if cleaning failed
        cleaned_pdb = cleaned_pdbs.get(pdb_file, pdb_file)
        process_pdb_file(pdb_file, cleaned_pdb, args.model, args.skip_depth)
    
    print(f"\nAll processing complete. {len(pdb_files)} PDB files processed.")
    print(f"\nTo visualize the predicted binding sites, run the following command in your terminal: \nPyMOL: pymol <output_prefix>.pml \nChimera: chimera <output_prefix>.cmd")


if __name__ == "__main__":
    main()