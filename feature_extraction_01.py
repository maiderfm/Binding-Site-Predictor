import pandas as pd
import os
import re
import csv
import sys
import subprocess
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch, Selection, DSSP
from Bio.PDB.ResidueDepth import get_surface, residue_depth
from Bio.PDB.SASA import ShrakeRupley


# Three-letter to one-letter amino acid conversion
AA_TRANSLATION = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# Define amino acid properties
AAT_PROPERTY = {
    "Hydrophobicity": {'1': 'RKEDQN', '2': 'GASTPHY', '3': 'CLVIMFW'}, #1:polar, 2: neutral, 3:hydrophobic
    "NormalizedVDWV": {'1': 'CGASTPD', '2': 'NVEQIL', '3': 'MHKFRYW'},
    "Polarity" : {'1':'YFVMCWIL', '2': 'SGTAP', '3':'DENKRQH'}, #'1'stand for (4.9-6.2); '2'stand for (8.0-9.2), '3' stand for (10.4-13.0)
    "Charge": {'1': 'KR', '2': 'ANCQGHILMFPSTWYV', '3': 'DE'},
    "SecondaryStr": {'1': 'EALMQKRH', '2': 'VIYCWFT', '3': 'GNPSD'},
    "SolventAccessibility": {'1': 'ALFCGIVW', '2': 'RKQEND', '3': 'MPSTHY'},
    "Polarizability": {'1': 'GASDT', '2': 'CPNVEQIL', '3': 'KMHFRYW'}, 
    "H-Bond Propensity": {'1': 'RKEQNDSTYH','2': 'WC','3': 'AGPFLIVM'}
}


# Definir residuos polares y cargados para cálculo de densidad
POLAR_CHARGED = 'RKEQDNHSTY'  # Residuos polares y cargados

# Function to extract features based on binding site sequences
def extract_bs(subset_df):
    binding_sites = {}

    # Parse each row in the dataframe
    for _, row in subset_df.iterrows():
        protein_name = row['protein_name']
        chain_id = row['chain_id']
        bs_list = row['binding_site_sequence'].split()  # Split into individual residues

        if protein_name not in binding_sites:
            binding_sites[protein_name] = {}

        if chain_id not in binding_sites[protein_name]:
            binding_sites[protein_name][chain_id] = set()

        # Extract positions
        for binding in bs_list:
            if len(binding) > 1 and binding[1:].isdigit():  # Ensure valid format
                residue_position = int(binding[1:])  # Convert to int
                binding_sites[protein_name][chain_id].add(residue_position)

    return binding_sites

def get_residue_properties(residue):
    """Returns a dictionary of property values for a given residue (one-letter code)"""
    properties = {}
    for prop_name, prop_values in AAT_PROPERTY.items():
        assigned_class = "NA"  # Default if residue is not found
        for class_id, aa_group in prop_values.items():
            if residue in aa_group:
                assigned_class = class_id
                break
        properties[prop_name] = assigned_class
    return properties

def clean_pdb_files(pdb_dir, output_dir=None):
    """
    Clean PDB files by:
    1. Keeping only specific record types: HEADER, TITLE, ATOM, HETATM, TER, END
    2. Keeping only the first MODEL if multiple models exist
    
    Args:
        pdb_dir: Directory containing PDB files
        output_dir: Directory to save cleaned files (if None, will use pdb_dir)
    
    Returns:
        A list of paths to the cleaned PDB files
    """
    
    if output_dir is None:
        output_dir = pdb_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store paths of cleaned files
    cleaned_files = []
    
    # Regular expression pattern for lines to keep
    pattern = re.compile(r'^(HEADER|TITLE|ATOM|HETATM|TER|END)')
    
    # Process each PDB file
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".ent") or filename.endswith(".pdb"):
            input_path = os.path.join(pdb_dir, filename)
            
            # Create output filename (keep same name but add _clean)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"clean_{base_name}.pdb"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                with open(input_path, 'r') as infile:
                    lines = infile.readlines()
                
                new_pdb = []
                in_model = False  # Flag to track first MODEL

                for line in lines:
                    if line.startswith("MODEL"):
                        if in_model:  
                            break  # Si ya estábamos en un modelo, terminamos
                        in_model = True  # Marcamos que estamos en el primer modelo
                    
                    if pattern.match(line) or in_model:
                        new_pdb.append(line)
                    
                    if line.startswith("ENDMDL"):
                        break  # Terminamos cuando acaba el primer modelo
                
                # Escribir archivo limpio
                with open(output_path, 'w') as outfile:
                    outfile.writelines(new_pdb)

                cleaned_files.append(output_path)
                print(f"Cleaned {filename} -> {output_filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"Cleaned {len(cleaned_files)} PDB files")
    return cleaned_files

def parse_pdb(pdb_file):
    """Extracts residue information from a PDB file, converting to one-letter amino acid codes"""
    residue_info = []
    try:
        with open(pdb_file, 'r') as file:
            for line in file:
                if line.startswith("ATOM") and line[13:15].strip() == "CA":  # Extract only Cα atoms
                    chain = line[21]
                    
                    # Check if there's enough data in the line to extract residue information
                    if len(line) > 26:
                        try:
                            three_letter_res = line[17:20].strip()
                            position = int(line[22:26].strip())  # Convert to int for proper comparison

                            # Convert to one-letter amino acid code
                            one_letter_res = AA_TRANSLATION.get(three_letter_res, "X")  # "X" for unknown residues

                            residue_info.append((position, one_letter_res, chain))
                        except ValueError:
                            # Skip lines with invalid position information
                            continue
    except Exception as e:
        print(f"Error parsing PDB file {pdb_file}: {str(e)}")
    return residue_info

def find_mkdssp_path():
    """Find the full path to the mkdssp executable"""
    try:
        # Try to get mkdssp path using the 'which' command
        result = subprocess.run(['which', 'mkdssp'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        
        # If 'which' failed, try common locations
        common_paths = [
            '/usr/bin/mkdssp',
            '/usr/local/bin/mkdssp',
            '/opt/homebrew/bin/mkdssp',  # For Mac with Homebrew
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    except Exception as e:
        print(f"Error finding mkdssp path: {str(e)}")
        return None

def run_dssp_command_line(pdb_file, chain_id_mapping=None):
    """Run DSSP directly using command line and parse the output file"""
    sec_struct_dict = {}
    
    try:
        # Find the mkdssp executable
        mkdssp_path = find_mkdssp_path()
        if not mkdssp_path:
            print("Could not find mkdssp executable. Skipping secondary structure analysis.")
            return sec_struct_dict
        
        # Create a temporary output file
        output_file = f"{pdb_file}.dssp"
        
        # Run mkdssp
        result = subprocess.run([mkdssp_path, pdb_file, output_file], 
                               capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            print(f"DSSP command failed: {result.stderr}")
            return sec_struct_dict
        
        # Parse the DSSP output file
        if os.path.exists(output_file):
            reading_data = False
            with open(output_file, 'r') as f:
                for line in f:
                    # Find the start of the data section
                    if line.strip().startswith("#  RESIDUE"):
                        reading_data = True
                        continue
                    
                    if reading_data and len(line) > 17:  # Ensure line is long enough
                        try:
                            chain = line[11].strip()
                            if not chain:  # Some DSSP files might have empty chain IDs
                                chain = " "
                                
                            res_num = int(line[5:10].strip())
                            sec_struct = line[16].strip()
                            if not sec_struct:
                                sec_struct = "-"  # Default for no structure
                            
                            if chain not in sec_struct_dict:
                                sec_struct_dict[chain] = {}
                                
                            sec_struct_dict[chain][res_num] = sec_struct
                        except (ValueError, IndexError) as e:
                            # Skip problematic lines
                            continue
            
            # Clean up the temporary file
            try:
                os.remove(output_file)
            except:
                pass
                
    except Exception as e:
        print(f"Error running DSSP for {pdb_file}: {str(e)}")
    
    return sec_struct_dict

def extract_secondary_structure(pdb_file, structure=None):
    """Extracts residue positions and secondary structure using DSSP, and calculates ASA values."""
    sec_struct_dict = {}
    asa_dict = {}  # Dictionary to store ASA values
  
    try:    
        # Find mkdssp path once
        mkdssp_path = find_mkdssp_path()
        if not mkdssp_path:
            print("Could not find mkdssp executable. Skipping secondary structure analysis.")
            return sec_struct_dict, asa_dict
        
        # Initialize structure if not provided
        if structure is None:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_file)
        
        # Second try: Direct command-line approach (already has the mkdssp_path from above)
        print(f"Trying command-line DSSP for {pdb_file}")
        
        # Create a temporary output file
        output_file = f"{pdb_file}.dssp"
        
        # Run mkdssp
        result = subprocess.run([mkdssp_path, pdb_file, output_file], 
                               capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            print(f"DSSP command failed: {result.stderr}")
            return sec_struct_dict, asa_dict
        
        # Parse the DSSP output file
        if os.path.exists(output_file):
            reading_data = False
            with open(output_file, 'r') as f:
                for line in f:
                    # Find the start of the data section
                    if line.strip().startswith("#  RESIDUE"):
                        reading_data = True
                        continue
                    
                    if reading_data and len(line) > 35:  # Ensure line is long enough for ASA (column 36-38)
                        try:
                            chain = line[11].strip()
                            if not chain:  # Some DSSP files might have empty chain IDs
                                chain = " "
                                
                            res_num = int(line[5:10].strip())
                            sec_struct = line[16].strip()
                            if not sec_struct:
                                sec_struct = "-"  # Default for no structure
                            
                            # Extract ASA value (typically in columns 36-38)
                            asa_value = float(line[35:38].strip())
                            
                            if chain not in sec_struct_dict:
                                sec_struct_dict[chain] = {}
                                asa_dict[chain] = {}
                                
                            sec_struct_dict[chain][res_num] = sec_struct
                            asa_dict[chain][res_num] = asa_value
                        except (ValueError, IndexError) as e:
                            # Skip problematic lines
                            continue
            
            # Clean up the temporary file
            try:
                os.remove(output_file)
            except:
                pass
                
    except Exception as e:
        print(f"Error processing structure for secondary structure on {pdb_file}: {str(e)}")
    
    return sec_struct_dict, asa_dict
    

def calculate_residue_depths(structure, pdb_path):
    """Calculate residue depths for all residues in the structure """
    depth_dict = {}
    
    try:
        # Check if MSMS is available
        msms_found = False
        msms_path = os.environ.get("PATH", "").split(os.pathsep)
        for path in msms_path:
            if os.path.exists(os.path.join(path, "msms")) or os.path.exists(os.path.join(path, "msms.exe")):
                msms_found = True
                break
                
        if not msms_found:
            print(f"Warning: MSMS executable not found in PATH for {pdb_path}. Skipping depth calculations.")
            return depth_dict
            
        # Try to get surface using MSMS (only once per structure for efficiency)
        try:
            surface = get_surface(structure)
            
            # Create a dictionary to store depth values by chain and residue position
            for model in structure:
                for chain in model:
                    chain_id = chain.id
                    if chain_id not in depth_dict:
                        depth_dict[chain_id] = {}
                    
                    for residue in chain:
                        # Skip hetero-atoms and water
                        if residue.id[0] != " ":
                            continue
                        
                        res_pos = residue.id[1]  # Residue position
                        
                        try:
                            # Calculate residue depth 
                            depth = residue_depth(residue, surface)
                            depth_dict[chain_id][res_pos] = depth
                        except Exception as e:
                            print(f"Error calculating depth for residue {residue.get_resname()} {res_pos}: {e}")
                            depth_dict[chain_id][res_pos] = None
        except Exception as e:
            print(f"Error generating surface for {pdb_path}: {e}")
            
  
    except Exception as e:
        print(f"Error in residue depth calculation for {pdb_path}: {e}")
        print("Possible fix: Ensure the PDB file does not have duplicate atom positions.")
   
            
    return depth_dict

def calculate_curvature(structure, radius=10.0):
    """
    Calcula una aproximación de la curvatura local para cada residuo.
    Un valor positivo indica una superficie convexa, un valor negativo indica una cavidad.
    """
    curvature_dict = {}
    
    # Get all atoms
    atom_list = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atom_list)
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in curvature_dict:
                curvature_dict[chain_id] = {}
                
            for residue in chain:
                # Skip hetero-atoms and water
                if residue.id[0] != " ":
                    continue
                    
                res_pos = residue.id[1]
                
                try:
                    # Check if it's a protein residue (has CA atom)
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        
                        # Get neighboring atoms within radius
                        neighbors = ns.search(ca_atom.coord, radius, level='A')
                        
                        # Calculate center of mass of neighboring atoms
                        if len(neighbors) > 1:
                            com = np.zeros(3)
                            for atom in neighbors:
                                com += atom.coord
                            com /= len(neighbors)
                            
                            # Vector from CA to center of mass - approximates surface normal
                            normal_vector = com - ca_atom.coord
                            
                            # Distance from CA to center of mass
                            normal_length = np.linalg.norm(normal_vector)
                            
                            # Consider direction
                            avg_vector = np.zeros(3)
                            for atom in atom_list:
                                avg_vector += (atom.coord - ca_atom.coord)
                            avg_vector /= len(atom_list)
                            
                            dot_product = np.dot(normal_vector, avg_vector)
                            sign = -1 if dot_product < 0 else 1
                            
                            # Store curvature value (signed distance to COM)
                            curvature_dict[chain_id][res_pos] = sign * normal_length
                        else:
                            curvature_dict[chain_id][res_pos] = 0.0
                    else:
                        # For non-protein residues (like DNA/RNA), set to None instead of trying to calculate
                        curvature_dict[chain_id][res_pos] = None
                        
                except Exception as e:
                    # Handle error more gracefully - just set to None without printing error
                    # print(f"Error calculating curvature for residue {residue.get_resname()} {res_pos}: {e}")
                    curvature_dict[chain_id][res_pos] = None
    
    return curvature_dict

def get_structural_neighbors(structure, distance_threshold=5.0):
    """
    Encuentra los residuos vecinos dentro de un umbral de distancia específico.
    Retorna un diccionario con información de vecindad.
    """
    neighbors_dict = {}
    contact_freq_dict = {}  # Para almacenar la frecuencia de contacto
    
    # Get all atoms
    atom_list = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atom_list)
    
    # First collect all residues
    all_residues = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in all_residues:
                all_residues[chain_id] = {}
                neighbors_dict[chain_id] = {}
                contact_freq_dict[chain_id] = {}
            
            for residue in chain:
                # Skip hetero-atoms and water
                if residue.id[0] != " ":
                    continue
                
                res_pos = residue.id[1]
                all_residues[chain_id][res_pos] = residue
    
    # Now compute neighbors and contact frequency
    for chain_id, residues in all_residues.items():
        for res_pos, residue in residues.items():
            neighbors_dict[chain_id][res_pos] = []
            
            # Count atom contacts for contact frequency
            contact_count = 0
            total_atoms = 0
            
            for atom in residue:
                total_atoms += 1
                
                # Get atoms in contact with current atom
                neighbors = ns.search(atom.coord, distance_threshold, level='A')
                
                # Count contacts with atoms from other residues
                for neighbor_atom in neighbors:
                    if neighbor_atom.get_parent() != residue:  # Different residue
                        contact_count += 1
            
            # Store normalized contact frequency (contacts per atom)
            contact_freq = contact_count / max(1, total_atoms)
            contact_freq_dict[chain_id][res_pos] = contact_count
            
            # Get residue neighbors
            ca_neighbors = []
            if 'CA' in residue:
                ca_atom = residue['CA']
                neighbor_atoms = ns.search(ca_atom.coord, distance_threshold, level='R')
                
                for neighbor_res in neighbor_atoms:
                    if neighbor_res != residue and neighbor_res.id[0] == " ":  # Skip self and non-standard residues
                        neighbor_chain = neighbor_res.get_parent().id
                        neighbor_pos = neighbor_res.id[1]
                        
                        # Store as "chain:position"
                        ca_neighbors.append(f"{neighbor_chain}:{neighbor_pos}")
            
            # Store up to 10 closest neighbors to keep data size manageable
            neighbors_dict[chain_id][res_pos] = ",".join(ca_neighbors[:10]) if ca_neighbors else ""
    
    return neighbors_dict, contact_freq_dict

def calculate_bfactor_and_polar_density(structure):
    """
    Extrae los B-factors y calcula la densidad de residuos polares/cargados alrededor
    de cada residuo. Retorna dos diccionarios con los valores.
    """
    bfactor_dict = {}
    polar_density_dict = {}
    
    # Get all residues
    atom_list = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atom_list)
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in bfactor_dict:
                bfactor_dict[chain_id] = {}
                polar_density_dict[chain_id] = {}
                
            for residue in chain:
                # Skip hetero-atoms and water
                if residue.id[0] != " ":
                    continue
                    
                res_pos = residue.id[1]
                
                # Get B-factor (average over atoms in residue)
                bfactor_sum = 0.0
                atom_count = 0
                
                for atom in residue:
                    if hasattr(atom, "bfactor") and atom.bfactor is not None:
                        bfactor_sum += atom.bfactor
                        atom_count += 1
                
                # Store average B-factor
                if atom_count > 0:
                    bfactor_dict[chain_id][res_pos] = bfactor_sum / atom_count
                else:
                    bfactor_dict[chain_id][res_pos] = None
                
                # Calculate density of polar/charged residues within 8 Angstroms
                try:
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        neighbor_residues = ns.search(ca_atom.coord, 8.0, level='R')
                        
                        # Count polar/charged neighbors
                        total_neighbors = 0
                        polar_charged_count = 0
                        
                        for neighbor in neighbor_residues:
                            if neighbor != residue and neighbor.id[0] == " ":
                                total_neighbors += 1
                                
                                # Check if it's a polar/charged residue
                                res_name = neighbor.get_resname()
                                one_letter = AA_TRANSLATION.get(res_name, "X")
                                
                                if one_letter in POLAR_CHARGED:
                                    polar_charged_count += 1
                        
                        # Calculate density (percentage of polar/charged)
                        if total_neighbors > 0:
                            density = (polar_charged_count / total_neighbors) * 100
                        else:
                            density = 0.0
                            
                        polar_density_dict[chain_id][res_pos] = density
                    else:
                        polar_density_dict[chain_id][res_pos] = None
                except Exception as e:
                    print(f"Error calculating polar density for residue {residue.get_resname()} {res_pos}: {e}")
                    polar_density_dict[chain_id][res_pos] = None
    
    return bfactor_dict, polar_density_dict

def store_feature_value(pdb_dir, output_file="binding_features.csv", skip_depth=False, batch_size=100):
    """Extracts and stores residue features from PDB files with periodic saving"""
    # First, load BioLiP data and extract binding sites
    binding_sites_dict = {}
    try:
        if os.path.exists('BioLiP_nr.txt'):
            df = pd.read_csv('BioLiP_nr.txt', delimiter='\t', low_memory=False, header=None)
            subset_df = df[[0, 1, 7]]  # Select columns 0 and 7
            subset_df.columns = ['protein_name', 'chain_id', 'binding_site_sequence']  # Rename columns
            binding_sites_dict = extract_bs(subset_df)
            print("Successfully loaded BioLiP data")
        else:
            print("Warning: BioLiP_nr.txt not found. Binding site information will not be available.")
    except Exception as e:
        print(f"Error loading BioLiP data: {str(e)}")

    # Find and print DSSP executable path
    mkdssp_path = find_mkdssp_path()
    if mkdssp_path:
        print(f"Found mkdssp at: {mkdssp_path}")
    else:
        print("WARNING: mkdssp executable not found. Secondary structure analysis will be limited.")

    all_features = []
    total_files = 0
    processed_files = 0
    
    # Count total files for progress reporting
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb"):
            total_files += 1
    
    print(f"Found {total_files} .pdb files to process")
    
    # Variables for batch processing
    batch_counter = 0
    batch_number = 1
    
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb"):
            processed_files += 1
            batch_counter += 1
            pdb_path = os.path.join(pdb_dir, filename)
            
            # Process the filename to extract the protein name
            pdb_name = filename.rsplit('.pdb', 1)[0]
            if pdb_name.startswith("clean_pdb"):
                pdb_name = pdb_name[9:]  # Remove the "pdb" prefix
            
            print(f"Processing file {processed_files}/{total_files}: {filename} -> {pdb_name}")
            
            # Set up parser and parse structure once
            parser = PDBParser(QUIET=True)
            try:
                structure = parser.get_structure(pdb_name, pdb_path)
                
                # Extract secondary structure information and ASA
                sec_struct_dict, asa_dict = extract_secondary_structure(pdb_path, structure)
                
                # Calculate residue depths (existing feature)
                depth_dict = {}
                if not skip_depth:
                    depth_dict = calculate_residue_depths(structure, pdb_path)
                
                # Calculate curvature for all residues
                curvature_dict = calculate_curvature(structure)
                
                # Get structural neighbors and contact frequency
                neighbors_dict, contact_freq_dict = get_structural_neighbors(structure)
                
                # Get B-factors and polar density
                bfactor_dict, polar_density_dict = calculate_bfactor_and_polar_density(structure)
                
                # Parse residues from PDB
                residues = parse_pdb(pdb_path)
                
                if not residues:
                    print(f"Warning: No residues found in {filename}")
                    continue
                
                for position, residue, chain in residues:
                    properties = get_residue_properties(residue)
                    
                    # Get secondary structure for this residue (if available)
                    sec_struct = "Unknown"
                    if chain in sec_struct_dict and position in sec_struct_dict[chain]:
                        sec_struct = sec_struct_dict[chain][position]
                    
                    # Get ASA for this residue (if available)
                    asa = None
                    if chain in asa_dict and position in asa_dict[chain]:
                        asa = asa_dict[chain][position]
                    
                    # Get residue depth (if available)
                    depth = None
                    if chain in depth_dict and position in depth_dict[chain]:
                        depth = depth_dict[chain][position]
                    
                    # Get curvature (if available)
                    curvature = None
                    if chain in curvature_dict and position in curvature_dict[chain]:
                        curvature = curvature_dict[chain][position]
                    
                    # Get neighbors (if available)
                    neighbors = ""
                    if chain in neighbors_dict and position in neighbors_dict[chain]:
                        neighbors = neighbors_dict[chain][position]
                    
                    # Get contact frequency (if available)
                    contact_freq = None
                    if chain in contact_freq_dict and position in contact_freq_dict[chain]:
                        contact_freq = contact_freq_dict[chain][position]
                    
                    # Get B-factor (if available)
                    bfactor = None
                    if chain in bfactor_dict and position in bfactor_dict[chain]:
                        bfactor = bfactor_dict[chain][position]
                    
                    # Get polar density (if available)
                    polar_density = None
                    if chain in polar_density_dict and position in polar_density_dict[chain]:
                        polar_density = polar_density_dict[chain][position]
                    
                    # Check if this residue is a binding site according to BioLiP
                    is_binding_site = 0
                    if pdb_name in binding_sites_dict and chain in binding_sites_dict[pdb_name] and position in binding_sites_dict[pdb_name][chain]:
                        is_binding_site = 1
                    
                    row = {
                        "Protein": pdb_name, 
                        "Position": position, 
                        "Residue": residue, 
                        "Chain": chain,
                        **properties, 
                        "Secondary_Structure": sec_struct,
                        "ASA": asa,
                        "Surface_Curvature": curvature,
                        "Contact_Frequency": contact_freq,
                        "Bfactor": bfactor,
                        "Neighbors": neighbors,
                        "Polar_Density": polar_density,
                        "Residue_Depth": depth,
                        "Binding_Site": is_binding_site
                    }
                    
                    all_features.append(row)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
            
            # Save in batches to prevent data loss on interruption
            if batch_counter >= batch_size:
                # Append to csv if it exists, otherwise create new file
                if all_features:
                    mode = 'a' if processed_files > batch_size and os.path.exists(output_file) else 'w'
                    write_header = mode == 'w'
                    
                    try:
                        with open(output_file, mode, newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=all_features[0].keys())
                            if write_header:
                                writer.writeheader()
                            writer.writerows(all_features)
                        print(f"Batch {batch_number} saved to {output_file} ({len(all_features)} records)")
                    except Exception as e:
                        print(f"Error writing batch to output file: {str(e)}")
                
                # Reset for next batch
                all_features = []
                batch_counter = 0
                batch_number += 1
           
    # Write any remaining features to CSV
    if all_features:
        print(f"Processing complete. Found {len(all_features)} residue features in final batch.")
        
        try:
            mode = 'a' if processed_files > batch_size and os.path.exists(output_file) else 'w'
            write_header = mode == 'w'
            
            with open(output_file, mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_features[0].keys())
                if write_header:
                    writer.writeheader()
                writer.writerows(all_features)
            print(f"Final batch saved to {output_file}")
        except Exception as e:
            print(f"Error writing output file: {str(e)}")
    
    print(f"Total processing complete. Processed {processed_files} files.")

if __name__ == "__main__":
    # Get command line arguments if provided, otherwise use defaults
    pdb_directory = "0_Raw_Data/ml_pdb"
    output_file = "01_Feature_Data/feature_data.csv"

    clean_directory = None
    clean_only = False
    
    # Add command-line argument to skip depth calculation if MSMS is causing issues
    skip_depth = False
    batch_size = 100  # Default batch size
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--skip-depth":
            skip_depth = True
            print("Skipping residue depth calculations")
        elif sys.argv[i] == "--batch-size" and i + 1 < len(sys.argv):
            try:
                batch_size = int(sys.argv[i + 1])
                i += 1  # Skip the next argument since we used it
            except ValueError:
                print(f"Invalid batch size: {sys.argv[i+1]}. Using default: {batch_size}")
        elif sys.argv[i] == "--clean-dir" and i + 1 < len(sys.argv):
            clean_directory = sys.argv[i + 1]
            i += 1
        elif sys.argv[i] == "--clean-only":
            clean_only = True
        elif not sys.argv[i].startswith("--") and i == 1:
            pdb_directory = sys.argv[i]
        elif not sys.argv[i].startswith("--") and i == 2:
            output_file = sys.argv[i]
        i += 1
    
    print(f"Using PDB directory: {pdb_directory}")
    print(f"Output will be saved to: {output_file}")
    print(f"Batch size: {batch_size}")

    # Clean the PDB files first
    cleaned_files = clean_pdb_files(pdb_directory, clean_directory)
    
    # If clean_only flag is set, exit after cleaning
    if clean_only:
        print("Files cleaned. Exiting without further processing.")
        sys.exit(0)
    
    # If a clean directory was specified, use that for processing instead
    if clean_directory:
        pdb_directory = clean_directory
    
    store_feature_value(pdb_directory, output_file, skip_depth, batch_size)
