import pandas as pd
import os
import re
import csv
import sys
import subprocess
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch, Selection
from Bio.PDB.ResidueDepth import get_surface, residue_depth

# Dictionary for aminoacid abbreviations
AA_TRANSLATION = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# Definition of aminoacid properties
AAT_PROPERTY = {
    "Hydrophobicity": {'1': 'RKEDQN', '2': 'GASTPHY', '3': 'CLVIMFW'}, #1:polar, 2: neutral, 3:hydrophobic
    "NormalizedVDWV": {'1': 'CGASTPD', '2': 'NVEQIL', '3': 'MHKFRYW'},
    "Polarity" : {'1':'YFVMCWIL', '2': 'SGTAP', '3':'DENKRQH'}, #'1'stand for (4.9-6.2); '2'stand for (8.0-9.2), '3' stand for (10.4-13.0)
    "Charge": {'1': 'KR', '2': 'ANCQGHILMFPSTWYV', '3': 'DE'},
    "SecondaryStr": {'1': 'EALMQKRH', '2': 'VIYCWFT', '3': 'GNPSD'},
    "SolventAccessibility": {'1': 'ALFCGIVW', '2': 'RKQEND', '3': 'MPSTHY'},
    "Polarizability": {'1': 'GASDT', '2': 'CPNVEQIL', '3': 'KMHFRYW'}, 
    "H-Bond_Propensity": {'1': 'RKEQNDSTYH','2': 'WC','3': 'AGPFLIVM'}
}

POLAR_CHARGED = 'RKEQDNHSTY'  

def extract_bs(subset_df):
    """Returns a dictionary of binding sites aminoacids and positions"""
    binding_sites = {}

    for _, row in subset_df.iterrows():
        protein_name = row['protein_name']
        chain_id = row['chain_id']
        bs_list = row['binding_site_sequence'].split()  

        if protein_name not in binding_sites:
            binding_sites[protein_name] = {}

        if chain_id not in binding_sites[protein_name]:
            binding_sites[protein_name][chain_id] = set()

        for binding in bs_list:
            if len(binding) > 1 and binding[1:].isdigit():  
                residue_position = int(binding[1:])  
                binding_sites[protein_name][chain_id].add(residue_position)

    return binding_sites

def get_residue_properties(residue):
    """Returns a dictionary with aminoacid properties"""
    properties = {}
    for prop_name, prop_values in AAT_PROPERTY.items():
        assigned_class = "NA"  
        for class_id, aa_group in prop_values.items():
            if residue in aa_group:
                assigned_class = class_id
                break
        properties[prop_name] = assigned_class
    return properties

def clean_pdb_files(pdb_dir, output_dir=None):
    """Returns a list with the paths of the cleaned pdb files;
    the pdb files are cleaned by keeping HEADER, TITLE, ATOM, HETATM, TER, END;
    if multiple models of the opdb exist, only the first is kept"""
    
    if output_dir is None:
        output_dir = pdb_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_files = []
    
    pattern = re.compile(r'^(HEADER|TITLE|ATOM|HETATM|TER|END)')
    
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".ent") or filename.endswith(".pdb"):
            input_path = os.path.join(pdb_dir, filename)
            
            base_name = os.path.splitext(filename)[0]
            output_filename = f"clean_{base_name}.pdb"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                with open(input_path, 'r') as infile:
                    lines = infile.readlines()
                
                new_pdb = []
                in_model = False  

                for line in lines:
                    if line.startswith("MODEL"):
                        if in_model:  
                            break  
                        in_model = True  
                    
                    if pattern.match(line) or in_model:
                        new_pdb.append(line)
                    
                    if line.startswith("ENDMDL"):
                        break  
                
                with open(output_path, 'w') as outfile:
                    outfile.writelines(new_pdb)

                cleaned_files.append(output_path)
                print(f"Cleaned {filename} -> {output_filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"Cleaned {len(cleaned_files)} pdb files")
    return cleaned_files

def parse_pdb(pdb_file):
    """Extract information of residues (residue, position, chain) from a pdb file"""
    residue_info = []
    try:
        with open(pdb_file, 'r') as file:
            for line in file:
                if line.startswith("ATOM") and line[13:15].strip() == "CA": 
                    chain = line[21]
                    if len(line) > 26:
                        try:
                            three_letter_res = line[17:20].strip()
                            position = int(line[22:26].strip())  

                            one_letter_res = AA_TRANSLATION.get(three_letter_res, "X")  

                            residue_info.append((position, one_letter_res, chain))
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error parsing pdb file {pdb_file}: {str(e)}")
    return residue_info

def find_mkdssp_path():
    """Find the full path for executable mkdssp api"""
    try:
        result = subprocess.run(['which', 'mkdssp'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        common_paths = [
            '/usr/bin/mkdssp',
            '/usr/local/bin/mkdssp',
            '/opt/homebrew/bin/mkdssp',  
        ]
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        return None
    except Exception as e:
        print(f"Error finding path of mkdssp: {str(e)}")
        return None

def run_dssp_command_line(pdb_file, chain_id_mapping=None):
    """Run DSSP with mkdssp and manage the output file"""

    sec_struct_dict = {}
    
    try:
        mkdssp_path = find_mkdssp_path()
        if not mkdssp_path:
            print("Could not find mkdssp executable. Skipping the secondary structure analysis.")
            return sec_struct_dict
        
        output_file = f"{pdb_file}.dssp"
        
        result = subprocess.run([mkdssp_path, pdb_file, output_file], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"DSSP failed: {result.stderr}")
            return sec_struct_dict
        
        if os.path.exists(output_file):
            reading_data = False
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip().startswith("#  RESIDUE"):
                        reading_data = True
                        continue
                    
                    if reading_data and len(line) > 17:  
                        try:
                            chain = line[11].strip()
                            if not chain:  
                                chain = " "
                                
                            res_num = int(line[5:10].strip())
                            sec_struct = line[16].strip()
                            if not sec_struct:
                                sec_struct = "-"  
                            
                            if chain not in sec_struct_dict:
                                sec_struct_dict[chain] = {}
                                
                            sec_struct_dict[chain][res_num] = sec_struct
                        except (ValueError, IndexError) as e:
                            continue
            
            try:
                os.remove(output_file)
            except:
                pass
                
    except Exception as e:
        print(f"Error running DSSP of {pdb_file}: {str(e)}")
    
    return sec_struct_dict

def extract_secondary_structure(pdb_file, structure=None):
    """Extract secondary structure and asa value for each residue, including each position and chain""" 

    sec_struct_dict = {}
    asa_dict = {}  
  
    try:    
        mkdssp_path = find_mkdssp_path()
        if not mkdssp_path:
            print("Could not find mkdssp executable. Skipping the secondary structure analysis.")
            return sec_struct_dict, asa_dict
        
        if structure is None:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_file)
                
        output_file = f"{pdb_file}.dssp"
        
        result = subprocess.run([mkdssp_path, pdb_file, output_file], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"DSSP failed: {result.stderr}")
            return sec_struct_dict, asa_dict
        
        if os.path.exists(output_file):
            reading_data = False
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip().startswith("#  RESIDUE"):
                        reading_data = True
                        continue
                    
                    if reading_data and len(line) > 35:  
                        try:
                            chain = line[11].strip()
                            if not chain:  
                                chain = " "
                                
                            res_num = int(line[5:10].strip())
                            sec_struct = line[16].strip()
                            if not sec_struct:
                                sec_struct = "-"  
                            
                            asa_value = float(line[35:38].strip())
                            
                            if chain not in sec_struct_dict:
                                sec_struct_dict[chain] = {}
                                asa_dict[chain] = {}
                                
                            sec_struct_dict[chain][res_num] = sec_struct
                            asa_dict[chain][res_num] = asa_value
                        except (ValueError, IndexError) as e:
                            continue
            
            try:
                os.remove(output_file)
            except:
                pass
                
    except Exception as e:
        print(f"Error processing secondary structure of {pdb_file}: {str(e)}")
    
    return sec_struct_dict, asa_dict
    

def calculate_residue_depths(structure, pdb_path):
    """Calculate and extract residue depth for each residue of each pdb file."""
    depth_dict = {}
    
    try:
        msms_found = False
        msms_path = os.environ.get("PATH", "").split(os.pathsep)
        for path in msms_path:
            if os.path.exists(os.path.join(path, "msms")) or os.path.exists(os.path.join(path, "msms.exe")):
                msms_found = True
                break
                
        if not msms_found:
            print(f"MSMS executable not found. Skipping depth calculations.")
            return depth_dict
            
        try:
            surface = get_surface(structure)
            
            for model in structure:
                for chain in model:
                    chain_id = chain.id
                    if chain_id not in depth_dict:
                        depth_dict[chain_id] = {}
                    
                    for residue in chain:
                        if residue.id[0] != " ":
                            continue
                        
                        res_pos = residue.id[1] 
                        
                        try:
                            depth = residue_depth(residue, surface)
                            depth_dict[chain_id][res_pos] = depth
                        except Exception as e:
                            print(f"Error calculating depth for residue {residue.get_resname()} {res_pos}: {e}")
                            depth_dict[chain_id][res_pos] = None
        except Exception as e:
            print(f"Error generating surface for {pdb_path}: {e}")
            
    except Exception as e:
        print(f"Error calculating depth for residue {pdb_path}: {e}")
               
    return depth_dict

def calculate_curvature(structure, radius=10.0):
    """Calculate and extract the aproximation of the curvature for each residue of each pdb file"""

    curvature_dict = {}
    
    atom_list = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atom_list)
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in curvature_dict:
                curvature_dict[chain_id] = {}
                
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                    
                res_pos = residue.id[1]
                
                try:
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        
                        neighbors = ns.search(ca_atom.coord, radius, level='A')
                        
                        if len(neighbors) > 1:
                            com = np.zeros(3)
                            for atom in neighbors:
                                com += atom.coord
                            com /= len(neighbors)
                            
                            normal_vector = com - ca_atom.coord
                            
                            normal_length = np.linalg.norm(normal_vector)
                            
                            avg_vector = np.zeros(3)
                            for atom in atom_list:
                                avg_vector += (atom.coord - ca_atom.coord)
                            avg_vector /= len(atom_list)
                            
                            dot_product = np.dot(normal_vector, avg_vector)
                            sign = -1 if dot_product < 0 else 1
                            
                            curvature_dict[chain_id][res_pos] = sign * normal_length
                        else:
                            curvature_dict[chain_id][res_pos] = 0.0
                    else:
                        curvature_dict[chain_id][res_pos] = None
                        
                except Exception as e:
                    curvature_dict[chain_id][res_pos] = None
    
    return curvature_dict

def get_structural_neighbors(structure, distance_threshold=5.0):
    """Return a dictionary with information about neigbors for each residue taking into account each position and chain;
    it is based on the distance between residues"""

    neighbors_dict = {}
    contact_freq_dict = {}  
    
    atom_list = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atom_list)
    
    all_residues = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in all_residues:
                all_residues[chain_id] = {}
                neighbors_dict[chain_id] = {}
                contact_freq_dict[chain_id] = {}
            
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                
                res_pos = residue.id[1]
                all_residues[chain_id][res_pos] = residue
    
    for chain_id, residues in all_residues.items():
        for res_pos, residue in residues.items():
            neighbors_dict[chain_id][res_pos] = []
            
            contact_count = 0
            total_atoms = 0
            
            for atom in residue:
                total_atoms += 1
                
                neighbors = ns.search(atom.coord, distance_threshold, level='A')
                
                for neighbor_atom in neighbors:
                    if neighbor_atom.get_parent() != residue:  
                        contact_count += 1
            
            contact_freq = contact_count / max(1, total_atoms)
            contact_freq_dict[chain_id][res_pos] = contact_count
            
            ca_neighbors = []
            if 'CA' in residue:
                ca_atom = residue['CA']
                neighbor_atoms = ns.search(ca_atom.coord, distance_threshold, level='R')
                
                for neighbor_res in neighbor_atoms:
                    if neighbor_res != residue and neighbor_res.id[0] == " ": 
                        neighbor_chain = neighbor_res.get_parent().id
                        neighbor_pos = neighbor_res.id[1]
                        
                        ca_neighbors.append(f"{neighbor_chain}:{neighbor_pos}")
            
            neighbors_dict[chain_id][res_pos] = ",".join(ca_neighbors[:10]) if ca_neighbors else ""
    
    return neighbors_dict, contact_freq_dict

def calculate_bfactor_and_polar_density(structure):
    """Calculate and extract the b factor and the polar density for each residue in each pdb file."""

    bfactor_dict = {}
    polar_density_dict = {}
    
    atom_list = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atom_list)
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in bfactor_dict:
                bfactor_dict[chain_id] = {}
                polar_density_dict[chain_id] = {}
                
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                    
                res_pos = residue.id[1]
                
                bfactor_sum = 0.0
                atom_count = 0
                
                for atom in residue:
                    if hasattr(atom, "bfactor") and atom.bfactor is not None:
                        bfactor_sum += atom.bfactor
                        atom_count += 1
                
                if atom_count > 0:
                    bfactor_dict[chain_id][res_pos] = bfactor_sum / atom_count
                else:
                    bfactor_dict[chain_id][res_pos] = None
                
                try:
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        neighbor_residues = ns.search(ca_atom.coord, 8.0, level='R')
                        
                        total_neighbors = 0
                        polar_charged_count = 0
                        
                        for neighbor in neighbor_residues:
                            if neighbor != residue and neighbor.id[0] == " ":
                                total_neighbors += 1
                                
                                res_name = neighbor.get_resname()
                                one_letter = AA_TRANSLATION.get(res_name, "X")
                                
                                if one_letter in POLAR_CHARGED:
                                    polar_charged_count += 1
                        
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

def store_feature_value(pdb_dir, output_file, skip_depth=False, batch_size=100):
    """Store all predefined features for each residue of each pdb file."""

    binding_sites_dict = {}

    try:
        if os.path.exists('BioLiP_nr.txt'):
            df = pd.read_csv('BioLiP_nr.txt', delimiter='\t', low_memory=False, header=None)
            subset_df = df[[0, 1, 7]]  
            subset_df.columns = ['protein_name', 'chain_id', 'binding_site_sequence']  
            binding_sites_dict = extract_bs(subset_df)
            print("Successfully loaded BioLiP data")
    except Exception as e:
        print(f"Error loading BioLiP data: {str(e)}")

    all_features = []
    total_files = 0
    processed_files = 0
    
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb"):
            total_files += 1
    
    print(f"Found {total_files} .pdb files to process")
    
    batch_counter = 0
    batch_number = 1
    
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb"):
            processed_files += 1
            batch_counter += 1
            pdb_path = os.path.join(pdb_dir, filename)
            
            pdb_name = filename.rsplit('.pdb', 1)[0]
            if pdb_name.startswith("clean_pdb"):
                pdb_name = pdb_name[9:]  
            
            print(f"Processing file {processed_files}/{total_files}: {filename} -> {pdb_name}")
            
            parser = PDBParser(QUIET=True)
            try:
                structure = parser.get_structure(pdb_name, pdb_path)
                
                sec_struct_dict, asa_dict = extract_secondary_structure(pdb_path, structure)
                
                depth_dict = {}
                if not skip_depth:
                    depth_dict = calculate_residue_depths(structure, pdb_path)
                
                curvature_dict = calculate_curvature(structure)
                
                neighbors_dict, contact_freq_dict = get_structural_neighbors(structure)
                
                bfactor_dict, polar_density_dict = calculate_bfactor_and_polar_density(structure)
                
                residues = parse_pdb(pdb_path)
                
                if not residues:
                    print(f"Warning: No residues found in {filename}")
                    continue
                
                for position, residue, chain in residues:
                    properties = get_residue_properties(residue)
                    
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
            
            if batch_counter >= batch_size:
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
                
                all_features = []
                batch_counter = 0
                batch_number += 1
           
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

    pdb_directory = "0_Raw_Data/ml_pdb"
    output_file="binding_features.csv"

    clean_directory = None
    clean_only = False
    
    skip_depth = False
    batch_size = 100  
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--skip-depth":
            skip_depth = True
            print("Skipping residue depth calculations")
        elif sys.argv[i] == "--batch-size" and i + 1 < len(sys.argv):
            try:
                batch_size = int(sys.argv[i + 1])
                i += 1  
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

    cleaned_files = clean_pdb_files(pdb_directory, clean_directory)
    
    if clean_only:
        print("Files cleaned.")
        sys.exit(0)
    
    if clean_directory:
        pdb_directory = clean_directory
    
    store_feature_value(pdb_directory, output_file, skip_depth, batch_size)