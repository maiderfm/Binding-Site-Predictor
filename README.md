# 🔬 Binding Site Predictor

Machine Learning tool to predict and visualize protein residues partaking in binding sites from protein structural data in PDB format.

---

## Description

**Binding Site Predictor** is a Python-based tool that uses machine learning techniques to identify and visualize amino acid residues involved in binding interactions within a protein's 3D structure (PDB format). This tool aims to support researchers in drug design, structural biology, and protein-ligand interaction studies.

####  Features

- Predicts binding residues from structural data
- Allows for visualization of the highlighted binding sites' 3D structures using third-party programs Chimera and Pymol 
- Provides easy-to-interpret output formats
- Customizable model and threshold options
- Supports batch analysis

---

##  Tutorial

#### Installation

Clone the repository:

```bash
git clone https://github.com/maiderfm/Binding-Site-Predictor.git
cd Binding-Site-Predictor
```

Set executable permissions:

```bash
chmod a+x Binding-Site-Predictor
```
Once you're in the project directory, install the required Python dependencies:

```bash
pip install -r requirements.txt
```
Additionally, for full functionality, you'll need to install these external tools:

- DSSP (mkdssp) - For secondary structure assignment
```bash
# On Ubuntu/Debian
sudo apt-get install dssp

# On macOS with Homebrew
brew install brewsci/bio/dssp
```

- MSMS (optional) - For surface and depth calculations

    Download from: https://ccsb.scripps.edu/mgltools/#msms
    Add to your PATH environment variable



Make sure these external tools are properly installed and accessible from your PATH for optimal prediction results.

#### Usage

Run the predictor with a PDB file:

```bash
python BindingSitePredictor.py path/to/structure.pdb --model model.pkl
```

Input can be PDB file(s), a directory of PDB files, or file containing list of PDB files. The files must have .pdb or .ent as file extensions for the program to run.

#### Optional Arguments
- `-h, --help`       Show help page
- `--model`          Path to a custom trained model (default: model.pkl)
- `--skip-depth`     Prediction skips residue depth calculation (default: true)
- `--clean-dir`      Directory to store cleaned PDB files (default: cleaned_pdbs)

---

## Output

After running the predictor, the output directory will contain the following files (all named based on your input PDB file basename):

- `basename_sites.pdb`  
  A modified PDB file with **predicted binding site residues** included. These residues are annotated and can be visualized using molecular viewers.

- `basename.cmd` *(UCSF Chimera script)*  
  A script for visualizing the original protein in **grey** and highlighting the predicted binding site residues in **red** using Chimera.

- `basename.pml` *(PyMOL script)*  
  A script for visualizing the original protein in **grey** and highlighting the predicted binding site residues in **red** using PyMOL.

- `basename_summary.txt`  
  A human-readable text summary containing:
  - A list of **Predicted Active Site Residues**
  - **Predicted Sites**, grouped by chain

This output allows you to both programmatically and visually assess the predicted binding residues for further analysis or presentation.

---

## Analysis example

In this example, we will use the **1a3n.pdb** file, which represents the structure of **Hemoglobin**, a protein responsible for oxygen transport in red blood cells. Hemoglobin binds to oxygen in the lungs and releases it in tissues that need it. Studying the binding sites in hemoglobin is critical for understanding diseases like sickle cell anemia and understanding how small molecules, such as drugs or oxygen-binding enhancers, might interact with it.

To predict the binding sites of hemoglobin using a custom-trained model, run the following command from inside the project directory:

```bash
python BindingSitePredictor.py ./ExampleAnalysis/1a3n.pdb --model random_forest_model.pkl
```
This command will generate the following files:

- **1a3n_sites.pdb** (with predicted binding site residues)

- **1a3n.cmd** (Chimera visualization script)

- **1a3n.pml** (PyMOL visualization script)

- **1a3n_summary.txt** (predicted active site residues)

The Chimera or PyMOL scripts can then be loaded into their respective programs for 3D visualization of the predicted binding sites.
```bash
chimera ./ExampleAnalysis/1a3n.cmd 
pymol ./ExampleAnalysis/1a3n.pml 
```
---



