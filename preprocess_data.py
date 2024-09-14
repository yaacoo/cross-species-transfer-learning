import pandas as pd
import numpy as np
import scanpy as sc 
import matplotlib.pyplot as plt
import pickle as pkl
import re

# Set scanpy settings
sc.settings.verbosity = 3
sc.logging.print_header()

# Define paths to target data (HCL) and source data (MCA):
path_MCA="../data/MCA1.1_adata.h5ad"
path_HCL="../data/HCL_Fig1_adata.h5ad"

# Read cell atlas data (Mouse, Human) AnnData objects:
MCA=sc.read_h5ad(path_MCA)
HCL=sc.read_h5ad(path_HCL)

# Load cell type annotations:
obs_annotations_MCA=pd.read_excel("../data/MCA1.1_cell_info.xlsx", index_col=0)
obs_annotations_HCL=pd.read_excel("../data/HCL_Fig1_cell_Info.xlsx", index_col=0)

# Create a dictionary of cell types from the two cell atlases:
cellTypeDictionary= (set( list(obs_annotations_MCA.celltype.unique()) + list(obs_annotations_HCL.celltype.unique()) ))
cellTypeDictionary= {x: x.lower() for x in cellTypeDictionary}

# Match cell type names:
cellTypeDictionary={'Cd8+ T cell': 'T',
 'Fetal epithelial progenitor': 'epithelial progenitor',
 'Mesenchymal stem cell': 'mesenchymal',
 'Female fetal gonad cell': 'female gonad',
 'Myocyte': 'myocyte',
 'Pancreatic endocrine cell': 'endocrine',
 'Enterocyte': 'enterocyte',
 'Chondrocyte': 'chondrocyte',
 'Acinar cell': 'acinar',
 'Fetal chondrocyte': 'chondrocyte',
 'Alveolar type II cell': 'alveolar type ii',
 'Fetal Proliferating cell': 'proliferating',
 'Proximal tubule brush border cell': 'proximal tubule brush border',
 'Endothelial cell (APC)': 'endothelial',
 'Neutrophil': 'neutrophil',
 'Endothelial cell': 'endothelial',
 'Fetal Neuron': 'neuron',
 'Hepatocyte/Endodermal cell': 'hepatocyte',
 'M2 Macrophage': 'macrophage',
 'Fetal stromal cell': 'mesenchymal',
 'Cardiocyte': 'cardiomyocyte',
 'Adrenal gland inflammatory cell': 'adrenal inflammatory',
 'Antigen presenting cell (RPS high)': 'APC',
 'hESC': 'hESC',
 'Embryonic stem cell': 'embryonic stem',
 'Ureteric bud cell': 'ureteric bud',
 'Mast cell': 'mast',
 'Adipocyte': 'adipocyte',
 'B cell (Plasmocyte)': 'B plasmocyte',
 'Mesothelial cell': 'mesothelial',
 'Fetal premeiotic germ cell': 'premeiotic germ',
 'Epithelial cell (intermediated)': 'intermediated',
 'Fetal kidney epithelial cell': 'epithelial',
 'Basal cell': 'basal',
 'B cell': 'B',
 'Sinusoidal endothelial cell': 'endothelial',
 'Microglial': 'microglial',
 'Cumulus cell': 'cumulus',
 'Enterocyte progenitor': 'enterocyte progenitor',
 'Pancreas exocrine cell': 'pancreas exocrine',
 'Neuron': 'neuron',
 'Fetal endocrine cell': 'endocrine',
 'Testicular cell': 'testicular',
 'Smooth muscle cell': 'smooth muscle',
 'Pre sertoli cell': 'pre sertoli',
 'Monocyte': 'monocyte',
 'Oligodendrocyte': 'oligodendrocyte',
 'Mammary gland in lactation': 'mammary',
 'Gastric endocrine cell': 'gastric endocrine',
 'Small luteal cell': 'small luteal',
 'Fetal neuron': 'neuron',
 'Hepatocyte': 'hepatocyte',
 'T cell': 'T',
 'Fasciculata cell': 'fasciculata',
 'Intercalated cells of collecting duct': 'intercalated',
 'Epithelial cell': 'epithelial',
 'Fetal fibroblast': 'fibroblast',
 'Lymphocyte': 'lymphocyte',
 'Myeloid cell': 'myeloid',
 'Fetal enterocyte ': 'enterocyte',
 'Erythroid cell': 'erythroid',
 'CB CD34+': 'CB cd34+',
 'Endocrine cell': 'endocrine',
 'Astrocyte': 'astrocyte',
 'Dendritic cell': 'dendritic',
 'Goblet cell': 'goblet',
 'Invasive spongiotrophoblast': 'invasive spongiotrophoblast',
 'Secretory epithelial cell': 'secretory epithelial',
 'Spermicyte': 'spermicyte',
 'Stratified epithelial cell': 'stratified epithelial',
 'Fibroblast': 'fibroblast',
 'Proliferating T cell': 'T',
 'Neutrophil (RPS high)': 'neutrophil',
 'Immature sertoli cell (Pre-Sertoli cell)': 'pre-sertoli',
 'Kidney intercalated cell': 'intercalated',
 'AT2 cell': 'at2',
 'Gastric chief cell': 'gastric chief',
 'Ventricle cardiomyocyte': 'cardiomyocyte',
 'Myoloid cell': 'myoloid',
 'Endothelial cell (endothelial to mesenchymal transition)': 'endothelial',
 'Fetal Premeiotic germ cell': 'premeiotic germ',
 'Macrophage': 'macrophage',
 'Intercalated cell': 'intercalated',
 'Stromal cell': 'mesenchymal',
 'Proximal tubule progenitor': 'proximal tubule progenitor',
 'B cell(Plasmocyte)': 'B plasmocyte',
 'Fetal mesenchymal progenitor': 'mesenchymal',
 'Cd4+ T cell': 'T',
 'Erythroid progenitor cell (RP high)': 'erythroid progenitor',
 'Fetal acinar cell': 'acinar',
 'Spermatogonia': 'spermatogonia',
 'Loop of Henle': 'loop of henle',
 'Fetal skeletal muscle cell': 'skeletal muscle',
 'Thyroid follicular cell': 'follicular',
 'Primordial germ cell': 'primordial germ',
 'Intermediated cell': 'intermediated',
 'Prostate epithelial cell': 'epithelial'}

# Change the cellype values by the cellTypeDictionary dictionary:
obs_annotations_MCA.celltype= obs_annotations_MCA.celltype.map(cellTypeDictionary)
obs_annotations_HCL.celltype= obs_annotations_HCL.celltype.map(cellTypeDictionary)

# Rename the column "sample" to "tissue" in HCL:
obs_annotations_HCL.rename(columns={'sample': 'tissue'}, inplace=True)

# remove the "-0" or "-1" suffix from each cell name in HCL:
HCL.obs.index=HCL.obs.index.str.replace(r'-\d', '')

# Verify that cells are in the same order in the annotations dataframe and the AnnData object:
print(all (obs_annotations_MCA.index == MCA.obs.index ))
print(all (obs_annotations_HCL.index == HCL.obs.index ))

# Add the cell type and tissue anotations to the AnnData objects:
HCL.obs['celltype'] = obs_annotations_HCL.celltype
MCA.obs['celltype'] = obs_annotations_MCA.celltype
MCA.obs['tissue'] = obs_annotations_MCA.tissue

# Match the feature (gene) names between the datasets- change all to upper case:
HCL.var.index = HCL.var.index.str.upper()
MCA.var.index = MCA.var.index.str.upper()

# Find the intersection of the feature set (genes):
intersection= set(list(MCA.var.index)).intersection(set(list(HCL.var.index)))

# Keep only the intersection of the features (genes):
HCL = HCL[:, HCL.var.index.isin(intersection)]
MCA = MCA[:, MCA.var.index.isin(intersection)]

# Check if there is the same number of features, or there are duplicated columns:
print("HCL features:"+ str(HCL.shape[1]) + ", MCA features:" + str(MCA.shape[1]))

# Remove the duplicated columns (features):
HCL = HCL[:, ~HCL.var.index.duplicated()]
MCA = MCA[:, ~MCA.var.index.duplicated()]

print("HCL features:"+ str(HCL.shape[1]) + ", MCA features:" + str(MCA.shape[1]))

# Save processed data per tissue as pickle files:
# Get a list of the tissues that appear in both datasets (intersection) and save as pickle:
tissues = list(set(list(MCA.obs.tissue)).intersection(set(list(HCL.obs.tissue))))
with open("../data/list_of_tissues.pkl", 'wb') as f:
    pkl.dump(tissues, f)

# Process each tissue separately:
for current_tissue in tissues:
    print(current_tissue)
    # Create a temporary object for each tissue:
    MCA_current_tissue=MCA[MCA.obs.tissue==current_tissue]
    HCL_current_tissue=HCL[HCL.obs.tissue==current_tissue]
    # Create a list of cell types (labels) that have at least 50 cells in each dataset:
    cell_types = list(set(list(MCA_current_tissue.obs.celltype)).intersection(set(list(HCL_current_tissue.obs.celltype))))
    cell_types = [x for x in cell_types if (MCA_current_tissue.obs.celltype==x).sum()>=50 and (HCL_current_tissue.obs.celltype==x).sum()>=50]
    # Skip the tissue if there are less than 3 cell types:
    if len(cell_types)<3:
        continue
    # Filter the current tissue object by the cell types list to exclude cell types with less than 50 cells:
    MCA_current_tissue=MCA_current_tissue[MCA_current_tissue.obs.celltype.isin(cell_types)]
    HCL_current_tissue=HCL_current_tissue[HCL_current_tissue.obs.celltype.isin(cell_types)]
    # Save a bar plot of the number of cells per cell type:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    MCA_current_tissue.obs.celltype.value_counts().plot(kind='bar', ax=ax[0])
    HCL_current_tissue.obs.celltype.value_counts().plot(kind='bar', ax=ax[1])
    ax[0].set_title('MCA')  
    ax[1].set_title('HCL')
    # Add a title of the tissue name:
    # Create a formated current_tissue name that has a space before an upper case letter:
    current_tissue_formated = re.sub(r'([A-Z])', r' \1', current_tissue)
    fig.suptitle(current_tissue_formated, fontsize=16)
    # Save the plot as a PDF file:
    fig.savefig("../figures/"+current_tissue + '_cell_types_counts.pdf', bbox_inches='tight')
    # Create a "y" vector for the cell types:
    MCA_y= MCA_current_tissue.obs.celltype
    HCL_y= HCL_current_tissue.obs.celltype
    # Change the AnnData objects to dataframes:
    MCA_current_tissue=MCA_current_tissue.to_df()
    HCL_current_tissue=HCL_current_tissue.to_df()
    # Reorder the columns (features) based on their alphabetical order to match in both datasets:
    MCA_current_tissue=MCA_current_tissue.reindex(sorted(MCA_current_tissue.columns), axis=1)
    HCL_current_tissue=HCL_current_tissue.reindex(sorted(HCL_current_tissue.columns), axis=1)
    # Check if the columns are in the same order:
    print(all (MCA_current_tissue.columns == HCL_current_tissue.columns ))
    # Save dataframes as pickle objects:
    MCA_current_tissue.to_pickle("../data/"+current_tissue+"_MCA.pkl")
    HCL_current_tissue.to_pickle("../data/"+current_tissue+"_HCL.pkl")
    # Save lables (y) as pickle objects:
    MCA_y.to_pickle("../data/"+current_tissue+"_MCA_y.pkl")
    HCL_y.to_pickle("../data/"+current_tissue+"_HCL_y.pkl")