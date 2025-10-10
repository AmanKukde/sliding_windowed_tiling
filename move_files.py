import os
import shutil
from tqdm import tqdm

# Configuration
MODALITIES = {
    "PAVIA_ATN": "ActTub ActNuc TubNuc",
    "Hagen": "MitoVsAct"
}

SOURCE_ROOT = "/group/jug/aman/ConsolidatedResults/Results_usplit_64"
DEST_ROOT = "/group/jug/aman/usplit-results/"
LC_TYPES = ["DeepLC", "LeanLC"]
FILENAMES = ["pred_test_dset_usplit_og.tiff", "pred_test_dset_usplit_og.pkl"]

# Count total number of iterations for progress bar
total_files = sum(len(mod.split()) * len(LC_TYPES) * len(FILENAMES) for mod in MODALITIES.values())

# Progress bar
with tqdm(total=total_files, desc="Copying files") as pbar:
    for dataset, modalities_str in MODALITIES.items():
        modalities = modalities_str.split()
        for modality in modalities:
            for lc in LC_TYPES:
                source_dir = os.path.join(SOURCE_ROOT, dataset, modality, lc)
                dest_dir = os.path.join(DEST_ROOT, dataset, modality, lc)
                
                # Create destination directory if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)
                
                for filename in FILENAMES:
                    source_file = os.path.join(source_dir, filename)
                    dest_file = os.path.join(dest_dir, filename)
                    
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, dest_file)
                    # Optionally, you can print a warning for missing files
                    # else:
                    #     print(f"File not found, skipping: {source_file}")
                    
                    pbar.update(1)
print("DONE!")