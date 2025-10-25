import os
import zipfile

def extract_zip_files(source_dir):
    """
    Extract zip files from ViLaSR-data-hf/ to its subfolders:
    cold_start_part*.zip -> ViLaSR-data-hf/cold_start/
    rl_part*.zip -> ViLaSR-data-hf/rl/
    """
    os.makedirs(os.path.join(source_dir, 'cold_start'), exist_ok=True)
    os.makedirs(os.path.join(source_dir, 'reflective_rejection_sampling'), exist_ok=True)
    os.makedirs(os.path.join(source_dir, 'rl'), exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith('.zip'):
            zip_path = os.path.join(source_dir, filename)
            
            if filename.startswith('cold_start_part'):
                extract_dir = os.path.join(source_dir, 'cold_start')
            elif filename.startswith('reflective_rejection_sampling_part'):
                extract_dir = os.path.join(source_dir, 'reflective_rejection_sampling')
            elif filename.startswith('rl_part'):
                extract_dir = os.path.join(source_dir, 'rl')
            else:
                continue

            print(f"Extracting: {zip_path} to {extract_dir}/")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Successfully extracted: {zip_path}")
            except Exception as e:
                print(f"Failed to extract {zip_path}: {e}")

if __name__ == '__main__':
    # Specify path as ViLaSR-data
    source_directory = "ViLaSR-data"
    extract_zip_files(source_directory)
