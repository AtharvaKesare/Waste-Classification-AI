import os
import zipfile

def zip_project(source_dir, output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Exclude folders
            dirs[:] = [d for d in dirs if d not in ['.git', 'venv', '__pycache__', 'Dataset', 'Train', 'Test']]
            
            for file in files:
                # Exclude specific files and large dataset images
                if file.endswith('.db') or file.endswith('.sqlite3') or file.endswith('.hdf5') or file.endswith('.tflite') or file == output_filename or file == 'zip_project.py' or 'Dataset' in root:
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
                print(f"Added {arcname}")

if __name__ == '__main__':
    zip_project('.', 'waste_project_ready_to_deploy.zip')
    print("Successfully created waste_project_ready_to_deploy.zip!")
