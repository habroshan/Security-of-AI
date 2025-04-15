#!/usr/bin/env python3
import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and display its output."""
    print(f"\n=== {description} ===")
    print(f"Executing: {command}")
    
    try:
        process = subprocess.run(command, shell=True, check=True, 
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("Starting ML environment setup...")
    
    # Step 1: Install Python 3.10 and related development tools
    if not run_command("sudo apt install python3.10 python3.10-venv python3.10-dev -y", 
                     "Installing Python 3.10 and development tools"):
        print("Failed to install Python 3.10. Please check your system and try again.")
        sys.exit(1)
    
    # Step 2: Create a virtual environment
    if not run_command("python3.10 -m venv venv", 
                     "Creating virtual environment"):
        print("Failed to create virtual environment. Please check your Python installation and try again.")
        sys.exit(1)
    
    # Step 3: Activate the virtual environment and install packages
    # Note: We need to modify the shell environment, which can't be done directly
    # So we'll create an activation script that the user can run after this script completes
    
    # Install the packages within this script by using the venv's pip directly
    pip_path = os.path.join("venv", "bin", "pip")
    
    # Step 4: Install TensorFlow and essential libraries
    if not run_command(f"{pip_path} install tensorflow pandas scikit-learn nltk", 
                     "Installing TensorFlow and essential libraries"):
        print("Failed to install TensorFlow and essential libraries.")
        sys.exit(1)
    
    # Step 5: Install PyTorch
    if not run_command(f"{pip_path} install torch torchvision torchaudio", 
                     "Installing PyTorch"):
        print("Failed to install PyTorch.")
        sys.exit(1)
    
    # Step 6: Install Hugging Face Transformers
    if not run_command(f"{pip_path} install transformers", 
                     "Installing Hugging Face Transformers"):
        print("Failed to install Transformers.")
        sys.exit(1)
    
    # Create an activation script for the user
    with open("activate_env.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("source venv/bin/activate\n")
        f.write('echo "ML environment activated! Use deactivate to exit."\n')
    
    run_command("chmod +x activate_env.sh", "Making activation script executable")
    
    print("\n=== Setup Complete! ===")
    print("To activate your virtual environment, run:")
    print("source activate_env.sh")
    print("\nYour ML environment has been set up with:")
    print(" - Python 3.10")
    print(" - TensorFlow")
    print(" - PyTorch, torchvision, torchaudio")
    print(" - Pandas, scikit-learn, nltk")
    print(" - Hugging Face Transformers")

if __name__ == "__main__":
    main()
