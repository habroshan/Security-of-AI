#!/bin/bash

# AI Security Lab Setup Script
# This script sets up the entire AI Security laboratory environment
# including Python dependencies and React applications.

# Text styling
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${BOLD}${GREEN}$1${NC}\n"
    echo "==============================================="
}

# Function to print info messages
print_info() {
    echo -e "${YELLOW}$1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}$1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if script is run from the project root directory
if [[ ! -d "AI_Attacks_Material" || ! -d "AI_Driven_Attacks" || ! -d "Lab_Demonstration" ]]; then
    print_error "Error: Please run this script from the project root directory!"
    exit 1
fi

print_header "Starting AI Security Lab Setup"

# Create logs directory
mkdir -p logs

# Check and install system dependencies
print_header "Checking System Dependencies"

# Install Python 3.10 specifically
print_info "Installing Python 3.10..."
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Get Python version
if command_exists python3.10; then
    PYTHON_VERSION=$(python3.10 --version)
    print_info "Found Python version: $PYTHON_VERSION"
else
    print_error "Python 3.10 installation failed. Please install manually."
    exit 1
fi

# Install pip for Python 3.10
print_info "Installing pip for Python 3.10..."
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10

# Check for nodejs and npm
if ! command_exists node; then
    print_info "Node.js not found. Installing Node.js and npm..."
    sudo apt-get update
    sudo apt-get install -y curl
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    print_info "Node.js $(node --version) and npm $(npm --version) installed."
else
    print_info "Found Node.js $(node --version) and npm $(npm --version)"
fi

# Check for system libraries
print_info "Installing required system libraries..."
sudo apt-get update
sudo apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 build-essential

# Create Python virtual environment
print_header "Setting up Python Virtual Environment"

# Remove existing venv if it exists (to fix potential issues)
if [ -d "venv" ]; then
    print_info "Removing existing virtual environment to create a fresh one..."
    rm -rf venv
fi

# Create a new virtual environment with Python 3.10
print_info "Creating new virtual environment with Python 3.10..."
python3.10 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Verify the venv is activated correctly
which python
which pip

# Create requirements.txt
print_info "Creating comprehensive requirements.txt..."
cat > requirements.txt << EOL
# Core ML libraries - essential for model deployment
numpy
pandas
matplotlib
scipy
scikit-learn

# Deep Learning frameworks - now compatible since we're using Python 3.10
torch
torchvision
tensorflow

# Image processing
opencv-python
Pillow

# Data visualization
seaborn
plotly

# Web and API
flask
flask-cors
requests
werkzeug
gunicorn

# Jupyter and development
jupyter
ipykernel
notebook
ipywidgets

# NLP libraries - for phishing demos
nltk
transformers
gensim

# Utilities and data handling
tqdm
h5py
pyyaml
joblib
dill
cloudpickle

# Feature engineering
category_encoders
feature-engine
EOL

print_info "Installing Python packages (this may take a while)..."
# Update pip first
pip install --upgrade pip

# Install packages with more detailed output
pip install -r requirements.txt || {
    print_error "Error: Some packages failed to install. Attempting individual installation..."
    
    # Try installing core packages individually
    pip install numpy
    pip install pandas matplotlib
    pip install scipy scikit-learn
    pip install flask flask-cors requests
    pip install jupyter ipykernel
    pip install torch torchvision
    pip install tensorflow
    pip install tqdm h5py pyyaml
    pip install opencv-python Pillow
    
    print_info "Core packages installed. Some optional packages may be missing."
}

# Set up Jupyter kernel
print_info "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=venv --display-name="AI Security Lab"

# Download NLTK data
print_info "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Set up React applications
print_header "Setting up React Applications"

REACT_APPS=(
    "Lab_Demonstration/1. evasion_attack_demo"
    "Lab_Demonstration/2. poisoning-attack-demo"
    "Lab_Demonstration/3. backdoor-attack-demo"
    "Lab_Demonstration/4. model-stealing-demo"
    "Lab_Demonstration/5. adversarial_attack"
)

for app_dir in "${REACT_APPS[@]}"; do
    if [ -d "$app_dir" ]; then
        print_info "Setting up React app in $app_dir..."
        
        # Navigate to app directory
        cd "$app_dir"
        
        # Check if package.json exists
        if [ ! -f "package.json" ]; then
            print_info "Initializing React app..."
            # Create a temporary log file
            log_file="../../logs/react_init_temp.log"
            mkdir -p "../../logs"
            
            npx create-react-app temp-app > "$log_file" 2>&1
            
            # Copy node_modules and package files
            cp -r temp-app/node_modules . 2>/dev/null || mkdir node_modules
            cp temp-app/package.json temp-app/package-lock.json . 2>/dev/null
            
            # Clean up
            rm -rf temp-app
        else
            print_info "package.json found, installing dependencies..."
        fi
        
        # Install dependencies
        npm install > "../../logs/npm_install_${app_dir//[^a-zA-Z0-9]/_}.log" 2>&1
        
        # Install specific dependencies based on the demo type
        base_name=$(basename "$app_dir")
        if [[ "$base_name" == "1. evasion_attack_demo" ]]; then
            npm install @mui/material @mui/icons-material @emotion/react @emotion/styled chart.js react-chartjs-2 axios > "../../logs/npm_deps_evasion.log" 2>&1
        elif [[ "$base_name" == "2. poisoning-attack-demo" ]]; then
            npm install @mui/material @mui/icons-material @emotion/react @emotion/styled d3 axios > "../../logs/npm_deps_poisoning.log" 2>&1
        elif [[ "$base_name" == "3. backdoor-attack-demo" ]]; then
            npm install @mui/material @mui/icons-material @emotion/react @emotion/styled three @react-three/fiber @react-three/drei axios > "../../logs/npm_deps_backdoor.log" 2>&1
        elif [[ "$base_name" == "4. model-stealing-demo" ]]; then
            npm install @mui/material @mui/icons-material @emotion/react @emotion/styled recharts axios > "../../logs/npm_deps_stealing.log" 2>&1
        elif [[ "$base_name" == "5. adversarial_attack" ]]; then
            npm install @mui/material @mui/icons-material @emotion/react @emotion/styled three @react-three/fiber @react-three/drei chart.js react-chartjs-2 axios > "../../logs/npm_deps_adversarial.log" 2>&1
        fi
        
        # Go back to project root
        cd ../..
    else
        print_error "Directory $app_dir not found. Skipping."
    fi
done

# Set up model directories
print_header "Setting up Model Directories"

MODEL_DIRS=(
    "AI_Attacks_Material/Backdoor_Attack/models"
    "AI_Attacks_Material/Evasion_Attack/models"
    "AI_Attacks_Material/Model_Stealing_Attack/models"
    "AI_Attacks_Material/Poisoning_Attack/models"
    "AI_Attacks_Material/Realizable_Attack/models"
    "AI_Driven_Attacks/AI_ML_Phishing/models"
    "AI_Driven_Attacks/Deep_Fake_Attack/models"
)

for dir in "${MODEL_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        print_info "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Create initialization script for each module
print_info "Creating initialization scripts for models..."

# Initialize Backdoor_Attack
cat > AI_Attacks_Material/Backdoor_Attack/initialize_models.py << EOL
import os
import numpy as np
import pickle

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.datasets import mnist
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. This should not happen with Python 3.10 setup.")
    print("Attempting to install TensorFlow now...")
    import subprocess
    subprocess.call(['pip', 'install', 'tensorflow'])
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.datasets import mnist
        print("TensorFlow successfully installed!")
        TF_AVAILABLE = True
    except ImportError:
        print("TensorFlow still not available. Using PyTorch instead.")
        TF_AVAILABLE = False
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision

# Set random seed for reproducibility
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)
else:
    torch.manual_seed(42)

def initialize_model():
    print("Initializing MNIST model for Backdoor Attack module...")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Create model info
    model_info = {
        'input_shape': (28, 28, 1),
        'class_names': [str(i) for i in range(10)],
        'test_accuracy': 0.0,
        'preprocessing': 'normalize between 0 and 1'
    }
    
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    # If TensorFlow is available, create TF model
    if TF_AVAILABLE:
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Preprocess data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)
        
        # Build model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Save untrained model
        model.save('models/mnist_cnn_model.h5')
        model.save('models/mnist_cnn_model.keras')
    else:
        # Create a placeholder file to indicate model creation was attempted
        with open('models/model_creation_attempted.txt', 'w') as f:
            f.write("TensorFlow model creation was attempted but TensorFlow is not available.\n")
            f.write("Please install TensorFlow manually to use TensorFlow models.\n")
    
    print("Model initialization complete!")
    
if __name__ == "__main__":
    initialize_model()
EOL

# Copy to other modules
cp AI_Attacks_Material/Backdoor_Attack/initialize_models.py AI_Attacks_Material/Evasion_Attack/initialize_models.py
cp AI_Attacks_Material/Backdoor_Attack/initialize_models.py AI_Attacks_Material/Poisoning_Attack/initialize_models.py
cp AI_Attacks_Material/Backdoor_Attack/initialize_models.py AI_Attacks_Material/Realizable_Attack/initialize_models.py

# Initialize Model_Stealing_Attack
cat > AI_Attacks_Material/Model_Stealing_Attack/initialize_models.py << EOL
import os
import numpy as np
import json

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
    from tensorflow.keras.datasets import mnist
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. This should not happen with Python 3.10 setup.")
    print("Attempting to install TensorFlow now...")
    import subprocess
    subprocess.call(['pip', 'install', 'tensorflow'])
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
        from tensorflow.keras.datasets import mnist
        print("TensorFlow successfully installed!")
        TF_AVAILABLE = True
    except ImportError:
        print("TensorFlow still not available. Using PyTorch instead.")
        TF_AVAILABLE = False
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision

def initialize_model():
    print("Initializing target model for Model Stealing Attack module...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load MNIST dataset using numpy directly to create test data
    # This will work regardless of TF/PyTorch availability
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except:
        try:
            from torchvision.datasets import MNIST
            from torchvision import transforms
            mnist_test = MNIST('./', train=False, download=True, transform=transforms.ToTensor())
            x_test = np.array([np.array(data[0]).reshape(28, 28) for data in mnist_test])
            y_test = np.array([data[1] for data in mnist_test])
        except:
            # Create dummy data if neither is available
            print("Creating dummy test data since neither TensorFlow nor PyTorch could load MNIST")
            x_test = np.random.rand(100, 28, 28)
            y_test = np.random.randint(0, 10, size=100)
    
    # Save some test examples for demonstration
    np.save('models/test_examples.npy', x_test[:100])
    np.save('models/test_labels.npy', y_test[:100])
    
    # Save metadata
    metadata = {
        'model_name': 'MNIST Target Model',
        'input_shape': [28, 28, 1],
        'num_classes': 10,
        'preprocessing': 'normalize to [0,1]'
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # If TensorFlow is available, create TF model
    if TF_AVAILABLE:
        # Preprocess data
        x_train = x_train.astype('float32') / 255.0
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # Build the target model
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.25),
            Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        # Save the model
        model.save('models/target_mnist_model.keras')
    else:
        # Create a placeholder file to indicate model creation was attempted
        with open('models/model_creation_attempted.txt', 'w') as f:
            f.write("TensorFlow model creation was attempted but TensorFlow is not available.\n")
            f.write("Please install TensorFlow manually to use TensorFlow models.\n")
    
    print("Model initialization complete!")

if __name__ == "__main__":
    initialize_model()
EOL

# Initialize scripts for AI Driven Attacks
cat > AI_Driven_Attacks/AI_ML_Phishing/initialize_data.py << EOL
import os
import json
import random
import numpy as np

def initialize_data():
    print("Initializing data for AI-Driven Phishing module...")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('templates/email', exist_ok=True)
    os.makedirs('templates/sms', exist_ok=True)
    
    # Create sample email templates
    email_templates = [
        {
            'name': 'password_reset',
            'content': """
            <html>
            <body>
            <h2>Important: Your Password Needs to be Reset</h2>
            <p>Dear {{name}},</p>
            <p>We have detected unusual activity on your account. For your security, please reset your password immediately by clicking the link below:</p>
            <p><a href="{{reset_link}}">Reset Password</a></p>
            <p>If you did not request this change, please contact our support team immediately.</p>
            <p>Best regards,<br>Security Team</p>
            </body>
            </html>
            """
        },
        {
            'name': 'account_verification',
            'content': """
            <html>
            <body>
            <h2>Verify Your Account</h2>
            <p>Hello {{name}},</p>
            <p>Thank you for creating an account with us. To complete the registration process, please verify your account by clicking the link below:</p>
            <p><a href="{{verification_link}}">Verify Account</a></p>
            <p>This link will expire in 24 hours.</p>
            <p>Regards,<br>Account Team</p>
            </body>
            </html>
            """
        }
    ]
    
    # Save email templates
    for template in email_templates:
        with open(f'templates/email/{template["name"]}.html', 'w', encoding='utf-8') as f:
            f.write(template['content'])
    
    # Create sample SMS templates
    sms_templates = [
        {
            'name': 'package_delivery',
            'content': "{{name}}, your package could not be delivered. Reschedule delivery here: {{tracking_link}}"
        },
        {
            'name': 'bank_alert',
            'content': "ALERT: Unusual activity detected on your {{bank_name}} account. Verify your identity here: {{verification_link}}"
        }
    ]
    
    # Save SMS templates
    for template in sms_templates:
        with open(f'templates/sms/{template["name"]}.txt', 'w', encoding='utf-8') as f:
            f.write(template['content'])
    
    # Create sample target data
    sample_targets = [
        {
            'name': 'John Smith',
            'email': 'john.smith@example.com',
            'phone': '555-123-4567',
            'company': 'ABC Corp',
            'position': 'Manager',
            'department': 'IT',
            'interests': 'technology, security, cloud computing'
        },
        {
            'name': 'Jane Doe',
            'email': 'jane.doe@example.com',
            'phone': '555-234-5678',
            'company': 'XYZ Inc',
            'position': 'Developer',
            'department': 'Engineering',
            'interests': 'programming, data science, AI'
        },
        {
            'name': 'Robert Johnson',
            'email': 'robert.j@example.com',
            'phone': '555-345-6789',
            'company': 'Acme LLC',
            'position': 'Analyst',
            'department': 'Finance',
            'interests': 'finance, analytics, reporting'
        }
    ]
    
    # Save target data
    with open('data/sample_targets.json', 'w', encoding='utf-8') as f:
        json.dump(sample_targets, f, indent=2)
    
    print("Data initialization complete!")

if __name__ == "__main__":
    initialize_data()
EOL

cat > AI_Driven_Attacks/Deep_Fake_Attack/initialize_data.py << EOL
import os
import numpy as np
import json

def initialize_data():
    print("Initializing data for Deep Fake Attack module...")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/samples', exist_ok=True)
    
    # Create metadata file
    metadata = {
        'dataset_name': 'Demo DeepFake Dataset',
        'required_models': [
            'face_recognition',
            'face_swap_model',
            'deepfake_detector'
        ],
        'instructions': 'This is a demo dataset for educational purposes. Real models need to be downloaded separately.'
    }
    
    with open('data/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Create a readme
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write("""# DeepFake Attack Module

This module demonstrates DeepFake creation and detection techniques for educational purposes.

## Setup

1. Initialize the environment:
   ```
   python initialize_data.py
   ```

2. Download required models:
   ```
   python download_models.py
   ```

3. Run the demo:
   ```
   python deepfake_demo.py
   ```

## Note

This is for educational purposes only. Creating and distributing DeepFakes without permission may be illegal in many jurisdictions.
""")
    
    print("Data initialization complete!")

if __name__ == "__main__":
    initialize_data()
EOL

# Create a script to initialize all models
print_info "Creating script to initialize all models..."
cat > initialize_all_models.sh << EOL
#!/bin/bash

echo "Initializing models for all modules..."

# Activate virtual environment
source venv/bin/activate

# Initialize models for each attack module
echo "Initializing Backdoor Attack models..."
cd AI_Attacks_Material/Backdoor_Attack
python initialize_models.py
cd ../..

echo "Initializing Evasion Attack models..."
cd AI_Attacks_Material/Evasion_Attack
python initialize_models.py
cd ../..

echo "Initializing Poisoning Attack models..."
cd AI_Attacks_Material/Poisoning_Attack
python initialize_models.py
cd ../..

echo "Initializing Realizable Attack models..."
cd AI_Attacks_Material/Realizable_Attack
python initialize_models.py
cd ../..

echo "Initializing Model Stealing Attack models..."
cd AI_Attacks_Material/Model_Stealing_Attack
python initialize_models.py
cd ../..

# Initialize AI-driven attack data
echo "Initializing AI-Driven Phishing data..."
cd AI_Driven_Attacks/AI_ML_Phishing
python initialize_data.py
cd ../..

echo "Initializing DeepFake Attack data..."
cd AI_Driven_Attacks/Deep_Fake_Attack
python initialize_data.py
cd ../..

echo "All models and data initialized successfully!"
EOL

chmod +x initialize_all_models.sh

# Create a script to start all services
print_info "Creating startup script..."
cat > start_lab.sh << EOL
#!/bin/bash

# Start AI Security Lab services
echo "Starting AI Security Lab services..."

# Activate virtual environment
source venv/bin/activate

# Start Flask backends in background
echo "Starting Flask backends..."
cd AI_Attacks_Material/Backdoor_Attack
python deploy.py > ../../logs/backdoor_flask.log 2>&1 &
BACKDOOR_PID=\$!
cd ../..

cd AI_Attacks_Material/Evasion_Attack
python deploy.py > ../../logs/evasion_flask.log 2>&1 &
EVASION_PID=\$!
cd ../..

# Wait for Flask servers to initialize
echo "Waiting for Flask servers to initialize..."
sleep 5

# Start React development servers in separate terminals
echo "Starting React applications..."
gnome-terminal --tab -- bash -c "cd Lab_Demonstration/1.\ evasion_attack_demo && npm start; bash"
gnome-terminal --tab -- bash -c "cd Lab_Demonstration/2.\ poisoning-attack-demo && PORT=3001 npm start; bash"
gnome-terminal --tab -- bash -c "cd Lab_Demonstration/3.\ backdoor-attack-demo && PORT=3002 npm start; bash"
gnome-terminal --tab -- bash -c "cd Lab_Demonstration/4.\ model-stealing-demo && PORT=3003 npm start; bash"
gnome-terminal --tab -- bash -c "cd Lab_Demonstration/5.\ adversarial_attack && PORT=3004 npm start; bash"

echo "All services started!"
echo "To stop Flask servers, run: kill \$BACKDOOR_PID \$EVASION_PID"
echo "To stop React servers, close their terminal tabs or press Ctrl+C in each tab"
EOL

chmod +x start_lab.sh

# Create a post-installation instructions file
print_header "Creating Post-Installation Instructions"
cat > POST_INSTALL_INSTRUCTIONS.md << EOL
# AI Security Lab - Post-Installation Instructions

## Environment Activation

Always activate the virtual environment before running any Python scripts:

\`\`\`bash
source venv/bin/activate
\`\`\`

## Python and TensorFlow Note

This lab uses Python 3.10 to ensure compatibility with TensorFlow and other libraries.
All required packages, including TensorFlow, should already be installed in the virtual environment.

## Initialize Models

Before using the lab, you need to initialize the models:

\`\`\`bash
./initialize_all_models.sh
\`\`\`

This will create the necessary model files for all modules.

## Starting the Lab

Run the startup script to start all services:

\`\`\`bash
./start_lab.sh
\`\`\`

This will:
1. Start Flask backend servers
2. Start React development servers in separate terminal tabs

## Accessing the Lab

1. Evasion Attack Demo: http://localhost:3000
2. Poisoning Attack Demo: http://localhost:3001
3. Backdoor Attack Demo: http://localhost:3002
4. Model Stealing Demo: http://localhost:3003
5. Adversarial Attack Demo: http://localhost:3004

## Running Individual Modules

### Python Notebooks

To run Jupyter notebooks:

\`\`\`bash
source venv/bin/activate
jupyter notebook
\`\`\`

### Flask Backends

To run a specific Flask backend:

\`\`\`bash
source venv/bin/activate
cd AI_Attacks_Material/Backdoor_Attack  # or any other module directory
python deploy.py
\`\`\`

### React Applications

To run a specific React application:

\`\`\`bash
cd Lab_Demonstration/1.\ evasion_attack_demo  # or any other React app directory
npm start
\`\`\`

## Directory Structure

\`\`\`
├── AI_Attacks_Material
│   ├── Backdoor_Attack          # Backdoor attacks against ML models 
│   ├── Evasion_Attack           # Various evasion attack implementations
│   ├── Model_Stealing_Attack    # Model extraction and stealing techniques
│   ├── Poisoning_Attack         # Training data poisoning attacks
│   └── Realizable_Attack        # Practical real-world adversarial examples
├── AI_Driven_Attacks
│   ├── AI_ML_Phishing           # Educational AI-driven phishing attack demonstration
│   └── Deep_Fake_Attack         # Deepfake creation and detection
├── Lab_Demonstration            # Interactive web-based demonstrations
│   ├── 1. evasion_attack_demo
│   ├── 2. poisoning-attack-demo
│   ├── 3. backdoor-attack-demo