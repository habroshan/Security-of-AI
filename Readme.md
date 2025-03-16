# AI Security and Privacy Lab Environment

This repository contains a comprehensive lab environment and educational materials for exploring AI/ML security and privacy concepts, based on the CyBOK (Cyber Security Body of Knowledge) Security and Privacy of AI Knowledge Guide.

## Overview

This project provides hands-on tools, demonstrations, and learning materials to understand security vulnerabilities and privacy concerns in artificial intelligence and machine learning systems. It includes practical implementations of various attack vectors and defense strategies, allowing users to experiment with real-world AI security scenarios.

## Project Structure

The repository is organized into separate modules, each focusing on different attack vectors:

```
├── AI_ML_Phishing               # Educational AI-driven phishing attack demonstration
├── Backdoor_Attack              # Backdoor attacks against ML models
├── Deep_Fake_Attack             # Deepfake creation and detection
├── Evasion_Attack               # Various evasion attack implementations
├── Lab                          # Interactive web-based demonstrations
├── Model_Stealing_Attack        # Model extraction and stealing techniques
├── Poisoning_Attack             # Training data poisoning attacks
├── Realizable_Attack            # Practical real-world adversarial examples
└── Requirements.txt             # Project dependencies
```

## Attack Types Implemented

### 1. Evasion Attacks (`Evasion_Attack/`)
Techniques for creating adversarial examples at test time:
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Carlini & Wagner (C&W) Attack
- Fast Minimum-Norm (FMN) Attack

### 2. Poisoning Attacks (`Poisoning_Attack/`)
Methods for manipulating training data:
- Adversarial Poisoning
- False Data Injection Attack

### 3. Backdoor Attacks (`Backdoor_Attack/`)
Techniques for embedding hidden behaviors in ML models:
- Trigger pattern injection
- Target modification attacks

### 4. Model Stealing (`Model_Stealing_Attack/`)
Approaches for extracting model information:
- Model parameter extraction
- Model functionality stealing
- Black-box model extraction

### 5. Realizable Attacks (`Realizable_Attack/`)
Practical adversarial examples with real-world constraints:
- Geometric transformations
- Feature-space to problem-space mapping

### 6. AI-driven Phishing (`AI_ML_Phishing/`)
Educational demonstration of how AI could enhance phishing:
- Personalized content generation
- Target analysis frameworks

### 7. Deepfake Attack (`Deep_Fake_Attack/`)
Demonstrates deepfake creation and detection methods:
- Face swapping techniques
- Deepfake detection models

## Interactive Demonstrations (`Lab/`)

The `Lab/` directory contains interactive web-based demonstrations:

1. `evasion_attack_demo/`: Interactive visualization of evasion attacks
2. `poisoning-attack-demo/`: Demonstration of data poisoning
3. `backdoor-attack-demo/`: Visual explanation of backdoor attacks
4. `model-stealing-demo/`: Illustration of model extraction
5. `adversarial_attack/`: Simplified PGD attack demonstration

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- PyTorch (for some notebooks)
- Flask
- Node.js and npm (for web demos)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-security-privacy-lab.git
   cd ai-security-privacy-lab
   ```

2. Install Python & Other Libraries dependencies:
   ```bash
     chmod +x setup.sh
    ./setup.sh
   ```

3. For Lab Demonstrations dependencies:
   ```bash
   ./start_lab.sh
   ```

### Running Attacks

Each attack module contains Jupyter notebooks (.ipynb files) with detailed implementations and explanations. To run a specific attack, navigate to its directory and open the notebook:

```bash
cd Evasion_Attack
jupyter notebook fgsm-attack.ipynb
```

### Deploying Models

Many modules include a `deploy.py` script for deploying trained models as APIs:

```bash
cd Backdoor_Attack
python deploy.py
```

This will start a Flask server that exposes the model through a REST API.

## Educational Materials

Each module contains:
- Jupyter notebooks with detailed explanations and code
- PDF files with theoretical background
- Instructions documents with step-by-step guidance
- Pre-trained models for experimentation

## IMPORTANT: Ethical Usage Disclaimer

**This project is provided strictly for educational purposes.**

The techniques demonstrated in this repository can be harmful if misused. In particular:

- The AI-ML phishing demonstration shows potential attack techniques to understand and defend against them, not to implement them.
- Deepfake technology should only be used ethically with proper consent.
- Adversarial attacks against production ML systems without authorization are illegal.

Always practice responsible disclosure and obtain proper authorization before security testing.

## References

This project implements techniques from various academic papers:

- Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (FGSM)
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
- Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (C&W)
- Pintor et al., "Fast Minimum-Norm Adversarial Attacks Through Adaptive Norm Constraints" (FMN)
- Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain" (Backdoor)
- Tramèr et al., "Stealing Machine Learning Models via Prediction APIs" (Model Stealing)
- Pierazzi et al., "Intriguing Properties of Adversarial ML Attacks in the Problem Space" (Realizable Attacks)

## License



## Acknowledgments

- Lorenzo Cavallaro and Emiliano De Cristofaro for the CyBOK Security and Privacy of AI Knowledge Guide
- The CyBOK project for funding the development of these educational resources
