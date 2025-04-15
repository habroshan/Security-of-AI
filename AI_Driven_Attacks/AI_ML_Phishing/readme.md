# AI/ML Phishing Framework

This repository contains an AI/ML-powered framework for demonstrating phishing vulnerability assessment and email generation for security research and awareness training.

## Setup Instructions

### 1. Install Required Libraries and Create Environment

First, make the setup script executable:
```bash
chmod +x setup_script.py
```

Run the setup script to install all dependencies:
```bash
./setup_script.py
```

Activate the environment:
```bash
source venv/bin/activate
```

## Usage Instructions

### Train the ML Models
Train the machine learning models on target data:
```bash
python ai_ml_phishing_framework.py --train-models
```

### Analyze Target Vulnerability
Analyze a specific target's vulnerability:
```bash
python ai_ml_phishing_framework.py --analyze-target --target-index 0
```

### Generate Phishing Email
Generate a simulated phishing email using all the AI/ML components:
```bash
python ai_ml_phishing_framework.py --generate-email --target-index 0
```

## Important Note

This tool is intended for security research, awareness training, and authorized penetration testing only. Do not use this tool for malicious purposes or against targets without proper authorization.
