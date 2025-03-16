#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced ML/AI-Driven Phishing Framework - EDUCATIONAL ONLY
===========================================================

This code demonstrates how real machine learning and AI techniques could be applied
to create sophisticated phishing attacks. It includes actual ML model components,
NLP text generation, target classification, feature extraction, and optimization.

IMPORTANT: This code is provided STRICTLY FOR EDUCATIONAL PURPOSES to demonstrate
security vulnerabilities. The models are simplified for educational understanding.
Using this for actual phishing attempts is illegal and unethical.

For CYBOK (Cyber Security Body of Knowledge) educational materials.
"""

import os
import numpy as np
import pandas as pd
import json
import random
import re
import time
from datetime import datetime, timedelta
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

# Ensure required NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class MLPhishingFramework:
    """
    Educational framework demonstrating ML/AI-driven phishing techniques
    
    This class shows how machine learning and AI could be applied to create
    sophisticated phishing attacks, including:
    
    1. Target Analysis: ML models to analyze and profile potential targets
    2. Content Generation: NLP models to generate convincing phishing content
    3. Personalization: ML to customize attacks based on target characteristics
    4. Optimization: Learning algorithms to improve effectiveness over time
    
    EDUCATIONAL PURPOSE ONLY: This framework is for understanding threats.
    """
    def cluster_targets(self, n_clusters=3):
        """
        Cluster targets into groups with similar characteristics
        
        This helps identify distinct groups of targets that might be
        susceptible to different phishing approaches
        """
        if self.target_features is None:
            print("[!] No target features available for clustering")
            return False
        
        print(f"[+] Clustering targets into {n_clusters} groups")
        
        # Normalize features for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.target_features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to targets
        self.targets['cluster'] = clusters
        
        # Analyze clusters
        print("\nCluster Analysis:")
        for cluster_id in range(n_clusters):
            cluster_size = np.sum(clusters == cluster_id)
            cluster_percent = cluster_size / len(clusters) * 100
            
            print(f"\nCluster {cluster_id}: {cluster_size} targets ({cluster_percent:.1f}%)")
            
            # Get targets in this cluster
            cluster_targets = self.targets[self.targets['cluster'] == cluster_id]
            
            # Show cluster characteristics
            print("  Departments:", ", ".join(cluster_targets['department'].value_counts().index[:3]))
            print(f"  Avg Security Awareness: {cluster_targets['security_awareness'].mean():.2f}")
            print(f"  Avg Susceptibility: {cluster_targets['susceptibility_score'].mean():.2f}")
            
            # Show a sample target from the cluster
            sample_idx = cluster_targets.index[0]
            sample = cluster_targets.iloc[0]
            print(f"  Sample Target: {sample['name']}, {sample['position']} at {sample['company']}")
        
        # Save the model
        self.models['target_clustering'] = kmeans
        
        return True
    

    def initialize_text_generator(self, pretrained_model='gpt2'):
        """
        Initialize a text generation model for creating personalized content
        
        For educational purposes, this uses a pre-trained model with simple fine-tuning.
        A real attack might use a custom-trained model on specific corporate communications.
        
        In a real-world attack scenario, the adversary might:
        1. Scrape company communications, emails, and documentation
        2. Fine-tune a language model on this corpus to mimic corporate style
        3. Use reinforcement learning to optimize for user engagement
        """
        print(f"[+] Initializing text generation model ({pretrained_model})")
        
        try:
            # Try to load transformers if available
            from transformers import GPT2Tokenizer, GPT2LMHeadModel

            # Load pre-trained GPT-2 model and tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
            model = GPT2LMHeadModel.from_pretrained(pretrained_model)
            
            # Add required special tokens
            special_tokens = {
                'additional_special_tokens': [
                    '{{name}}', '{{company}}', '{{position}}', 
                    '{{email}}', '{{reset_link}}'
                ]
            }
            tokenizer.add_special_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer))
            
            # Store the model and tokenizer
            self.text_generator = model
            self.text_tokenizer = tokenizer
            
            print("[+] Text generation model initialized successfully")
            return True
            
        except Exception as e:
            print(f"[!] Error initializing text generation model: {e}")
            
            # Fallback to template-based generation for educational demo
            print("[+] Using template-based generation instead")
            self.text_generator = None
            self.text_tokenizer = None
            
            print("[+] Simple text generation initialized (template-based)")
            return False
    
    def train_personalization_model(self):
        """
        Train a model to personalize phishing content for each target
        
        This model predicts which elements to include in a phishing email
        based on the target's profile
        """
        if self.target_features is None:
            print("[!] No target features available for training personalization model")
            return False
        
        print("[+] Training email personalization model")
        
        # Generate synthetic personalization data
        # In a real scenario, this would come from historical campaign data
        
        # Features that might be effective for different targets
        personalization_features = [
            'include_position_reference',  # Reference their job role
            'include_technical_details',   # Include technical details
            'emphasize_urgency',           # Emphasize time sensitivity
            'use_formal_language',         # Use formal vs casual language
            'include_company_branding',    # Include company branding/references
            'reference_colleagues',        # Reference colleagues or team
            'mention_specific_systems',    # Mention specific IT systems
            'include_financial_details'    # Include financial/budget references
        ]
        
        # Generate synthetic effectiveness data based on target profiles
        X = self.target_features.values
        y = np.zeros((len(X), len(personalization_features)))
        
        for i, row in enumerate(self.targets.iterrows()):
            target = row[1]
            dept = target['department']
            position = str(target['position']).lower()
            tech_savvy = target['tech_savviness']
            
            # Position reference effectiveness
            y[i, 0] = np.random.normal(0.8, 0.1)  # Generally effective
            
            # Technical details effectiveness
            if dept == 'IT' or tech_savvy > 7.0:
                y[i, 1] = np.random.normal(0.9, 0.1)  # More effective for technical roles
            else:
                y[i, 1] = np.random.normal(0.4, 0.1)  # Less effective for non-technical roles
            
            # Urgency effectiveness
            if dept == 'Executive' or 'manager' in position or 'director' in position:
                y[i, 2] = np.random.normal(0.9, 0.1)  # Executives respond to urgency
            else:
                y[i, 2] = np.random.normal(0.7, 0.1)  # Still fairly effective for others
            
            # Formal language effectiveness
            if dept == 'Executive' or dept == 'Finance':
                y[i, 3] = np.random.normal(0.8, 0.1)  # More formal for executives/finance
            else:
                y[i, 3] = np.random.normal(0.6, 0.1)  # Less formal for others
            
            # Company branding effectiveness
            y[i, 4] = np.random.normal(0.85, 0.1)  # Generally effective
            
            # Colleague references effectiveness
            if dept == 'HR' or 'manager' in position:
                y[i, 5] = np.random.normal(0.85, 0.1)  # More effective for people-focused roles
            else:
                y[i, 5] = np.random.normal(0.6, 0.1)
            
            # System reference effectiveness
            if dept == 'IT' or tech_savvy > 6.0:
                y[i, 6] = np.random.normal(0.9, 0.1)  # More effective for IT
            else:
                y[i, 6] = np.random.normal(0.5, 0.1)
            
            # Financial details effectiveness
            if dept == 'Finance' or dept == 'Executive' or 'account' in position:
                y[i, 7] = np.random.normal(0.9, 0.1)  # More effective for finance-related roles
            else:
                y[i, 7] = np.random.normal(0.5, 0.1)
            
            # Clip to valid range [0, 1]
            y[i] = np.clip(y[i], 0, 1)
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            # Create a neural network for personalization
            # For simplicity, we'll use a simpler model that doesn't require tensorflow
            from sklearn.ensemble import RandomForestRegressor
            
            # Train a model for each personalization feature
            personalization_models = []
            for i, feature_name in enumerate(personalization_features):
                print(f"[+] Training model for: {feature_name}")
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train[:, i])
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = np.mean((y_test[:, i] - y_pred) ** 2)
                print(f"    - MSE: {mse:.4f}")
                
                personalization_models.append(model)
            
            # Save the models
            self.models['personalization_models'] = personalization_models
            self.models['personalization_features'] = personalization_features
            
            # For educational purposes, show how different target characteristics 
            # influence personalization strategies
            print("\nPersonalization Strategy Analysis:")
            print("(How target characteristics influence email customization)")
            
            # Show feature importance for each personalization strategy
            for i, feature_name in enumerate(personalization_features):
                model = personalization_models[i]
                
                if hasattr(model, 'feature_importances_'):
                    print(f"\n  {feature_name}:")
                    feature_importances = model.feature_importances_
                    feature_names = self.target_features.columns
                    
                    # Get top features
                    sorted_idx = np.argsort(feature_importances)[::-1][:3]
                    for idx in sorted_idx:
                        print(f"    - {feature_names[idx]}: {feature_importances[idx]:.4f}")
            
            print(f"\n[+] Personalization models trained successfully")
            return True
            
        except Exception as e:
            print(f"[!] Error training personalization model: {str(e)}")
            print("[+] Using simplified personalization logic instead")
            
            # Create a simple dictionary-based model as fallback
            self.models['personalization_features'] = personalization_features
            
            # Simple rules-based model as fallback
            self.models['personalization_rules'] = {
                'IT': ['include_technical_details', 'mention_specific_systems', 'include_position_reference'],
                'Finance': ['include_financial_details', 'use_formal_language', 'emphasize_urgency'],
                'Executive': ['use_formal_language', 'emphasize_urgency', 'include_company_branding'],
                'HR': ['reference_colleagues', 'include_position_reference', 'include_company_branding'],
                'Marketing': ['include_company_branding', 'reference_colleagues', 'emphasize_urgency']
            }
            
            print("[+] Simplified personalization rules created")
            return False

    
    def __init__(self, output_dir="output", seed=42):
        """Initialize the ML phishing framework"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        
        # Create model directories
        self.models_dir = os.path.join(output_dir, "models")
        self.data_dir = os.path.join(output_dir, "data")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize components
        self.models = {}
        self.templates = []
        self.targets = None
        self.target_features = None
        self.text_tokenizer = None
        self.text_generator = None
        self.stop_words = set(stopwords.words('english'))
        
        print("[+] Advanced ML Phishing Framework initialized")
    
    def load_training_data(self, emails_path=None, targets_path=None):
        """
        Load or generate training data for the ML models
        
        Args:
            emails_path: Path to dataset of emails for training text generation
            targets_path: Path to dataset of target profiles for classifier training
        """
        # Load or generate email templates
        if emails_path and os.path.exists(emails_path):
            print(f"[+] Loading email templates from {emails_path}")
            # In a real implementation, this would load actual email data
            self.templates = self._load_email_templates(emails_path)
        else:
            print("[+] Generating synthetic email templates")
            self.templates = self._generate_synthetic_email_data()
        
        # Load or generate target profiles
        if targets_path and os.path.exists(targets_path):
            print(f"[+] Loading target profiles from {targets_path}")
            try:
                self.targets = pd.read_csv(targets_path)
            except Exception as e:
                print(f"[!] Error loading target data: {e}")
                self.targets = self._generate_synthetic_target_data()
        else:
            print("[+] Generating synthetic target profiles")
            self.targets = self._generate_synthetic_target_data()
        
        # Extract features from target profiles
        self._extract_target_features()
        
        # Save synthetic data for reference
        self._save_synthetic_data()
    
    def _load_email_templates(self, path):
        """Load email templates from a file"""
        templates = []
        
        # In a real implementation, this would parse actual email data
        # For educational purposes, we'll use predefined templates
        with open(path, 'r', encoding='utf-8') as f:
            try:
                templates = json.load(f)
            except json.JSONDecodeError:
                # If not JSON, try reading as CSV
                try:
                    email_df = pd.read_csv(path)
                    for _, row in email_df.iterrows():
                        if 'subject' in row and 'content' in row:
                            templates.append({
                                'subject': row['subject'],
                                'content': row['content'],
                                'type': row.get('type', 'general')
                            })
                except Exception:
                    print("[!] Could not parse email data, using synthetic data instead")
                    templates = self._generate_synthetic_email_data()
        
        return templates
    
    def _generate_synthetic_email_data(self):
        """Generate synthetic email data for training the models"""
        templates = []
        
        # Security alert templates
        security_subjects = [
            "Security Alert: Unusual Account Activity Detected",
            "Action Required: Security Notification for Your Account",
            "URGENT: Security Alert for Your {{company}} Account",
            "Security Warning: Immediate Action Required",
            "Your {{company}} Account: Security Verification Needed"
        ]
        
        security_content = [
            """
            <p>Dear {{name}},</p>
            <p>Our security system has detected unusual login activity on your account. To ensure your account remains secure, please verify your identity by clicking the link below:</p>
            <p><a href="{{reset_link}}">Verify Account Now</a></p>
            <p>If you did not attempt to access your account, please reset your password immediately.</p>
            <p>Thank you,<br>{{company}} Security Team</p>
            """,
            """
            <p>Dear {{name}},</p>
            <p>We have detected multiple failed login attempts to your {{company}} account. For security purposes, we need to verify your identity.</p>
            <p>Please click <a href="{{reset_link}}">here</a> to confirm your account details and restore full access to your account.</p>
            <p>This is an automated message. Please do not reply.</p>
            <p>Regards,<br>Account Security</p>
            """,
            """
            <p>Dear {{name}},</p>
            <p>As part of our ongoing security measures, we regularly verify account activity. We've noticed some unusual patterns in your recent account usage.</p>
            <p>To continue using your account without interruption, please <a href="{{reset_link}}">confirm your account details</a> within 24 hours.</p>
            <p>If you have any questions, please contact support.</p>
            <p>{{company}} Security</p>
            """
        ]
        
        # Document sharing templates
        document_subjects = [
            "{{sender_name}} shared a document with you: {{document_name}}",
            "New shared document: {{document_name}}",
            "Document shared with you from {{sender_name}}",
            "Please review: {{document_name}}",
            "{{company}} document shared: {{document_name}}"
        ]
        
        document_content = [
            """
            <p>Hi {{name}},</p>
            <p>{{sender_name}} ({{sender_email}}) has shared a document with you: <strong>{{document_name}}</strong></p>
            <p>You can access the document by clicking <a href="{{reset_link}}">here</a>.</p>
            <p>The document will be available for 7 days.</p>
            <p>Regards,<br>{{company}} Document Sharing</p>
            """,
            """
            <p>Hello {{name}},</p>
            <p>A new document has been shared with you by {{sender_name}}.</p>
            <p><strong>Document:</strong> {{document_name}}<br>
            <strong>Shared on:</strong> {{shared_date}}<br>
            <strong>Message:</strong> "Please review this as soon as possible."</p>
            <p><a href="{{reset_link}}">View Document</a></p>
            <p>Thank you,<br>{{company}} Team</p>
            """
        ]
        
        # Invoice/payment templates
        invoice_subjects = [
            "Invoice #{{invoice_number}} from {{company}} is due",
            "Your invoice #{{invoice_number}} - Payment required",
            "REMINDER: Outstanding payment for invoice #{{invoice_number}}",
            "Invoice payment notification: #{{invoice_number}}",
            "Action required: Invoice #{{invoice_number}} payment"
        ]
        
        invoice_content = [
            """
            <p>Dear {{name}},</p>
            <p>This is a reminder that invoice #{{invoice_number}} for services provided to {{company}} is due for payment.</p>
            <p><strong>Amount due:</strong> ${{invoice_amount}}<br>
            <strong>Due date:</strong> {{due_date}}</p>
            <p>To view and pay your invoice, please <a href="{{reset_link}}">click here</a>.</p>
            <p>Thank you for your business.</p>
            <p>Accounts Receivable<br>{{company}}</p>
            """,
            """
            <p>Dear {{name}},</p>
            <p>Your payment of ${{invoice_amount}} for invoice #{{invoice_number}} is now overdue.</p>
            <p>To avoid late fees, please process this payment immediately by clicking <a href="{{reset_link}}">here</a>.</p>
            <p>If you have already made this payment, please disregard this notice.</p>
            <p>Regards,<br>Billing Department</p>
            """
        ]
        
        # Create templates
        for subject in security_subjects:
            for content in security_content:
                templates.append({
                    'subject': subject,
                    'content': content,
                    'type': 'security'
                })
        
        for subject in document_subjects:
            for content in document_content:
                templates.append({
                    'subject': subject,
                    'content': content,
                    'type': 'document'
                })
        
        for subject in invoice_subjects:
            for content in invoice_content:
                templates.append({
                    'subject': subject,
                    'content': content,
                    'type': 'invoice'
                })
        
        print(f"[+] Generated {len(templates)} synthetic email templates")
        return templates
    
    def _generate_synthetic_target_data(self, num_samples=100):
        """Generate synthetic target data for training the models"""
        # Random data generation helpers
        first_names = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "Robert", "Lisa", 
                      "William", "Jennifer", "James", "Maria", "Thomas", "Linda", "Daniel", "Patricia"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia",
                      "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson"]
        companies = ["Acme Inc", "Tech Solutions", "Global Industries", "Summit Corp", "Pinnacle Systems",
                    "Infinite Software", "Apex Consulting", "Venture Partners", "Prime Technologies", "Elite Services"]
        domains = ["acme.com", "techsolutions.com", "globalind.com", "summitcorp.com", "pinnacle-sys.com",
                   "infinite-soft.com", "apexconsult.com", "venture-partners.com", "primetech.com", "eliteserv.com"]
        
        it_positions = ["IT Manager", "Security Analyst", "Network Administrator", "System Engineer", 
                       "IT Director", "Security Officer", "Database Administrator", "Cloud Architect"]
        
        exec_positions = ["CEO", "CFO", "CTO", "COO", "President", "Vice President", "Director", 
                         "Executive Assistant", "Chairman", "Board Member"]
        
        finance_positions = ["Accountant", "Financial Analyst", "Controller", "Finance Manager", 
                            "Accounts Payable", "Accounts Receivable", "Payroll Specialist", "Bookkeeper"]
        
        hr_positions = ["HR Manager", "Recruiter", "HR Specialist", "Talent Acquisition", "Training Coordinator", 
                       "HR Director", "Onboarding Specialist", "Benefits Administrator"]
        
        marketing_positions = ["Marketing Manager", "Social Media Specialist", "Marketing Director", 
                              "Content Writer", "SEO Specialist", "Brand Manager", "Marketing Analyst"]
        
        # Generate synthetic profiles
        data = {
            'name': [],
            'email': [],
            'position': [],
            'department': [],
            'company': [],
            'company_domain': [],
            'interests': [],
            'social_media': [],
            'age_group': [],
            'tech_savviness': [],
            'security_awareness': [],
            'susceptibility_score': []
        }
        
        departments = {
            'IT': it_positions,
            'Executive': exec_positions,
            'Finance': finance_positions,
            'HR': hr_positions,
            'Marketing': marketing_positions
        }
        
        interests_options = [
            "technology, cybersecurity, cloud computing",
            "finance, investments, accounting",
            "marketing, social media, advertising",
            "human resources, recruitment, training",
            "business strategy, leadership, management",
            "data science, analytics, big data",
            "sales, customer relations, negotiation",
            "project management, agile, scrum",
            "design, UX/UI, creative",
            "operations, logistics, supply chain"
        ]
        
        social_media_options = [
            "LinkedIn, Twitter",
            "LinkedIn, Facebook",
            "Twitter, Instagram",
            "LinkedIn, Twitter, Facebook",
            "LinkedIn only",
            "Facebook only",
            "No social media",
            "LinkedIn, Twitter, Instagram",
            "Twitter only",
            "LinkedIn, Facebook, Instagram"
        ]
        
        for i in range(num_samples):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            full_name = f"{first_name} {last_name}"
            
            company_idx = random.randint(0, len(companies)-1)
            company = companies[company_idx]
            domain = domains[company_idx]
            
            department = random.choice(list(departments.keys()))
            position = random.choice(departments[department])
            
            email = f"{first_name.lower()}.{last_name.lower()}@{domain}"
            
            # Assign interests, social media presence, and demographics
            interests = random.choice(interests_options)
            social_media = random.choice(social_media_options)
            age_group = random.choice(['18-30', '31-45', '46-60', '61+'])
            
            # Assign technical savviness and security awareness scores
            if department == 'IT':
                tech_savviness = random.uniform(7.0, 10.0)
                security_awareness = random.uniform(6.0, 10.0)
            elif department == 'Executive':
                tech_savviness = random.uniform(4.0, 8.0)
                security_awareness = random.uniform(5.0, 9.0)
            else:
                tech_savviness = random.uniform(3.0, 8.0)
                security_awareness = random.uniform(2.0, 7.0)
            
            # Calculate synthetic susceptibility score (inversely related to security awareness)
            susceptibility_base = 10 - security_awareness
            susceptibility_noise = random.uniform(-1.0, 1.0)
            susceptibility_score = max(1.0, min(10.0, susceptibility_base + susceptibility_noise))
            
            # Add to data dictionary
            data['name'].append(full_name)
            data['email'].append(email)
            data['position'].append(position)
            data['department'].append(department)
            data['company'].append(company)
            data['company_domain'].append(domain)
            data['interests'].append(interests)
            data['social_media'].append(social_media)
            data['age_group'].append(age_group)
            data['tech_savviness'].append(round(tech_savviness, 2))
            data['security_awareness'].append(round(security_awareness, 2))
            data['susceptibility_score'].append(round(susceptibility_score, 2))
        
        # Create DataFrame
        df = pd.DataFrame(data)
        print(f"[+] Generated {len(df)} synthetic target profiles")
        return df
    
    def _extract_target_features(self):
        """Extract features from target profiles for ML models"""
        if self.targets is None:
            print("[!] No target data available for feature extraction")
            return
        
        print("[+] Extracting features from target profiles")
        
        # Create feature dictionary
        features = {}
        
        # Process categorical features using one-hot encoding
        categorical_features = ['department', 'age_group']
        for feature in categorical_features:
            if feature in self.targets.columns:
                dummies = pd.get_dummies(self.targets[feature], prefix=feature)
                for col in dummies.columns:
                    features[col] = dummies[col].values
        
        # Process numerical features
        numerical_features = ['tech_savviness', 'security_awareness', 'susceptibility_score']
        for feature in numerical_features:
            if feature in self.targets.columns:
                features[feature] = self.targets[feature].values
        
        # Process text features using TF-IDF
        text_features = ['position', 'interests']
        for feature in text_features:
            if feature in self.targets.columns:
                # Convert to string in case there are any non-string values
                text_data = self.targets[feature].astype(str).tolist()
                
                # Create and fit TF-IDF vectorizer
                vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(text_data)
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Add TF-IDF features to feature dictionary
                for i, name in enumerate(feature_names):
                    feature_key = f"{feature}_tfidf_{name}"
                    features[feature_key] = tfidf_matrix[:, i].toarray().flatten()
        
        # Convert features to DataFrame
        self.target_features = pd.DataFrame(features)
        print(f"[+] Extracted {len(self.target_features.columns)} features from target profiles")
    
    def _save_synthetic_data(self):
        """Save synthetic data for reference"""
        # Save target profiles
        if self.targets is not None:
            targets_path = os.path.join(self.data_dir, "synthetic_targets.csv")
            self.targets.to_csv(targets_path, index=False)
            print(f"[+] Saved synthetic target profiles to {targets_path}")
        
        # Save email templates
        if self.templates:
            templates_path = os.path.join(self.data_dir, "synthetic_templates.json")
            with open(templates_path, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=2)
            print(f"[+] Saved synthetic email templates to {templates_path}")
            
        # Save a sample of targets with features for educational analysis
        if self.targets is not None and self.target_features is not None:
            # Take a small sample
            sample_size = min(5, len(self.targets))
            sample_indices = np.random.choice(len(self.targets), sample_size, replace=False)
            
            # Create a sample file with targets and their features
            sample_path = os.path.join(self.data_dir, "target_analysis_sample.txt")
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write("=== TARGET ANALYSIS SAMPLE ===\n")
                f.write("This file shows how ML algorithms extract and analyze features from target profiles\n\n")
                
                for idx in sample_indices:
                    target = self.targets.iloc[idx]
                    features = self.target_features.iloc[idx]
                    
                    f.write(f"TARGET: {target['name']}, {target['position']} at {target['company']}\n")
                    f.write(f"Email: {target['email']}\n")
                    f.write(f"Department: {target['department']}\n")
                    f.write(f"Interests: {target['interests']}\n")
                    f.write(f"Security Awareness: {target['security_awareness']}\n")
                    f.write("\nEXTRACTED FEATURES:\n")
                    
                    # Show most significant features
                    for feature, value in features.items():
                        if abs(value) > 0.1:  # Only show more significant features
                            f.write(f"  {feature}: {value:.4f}\n")
                    
                    f.write("\n" + "-"*50 + "\n\n")
            
            print(f"[+] Saved target analysis sample to {sample_path}")
    
    def train_target_classifier(self):
        """
        Train a classifier to predict the best template type for each target
        
        This model analyzes target profiles and predicts which phishing template
        category would be most effective (security, document, invoice, etc.)
        """
        if self.target_features is None or self.targets is None:
            print("[!] No target features available for training classifier")
            return False
        
        print("[+] Training target classifier model")
        
        # Create synthetic effectiveness labels for training
        # In a real scenario, this would come from historical data
        template_types = ['security', 'document', 'invoice']
        
        # Create synthetic labels based on department and position
        y = []
        for _, row in self.targets.iterrows():
            dept = row['department']
            position = str(row['position']).lower()
            
            # Assign the most effective template type based on department/position
            if dept == 'IT' or 'security' in position or 'admin' in position:
                most_effective = 'security'
            elif dept == 'Finance' or 'account' in position or 'finance' in position:
                most_effective = 'invoice'
            elif dept == 'Executive' or 'director' in position or 'manager' in position:
                most_effective = random.choice(['document', 'invoice'])
            else:
                most_effective = 'document'
            
            y.append(most_effective)
        
        # Convert labels to array
        y = np.array(y)
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            self.target_features, y, test_size=0.2, random_state=42
        )
        
        # Train a Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Evaluate the classifier
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[+] Target classifier trained with accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        self.models['target_classifier'] = classifier
        
        # For educational purposes, also show feature importances
        feature_importances = classifier.feature_importances_
        feature_names = self.target_features.columns
        
        print("\nTop 10 Most Important Features:")
        sorted_idx = np.argsort(feature_importances)[::-1][:10]
        for idx in sorted_idx:
            print(f"  {feature_names[idx]}: {feature_importances[idx]:.4f}")
        
        return True
    
    def generate_phishing_email(self, target_idx=0):
        """
        Generate a highly personalized phishing email for a specific target
        using the trained ML models
        
        Args:
            target_idx: Index of the target in the targets DataFrame
            
        Returns:
            Dictionary containing the generated email and metadata
        """
        if self.targets is None or target_idx >= len(self.targets):
            print(f"[!] Invalid target index: {target_idx}")
            return None
        
        # Get target information
        target = self.targets.iloc[target_idx].to_dict()
        target_features = self.target_features.iloc[target_idx].values.reshape(1, -1)
        
        print(f"[+] Generating phishing email for: {target['name']} ({target['position']} at {target['company']})")
        
        # Step 1: Use the classifier to determine the most effective template type
        if 'target_classifier' in self.models:
            template_type = self.models['target_classifier'].predict(target_features)[0]
            confidence = np.max(self.models['target_classifier'].predict_proba(target_features))
            print(f"[+] Selected template type: {template_type} (confidence: {confidence:.4f})")
        else:
            # Fallback if no classifier is trained
            template_type = random.choice(['security', 'document', 'invoice'])
            print(f"[+] Selected template type: {template_type} (random selection)")
        
        # Step 2: Select a template of the chosen type
        matching_templates = [t for t in self.templates if t.get('type') == template_type]
        if not matching_templates:
            matching_templates = self.templates  # Fallback to any template
        
        template = random.choice(matching_templates)
        
        # Step 3: Use the personalization model to determine customization strategies
        personalization_strategies = {}
        if 'personalization_model' in self.models and 'personalization_features' in self.models:
            # Predict personalization strategy scores
            strategy_scores = self.models['personalization_model'].predict(target_features)[0]
            
            # Get feature names
            feature_names = self.models['personalization_features']
            
            # Create dictionary of strategies and their scores
            for feature, score in zip(feature_names, strategy_scores):
                personalization_strategies[feature] = score
                
            print("[+] Top personalization strategies:")
            for feature, score in sorted(personalization_strategies.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  - {feature}: {score:.4f}")
        else:
            # Default personalization strategies
            default_strategies = {
                'include_position_reference': 0.8,
                'include_company_branding': 0.8,
                'emphasize_urgency': 0.7
            }
            personalization_strategies = default_strategies
        
        # Step 4: Generate personalized content
        email = self._generate_email_content(target, template, personalization_strategies)
        
        # Step 5: Create and insert phishing link
        email['content'] = self._insert_phishing_links(email['content'], target, template_type)
        
        # Save the email for educational analysis
        self._save_phishing_email(email)
        
        return email
    
    def _generate_email_content(self, target, template, personalization_strategies):
        """Generate personalized email content using AI"""
        # Start with template content
        subject = template['subject']
        content = template['content']
        
        # Basic replacements
        replacements = {
            '{{name}}': target['name'].split()[0],  # First name
            '{{full_name}}': target['name'],
            '{{company}}': target['company'],
            '{{position}}': target['position'],
            '{{email}}': target['email'],
            '{{company_domain}}': target['email'].split('@')[1] if '@' in target['email'] else 'example.com',
            '{{current_year}}': str(datetime.now().year)
        }
        
        # Add template-specific replacements
        if template['type'] == 'security':
            # Security alert details
            replacements.update({
                '{{random_city}}': random.choice(['London', 'Moscow', 'Beijing', 'Lagos', 'Sao Paulo']),
                '{{random_country}}': random.choice(['Russia', 'China', 'Nigeria', 'Brazil', 'Ukraine']),
                '{{access_time}}': (datetime.now() - timedelta(hours=random.randint(2, 8))).strftime('%Y-%m-%d %H:%M:%S'),
                '{{random_device}}': random.choice(['Windows NT 10.0', 'Unknown Device', 'Linux x86_64', 'Android 10.0'])
            })
        
        elif template['type'] == 'document':
            # Document details - personalized based on department
            dept = target['department']
            
            # Select document name based on department
            if dept == 'IT':
                doc_name = random.choice(['Security Policy.pdf', 'Network Diagram.pdf', 'System Requirements.xlsx'])
            elif dept == 'Finance':
                doc_name = random.choice(['Q3 Budget.xlsx', 'Financial Report.pdf', 'Invoice Records.xlsx'])
            elif dept == 'HR':
                doc_name = random.choice(['Employee Handbook.pdf', 'Benefits Update.docx', 'Training Materials.pptx'])
            elif dept == 'Executive':
                doc_name = random.choice(['Strategic Plan.pdf', 'Board Presentation.pptx', 'Annual Report.pdf'])
            elif dept == 'Marketing':
                doc_name = random.choice(['Campaign Results.xlsx', 'Marketing Plan.pptx', 'Brand Guidelines.pdf'])
            else:
                doc_name = random.choice(['Important Document.pdf', 'Team Update.docx', 'Project Plan.xlsx'])
            
            # Create a sender that would be relevant to the target
            sender_first_name = random.choice(['Alex', 'Sam', 'Taylor', 'Jordan', 'Morgan', 'Casey'])
            sender_last_name = random.choice(['Johnson', 'Smith', 'Williams', 'Brown', 'Davis', 'Miller'])
            sender_name = f"{sender_first_name} {sender_last_name}"
            
            # Create a position for the sender based on the target's department
            if dept == 'IT':
                sender_position = random.choice(['IT Director', 'Security Officer', 'System Administrator'])
            elif dept == 'Finance':
                sender_position = random.choice(['Finance Director', 'Controller', 'VP Finance'])
            elif dept == 'HR':
                sender_position = random.choice(['HR Director', 'Talent Manager', 'Benefits Coordinator'])
            elif dept == 'Executive':
                sender_position = random.choice(['Executive Assistant', 'Board Member', 'CEO'])
            elif dept == 'Marketing':
                sender_position = random.choice(['Marketing Director', 'Brand Manager', 'VP Marketing'])
            else:
                sender_position = random.choice(['Department Head', 'Team Lead', 'Project Manager'])
            
            # Create a sender that would interest the target
            domain = target['email'].split('@')[1] if '@' in target['email'] else 'example.com'
            sender_email = f"{sender_first_name.lower()}.{sender_last_name.lower()}@{domain}"
            
            replacements.update({
                '{{document_name}}': doc_name,
                '{{sender_name}}': sender_name,
                '{{sender_position}}': sender_position,
                '{{sender_email}}': sender_email,
                '{{shared_date}}': (datetime.now() - timedelta(days=random.randint(0, 3))).strftime('%b %d, %Y'),
                '{{document_size}}': f"{random.randint(1, 10)}.{random.randint(1, 9)} MB",
                '{{document_type}}': doc_name.split('.')[-1].upper(),
                '{{custom_message}}': self._generate_custom_message(target, sender_position)
            })
        
        elif template['type'] == 'invoice':
            # Invoice details
            replacements.update({
                '{{invoice_number}}': f"INV-{random.randint(10000, 99999)}",
                '{{issue_date}}': (datetime.now() - timedelta(days=random.randint(20, 30))).strftime('%b %d, %Y'),
                '{{due_date}}': (datetime.now() - timedelta(days=random.randint(1, 5))).strftime('%b %d, %Y'),
                '{{invoice_amount}}': f"{random.randint(1000, 9999)}.{random.randint(10, 99)}",
                '{{service_description}}': self._generate_service_description(target)
            })
        
        # Apply personalization strategies
        content = self._apply_personalization_strategies(content, target, personalization_strategies)
        
        # Apply basic replacements to subject and content
        for key, value in replacements.items():
            subject = subject.replace(key, str(value))
            content = content.replace(key, str(value))
        
        # Add any AI-enhanced text generation if available
        if self.text_generator is not None and self.text_tokenizer is not None:
            # Use NLP model to generate an additional personalized paragraph
            try:
                ai_paragraph = self._generate_ai_paragraph(target, template['type'])
                # Insert the AI-generated paragraph before the last paragraph
                content_parts = content.split('</p>')
                if len(content_parts) > 2:
                    content = '</p>'.join(content_parts[:-2]) + '</p>' + ai_paragraph + '</p>' + content_parts[-2] + '</p>' + content_parts[-1]
            except Exception as e:
                print(f"[!] Error generating AI paragraph: {e}")
        
        return {
            'subject': subject,
            'content': content,
            'type': template['type'],
            'replacements': replacements,
            'personalization_strategies': personalization_strategies,
            'metadata': {
                'target': target,
                'template_type': template['type'],
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def _generate_custom_message(self, target, sender_position):
        """Generate a custom message for document sharing"""
        dept = target['department']
        position = target['position']
        
        # Base messages by department
        messages = {
            'IT': [
                "Please review this document regarding our systems security update.",
                "Here's the technical documentation you requested for the project.",
                "I've shared the network configuration you asked about in our meeting."
            ],
            'Finance': [
                "Please review these financial statements before our next meeting.",
                "Here are the budget projections for next quarter as we discussed.",
                "I've shared the invoice records you requested for reconciliation."
            ],
            'HR': [
                "Here are the updated policy documents we discussed in the team meeting.",
                "Please review these training materials before the session next week.",
                "I've shared the employee information you requested for the report."
            ],
            'Executive': [
                "Please review this strategic document before our board meeting.",
                "Here's the confidential report we discussed yesterday.",
                "I've shared the presentation for your approval before the stakeholder meeting."
            ],
            'Marketing': [
                "Here are the campaign results you requested for the quarterly review.",
                "Please review this brand strategy document before our meeting.",
                "I've shared the marketing analytics you needed for your presentation."
            ]
        }
        
        # Default messages for any department
        default_messages = [
            "Please review this document at your earliest convenience.",
            "I've shared an important document for your review.",
            "Here's the document we discussed in our last meeting."
        ]
        
        # Select appropriate message
        department_messages = messages.get(dept, default_messages)
        message = random.choice(department_messages)
        
        # Add urgency if that strategy is selected
        if random.random() < 0.5:  # 50% chance to add urgency
            urgency_phrases = [
                " This is time-sensitive and requires your review by end of day.",
                " Please provide your feedback as soon as possible.",
                " This requires immediate attention."
            ]
            message += random.choice(urgency_phrases)
        
        return message
    
    def _generate_service_description(self, target):
        """Generate a service description relevant to the target"""
        dept = target['department']
        
        # Service descriptions by department
        services = {
            'IT': [
                "IT Support Services",
                "Network Infrastructure Maintenance",
                "Security Software Subscription",
                "Cloud Services",
                "System Monitoring Services"
            ],
            'Finance': [
                "Financial Analysis Services",
                "Accounting Software Subscription",
                "Tax Consulting Services",
                "Financial Reporting Tools",
                "Budgeting Software License"
            ],
            'HR': [
                "Recruitment Services",
                "HR Management Software",
                "Employee Training Program",
                "Benefits Administration Services",
                "Payroll Processing Services"
            ],
            'Executive': [
                "Executive Consulting Services",
                "Strategic Planning Services",
                "Board Meeting Facilitation",
                "Leadership Development Program",
                "Executive Coaching Services"
            ],
            'Marketing': [
                "Marketing Campaign Management",
                "Digital Marketing Services",
                "Brand Development",
                "Marketing Analytics Platform",
                "Social Media Management"
            ]
        }
        
        # Default services for any department
        default_services = [
            "Professional Services",
            "Consulting Services",
            "Software Subscription",
            "Annual Maintenance Contract",
            "Project Management Services"
        ]
        
        # Select appropriate service
        department_services = services.get(dept, default_services)
        return random.choice(department_services)
    
    def _apply_personalization_strategies(self, content, target, strategies):
        """Apply ML-selected personalization strategies to the email content"""
        # Apply each strategy if it scores above threshold
        threshold = 0.6  # Strategies with scores above this will be applied
        
        # Include position reference
        if strategies.get('include_position_reference', 0) > threshold:
            position_references = [
                f"As {target['position']} at {target['company']}, you understand the importance of this matter.",
                f"This is particularly relevant to your role as {target['position']}.",
                f"Given your responsibilities as {target['position']}, we need your input on this."
            ]
            position_ref = random.choice(position_references)
            
            # Insert after first paragraph
            content_parts = content.split('</p>')
            if len(content_parts) > 1:
                # Fix: Join the remaining parts with '</p>' before concatenating
                remaining_parts = '</p>'.join(content_parts[1:])
                content = content_parts[0] + '</p><p>' + position_ref + '</p>' + remaining_parts
            else:
                content += '<p>' + position_ref + '</p>'
        
        # Include technical details
        if strategies.get('include_technical_details', 0) > threshold:
            tech_details = {
                'security': [
                    "<p><strong>Technical Details:</strong> The login attempt used an unrecognized IP address (103.246.xxx.xxx) with browser user-agent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'.</p>",
                    "<p><strong>System Log:</strong> Multiple failed authentication attempts were detected using NTLM protocol on the affected account.</p>"
                ],
                'document': [
                    "<p><strong>Document Details:</strong> SHA-256 hash: 7b43e456fb... | Last modified: Today at 9:41 AM | Created by: Admin</p>",
                    "<p><strong>Compatibility:</strong> This document requires Adobe Reader 11.0+ or Microsoft Office 365 to view properly.</p>"
                ],
                'invoice': [
                    "<p><strong>Payment Details:</strong> ACH/Wire Transfer to Account #XXXX4582 | Reference: INV-{{invoice_number}}</p>",
                    "<p><strong>Tax Information:</strong> Federal Tax ID: 83-XXXXXXX | VAT/GST: Not Applicable</p>"
                ]
            }
            
            # Get appropriate technical details for the template type
            template_type = next((k for k in tech_details.keys() if k in content.lower()), 'security')
            tech_detail = random.choice(tech_details[template_type])
            
            # Insert before last paragraph
            content_parts = content.split('</p>')
            if len(content_parts) > 2:
                # Fix: Join the different parts properly
                beginning_parts = '</p>'.join(content_parts[:-2]) + '</p>'
                end_parts = '</p>'.join(content_parts[-2:])
                content = beginning_parts + tech_detail + end_parts
            else:
                content += tech_detail
        
        # Emphasize urgency
        if strategies.get('emphasize_urgency', 0) > threshold:
            urgency_phrases = {
                'security': [
                    "<p><strong style='color:red;'>URGENT:</strong> Your account access will be suspended in 24 hours if verification is not completed.</p>",
                    "<p><strong>TIME SENSITIVE:</strong> Please complete this security verification within 2 hours of receiving this message.</p>"
                ],
                'document': [
                    "<p><strong>URGENT:</strong> This document requires your review before the 3pm meeting today.</p>",
                    "<p><strong>TIME SENSITIVE:</strong> Please provide your feedback within 24 hours.</p>"
                ],
                'invoice': [
                    "<p><strong style='color:red;'>URGENT:</strong> Payment is required within 24 hours to avoid service interruption.</p>",
                    "<p><strong>FINAL NOTICE:</strong> This invoice is significantly overdue and requires immediate attention.</p>"
                ]
            }
            
            # Get appropriate urgency phrase for the template type
            template_type = next((k for k in urgency_phrases.keys() if k in content.lower()), 'security')
            urgency_phrase = random.choice(urgency_phrases[template_type])
            
            # Insert after first paragraph
            content_parts = content.split('</p>')
            if len(content_parts) > 1:
                # Fix: Join the remaining parts properly
                remaining_parts = '</p>'.join(content_parts[1:])
                content = content_parts[0] + '</p>' + urgency_phrase + remaining_parts
            else:
                content = urgency_phrase + content
        
        # Include company branding
        if strategies.get('include_company_branding', 0) > threshold:
            company = target['company']
            # Add a company logo placeholder and branding
            logo_html = f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="https://via.placeholder.com/200x50?text={company.replace(' ', '+')}" alt="{company} Logo" style="max-width: 200px;">
                <p style="color: #666; font-size: 12px;">{company} - Secure Communications</p>
            </div>
            """
            # Add to the top of the email
            content = logo_html + content
        
        # Reference colleagues (if that strategy is selected)
        if strategies.get('reference_colleagues', 0) > threshold:
            dept = target['department']
            colleague_references = [
                f"<p>Your colleague Sarah from {dept} has already completed this process.</p>",
                f"<p>We've already received verification from other members of the {dept} team.</p>",
                f"<p>John, your department head, asked us to prioritize your account verification.</p>"
            ]
            colleague_ref = random.choice(colleague_references)
            
            # Insert after second paragraph
            content_parts = content.split('</p>')
            if len(content_parts) > 2:
                # Fix: Join the parts properly with '</p>'
                beginning_parts = '</p>'.join(content_parts[:2]) + '</p>'
                end_parts = '</p>'.join(content_parts[2:])
                content = beginning_parts + colleague_ref + end_parts
            else:
                content += colleague_ref
        
        return content
        
    def _generate_ai_paragraph(self, target, template_type):
        """Generate a personalized paragraph using the NLP model"""
        # This would use the text generation model to create personalized content
        # For educational purposes, we'll use pre-written paragraphs
        
        if template_type == 'security':
            paragraphs = [
                f"<p>We take the security of {target['company']} accounts very seriously. Our security team continuously monitors for suspicious activities to protect sensitive company information. As per our updated security protocols implemented last month, all employees must verify their accounts when unusual access patterns are detected.</p>",
                
                f"<p>As part of our enhanced security measures at {target['company']}, we've implemented a new verification system for protecting employee accounts. This additional layer of security helps prevent unauthorized access to our network and safeguards against recent cybersecurity threats targeting our industry.</p>",
                
                f"<p>Our IT security team has noticed several attempted breaches of {target['company']} systems over the past week. As a precautionary measure, we're requiring verification of all accounts in the {target['department']} department, particularly those with access to sensitive information.</p>"
            ]
        
        elif template_type == 'document':
            paragraphs = [
                f"<p>This document contains information relevant to your role as {target['position']}. Your input is particularly valuable given your expertise in this area. The document will be discussed in our upcoming team meeting, so please review it beforehand.</p>",
                
                f"<p>The attached document requires your review before it can be finalized. Given your position as {target['position']} at {target['company']}, your approval is necessary to proceed with the next steps of the project. Your feedback will help ensure we meet all department requirements.</p>",
                
                f"<p>I'm sharing this confidential document with key members of the {target['department']} team. Please note that this information is sensitive and should not be distributed further without authorization from management.</p>"
            ]
        
        elif template_type == 'invoice':
            paragraphs = [
                f"<p>This invoice covers services specifically requested by the {target['department']} department. As per our records, you are the designated approver for these expenses. Please process this payment promptly to maintain uninterrupted service.</p>",
                
                f"<p>Our accounting system indicates that this invoice has been pending for longer than our standard payment terms. To avoid any service interruptions to {target['company']}'s critical systems, please process this payment at your earliest convenience.</p>",
                
                f"<p>This invoice relates to the services we discussed in our previous meeting regarding the {target['department']} department's requirements. As mentioned, these services are essential for maintaining operational efficiency and compliance with industry standards.</p>"
            ]
        
        else:
            paragraphs = [
                f"<p>We value our ongoing relationship with {target['company']} and want to ensure you have all the necessary information. As a valued client, your account has been flagged for priority handling by our team.</p>",
                
                f"<p>Thank you for your continued partnership with our services. We're committed to providing {target['company']} with the highest level of support and appreciate your prompt attention to this matter.</p>",
                
                f"<p>Our records indicate that you are the primary contact for {target['company']} in the {target['department']} department. This information requires your immediate review to ensure uninterrupted service.</p>"
            ]
        
        return random.choice(paragraphs)
    
    def _insert_phishing_links(self, content, target, template_type):
        """
        Insert sophisticated phishing links into the email content
        
        This method demonstrates how ML could generate convincing phishing URLs
        based on target information and template type
        """
        # Company domain for creating convincing lookalike domains
        company_domain = target['email'].split('@')[1] if '@' in target['email'] else 'example.com'
        
        # Create convincing spoofed domain using character substitution and similar domains
        spoofed_domain = self._create_spoofed_domain(company_domain)
        
        # Generate a random token for the URL
        token = self._generate_random_token()
        
        # Create URL path based on template type
        if template_type == 'security':
            paths = [
                "account/verification",
                "login/security-check",
                "identity/confirm",
                "account/password-reset",
                "security/validate"
            ]
        elif template_type == 'document':
            paths = [
                "documents/shared",
                "file/view",
                "sharepoint/access",
                "document/download",
                "files/shared-with-me"
            ]
        elif template_type == 'invoice':
            paths = [
                "invoice/payment",
                "billing/pay-now",
                "finance/invoice-view",
                "accounts/payment-portal",
                "billing/secure-payment"
            ]
        else:
            paths = [
                "portal/login",
                "account/validate",
                "secure/access"
            ]
        
        # Select a path and create the phishing URL
        path = random.choice(paths)
        phishing_url = f"https://{path}.{spoofed_domain}/auth?token={token}&email={target['email']}&src=corp"
        
        # Replace the phishing link placeholder
        content = content.replace("{{reset_link}}", phishing_url)
        
        return content
    
    def _create_spoofed_domain(self, real_domain):
        """
        Create a convincing spoofed domain using ML techniques
        
        A real ML system would analyze thousands of domains to learn
        patterns of convincing spoofing techniques
        """
        # Simple domain spoofing techniques for demonstration
        if '.' in real_domain:
            name, tld = real_domain.split('.', 1)
            
            # Technique 1: Add a subdomain
            if random.random() < 0.3:
                prefixes = ["secure", "account", "login", "auth", "portal", "mail"]
                return f"{random.choice(prefixes)}-{name}.{tld}"
            
            # Technique 2: Similar-looking TLD
            elif random.random() < 0.6:
                similar_tlds = {
                    "com": ["com-secure.net", "com-verify.org", "com-account.co"],
                    "org": ["org-secure.com", "org-id.net", "org-verify.com"],
                    "net": ["net-secure.com", "net-id.org", "net-verify.com"],
                    "edu": ["edu-portal.com", "edu-login.org", "edu-account.net"]
                }
                
                if tld in similar_tlds:
                    return f"{name}.{random.choice(similar_tlds[tld])}"
                return f"{name}-secure.{tld}"
            
            # Technique 3: Character substitution
            else:
                subs = {'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '4', 's': '5', 'b': '8'}
                spoofed_name = ''.join(subs.get(c, c) for c in name)
                if spoofed_name != name:
                    return f"{spoofed_name}.{tld}"
                
                # Fallback
                return f"{name}-verification.{tld}"
        
        return real_domain
    
    def _generate_random_token(self, length=32):
        """Generate a random token for phishing URLs"""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return ''.join(random.choice(chars) for _ in range(length))
    
    def _save_phishing_email(self, email):
        """Save the generated phishing email and analysis for educational purposes"""
        if not email or 'content' not in email or 'subject' not in email:
            print("[!] Invalid email content to save")
            return
        
        # Create timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_name = email['metadata']['target']['name'].replace(' ', '_')
        template_type = email['metadata']['template_type']
        
        # Create base filename
        base_filename = f"{timestamp}_{target_name}_{template_type}"
        
        # Save HTML content
        html_filename = f"{base_filename}.html"
        html_path = os.path.join(self.output_dir, html_filename)
        with open(html_path, 'w', encoding='utf-8') as f:
            # Add an educational disclaimer banner to the HTML
            disclaimer = """
            <div style="background-color: #ffebee; border: 2px solid #f44336; padding: 10px; margin-bottom: 20px; font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f; margin-top: 0;">EDUCATIONAL DEMONSTRATION ONLY</h2>
                <p>This is an <strong>educational example</strong> of an AI-generated phishing email. It demonstrates how 
                machine learning and AI could be used to create convincing targeted phishing attacks.</p>
                <p>Real phishing emails like this are <strong>illegal</strong> and <strong>harmful</strong>.</p>
                <p>Created for CyBOK (Cyber Security Body of Knowledge) educational materials.</p>
            </div>
            """
            
            # Insert disclaimer at the beginning of the content
            content_with_disclaimer = disclaimer + email['content']
            f.write(content_with_disclaimer)
        
        # Save metadata and analysis
        metadata_filename = f"{base_filename}_analysis.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        
        # Enhance metadata with attack analysis
        metadata = email['metadata'].copy()
        metadata['attack_analysis'] = {
            'subject': email['subject'],
            'template_type': template_type,
            'personalization_strategies': email['personalization_strategies'],
            'phishing_url_analysis': {
                'spoofing_technique': 'Character substitution and subdomain manipulation',
                'legitimate_appearance_score': random.uniform(0.7, 0.9),
                'detection_evasion_score': random.uniform(0.6, 0.85)
            },
            'psychological_triggers': [
                {"type": "urgency", "effectiveness": random.uniform(0.7, 0.9)},
                {"type": "authority", "effectiveness": random.uniform(0.75, 0.95)},
                {"type": "fear_of_missing_out", "effectiveness": random.uniform(0.65, 0.85)}
            ],
            'estimated_success_probability': random.uniform(0.3, 0.7),
            'defensive_recommendations': [
                "Implement DMARC, SPF, and DKIM email authentication",
                "Train users to verify suspicious requests through separate channels",
                "Deploy anti-phishing toolbars and email security solutions",
                "Use multi-factor authentication for sensitive accounts"
            ]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a human-readable analysis
        analysis_filename = f"{base_filename}_educational_analysis.txt"
        analysis_path = os.path.join(self.output_dir, analysis_filename)
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AI-DRIVEN PHISHING EMAIL ANALYSIS - EDUCATIONAL PURPOSES ONLY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"TARGET: {metadata['target']['name']}\n")
            f.write(f"POSITION: {metadata['target']['position']}\n")
            f.write(f"COMPANY: {metadata['target']['company']}\n")
            f.write(f"EMAIL: {metadata['target']['email']}\n\n")
            
            f.write(f"TEMPLATE TYPE: {template_type}\n")
            f.write(f"SUBJECT LINE: {email['subject']}\n\n")
            
            f.write("AI-DRIVEN PERSONALIZATION TECHNIQUES:\n")
            for strategy, score in email['personalization_strategies'].items():
                if score > 0.5:  # Only show strategies with significant scores
                    f.write(f"  - {strategy}: {score:.2f}\n")
            
            f.write("\nPSYCHOLOGICAL TRIGGERS EMPLOYED:\n")
            for trigger in metadata['attack_analysis']['psychological_triggers']:
                f.write(f"  - {trigger['type']}: {trigger['effectiveness']:.2f}\n")
            
            f.write("\nPHISHING URL ANALYSIS:\n")
            url_analysis = metadata['attack_analysis']['phishing_url_analysis']
            f.write(f"  - Technique: {url_analysis['spoofing_technique']}\n")
            f.write(f"  - Legitimate Appearance Score: {url_analysis['legitimate_appearance_score']:.2f}\n")
            f.write(f"  - Detection Evasion Score: {url_analysis['detection_evasion_score']:.2f}\n")
            
            f.write(f"\nESTIMATED SUCCESS PROBABILITY: {metadata['attack_analysis']['estimated_success_probability']:.2%}\n\n")
            
            f.write("DEFENSIVE RECOMMENDATIONS:\n")
            for i, rec in enumerate(metadata['attack_analysis']['defensive_recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("EDUCATIONAL NOTE: This analysis demonstrates how AI could be used to create\n")
            f.write("sophisticated phishing attacks. Understanding these techniques helps develop\n")
            f.write("better defenses and training programs to protect against such threats.\n")
            f.write("=" * 80 + "\n")
        
        print(f"[+] Phishing email saved to: {html_path}")
        print(f"[+] Analysis saved to: {analysis_path}")
        
        return {
            'html_path': html_path,
            'metadata_path': metadata_path,
            'analysis_path': analysis_path
        }
    
    def evaluate_target_vulnerability(self, target_idx=0):
        """
        Perform an ML-driven analysis of a target's vulnerability to phishing
        
        This demonstrates how ML could be used to predict susceptibility and
        provide actionable recommendations for security awareness training.
        """
        if self.targets is None or target_idx >= len(self.targets):
            print(f"[!] Invalid target index: {target_idx}")
            return None
        
        # Get target information
        target = self.targets.iloc[target_idx].to_dict()
        
        print(f"[+] Evaluating phishing vulnerability for: {target['name']} ({target['position']} at {target['company']})")
        
        # In a real ML system, this would use a trained model to predict vulnerability
        # For educational purposes, we'll use the features we have
        
        # Base vulnerability factors
        vulnerability_factors = {
            'position_risk': self._calculate_position_risk(target),
            'technical_savviness': 10 - target.get('tech_savviness', 5),  # Invert - lower tech savviness means higher risk
            'security_awareness': 10 - target.get('security_awareness', 5),  # Invert - lower awareness means higher risk
            'department_targeting_frequency': self._get_department_targeting_frequency(target['department']),
            'access_to_sensitive_data': self._estimate_data_access_level(target)
        }
        
        # Calculate overall vulnerability score (weighted average)
        weights = {
            'position_risk': 0.2,
            'technical_savviness': 0.25,
            'security_awareness': 0.3,
            'department_targeting_frequency': 0.15,
            'access_to_sensitive_data': 0.1
        }
        
        vulnerability_score = sum(factor * weights[key] for key, factor in vulnerability_factors.items())
        
        # Normalize to 0-10 scale and add small random variation for educational purposes
        vulnerability_score = max(1, min(10, vulnerability_score + random.uniform(-0.5, 0.5)))
        
        # Generate most effective attack vectors based on vulnerability factors
        attack_vectors = self._identify_effective_attack_vectors(target, vulnerability_factors)
        
        # Generate personalized security recommendations
        security_recommendations = self._generate_security_recommendations(target, vulnerability_factors)
        
        # Create comprehensive vulnerability report
        vulnerability_report = {
            'target': target,
            'vulnerability_score': vulnerability_score,
            'vulnerability_factors': vulnerability_factors,
            'effective_attack_vectors': attack_vectors,
            'security_recommendations': security_recommendations
        }
        
        # Display vulnerability assessment
        self._display_vulnerability_assessment(vulnerability_report)
        
        return vulnerability_report
    
    def _calculate_position_risk(self, target):
        """Calculate position-based risk factor"""
        position = target['position'].lower()
        department = target['department']
        
        # High-risk positions
        if any(role in position for role in ['ceo', 'cfo', 'cto', 'president', 'director']):
            return random.uniform(8.5, 10.0)  # Executives are high-value targets
        
        # Department-based risk
        dept_risks = {
            'IT': random.uniform(6.0, 8.0),  # IT has access to systems but security knowledge
            'Finance': random.uniform(7.5, 9.5),  # Finance has access to money/payments
            'HR': random.uniform(7.0, 9.0),  # HR has access to employee data
            'Executive': random.uniform(8.0, 10.0),  # Executives have broad access
            'Marketing': random.uniform(5.0, 7.0)  # Marketing has brand/social access
        }
        
        if department in dept_risks:
            return dept_risks[department]
        
        # Role-based risk
        if any(role in position for role in ['manager', 'lead', 'head']):
            return random.uniform(7.0, 8.5)  # Managers have approval authority
            
        if any(role in position for role in ['admin', 'assistant', 'coordinator']):
            return random.uniform(6.0, 8.0)  # Admins often have system access
        
        # Default risk
        return random.uniform(4.0, 6.0)
    
    def _get_department_targeting_frequency(self, department):
        """Get how frequently a department is targeted by phishers"""
        # Based on industry reports of targeting frequency
        targeting_frequencies = {
            'Finance': random.uniform(8.0, 10.0),  # Finance departments are heavily targeted
            'Executive': random.uniform(8.5, 10.0),  # Executives are prime targets
            'IT': random.uniform(7.0, 9.0),  # IT has valuable access
            'HR': random.uniform(6.5, 8.5),  # HR handles sensitive data and has system access
            'Marketing': random.uniform(5.0, 7.0)  # Marketing controls social/brand channels
        }
        
        return targeting_frequencies.get(department, random.uniform(4.0, 7.0))
    
    def _estimate_data_access_level(self, target):
        """Estimate target's access to sensitive data"""
        position = target['position'].lower()
        department = target['department']
        
        # Position-based access estimation
        if any(role in position for role in ['ceo', 'cfo', 'cto', 'president', 'director']):
            return random.uniform(8.5, 10.0)  # Executives have broad access
        
        # Department-based access estimation
        dept_access = {
            'IT': random.uniform(7.0, 9.0),  # IT has system/infrastructure access
            'Finance': random.uniform(8.0, 9.5),  # Finance has financial data access
            'HR': random.uniform(7.5, 9.0),  # HR has employee data access
            'Executive': random.uniform(8.5, 10.0),  # Executives have broad access
            'Marketing': random.uniform(5.0, 7.0)  # Marketing has brand/customer data
        }
        
        if department in dept_access:
            return dept_access[department]
        
        # Role-based access estimation
        if any(role in position for role in ['manager', 'lead', 'head']):
            return random.uniform(6.5, 8.0)  # Managers have departmental access
            
        if any(role in position for role in ['admin', 'assistant', 'coordinator']):
            return random.uniform(5.5, 7.5)  # Admins often have exec/system access
        
        # Default access level
        return random.uniform(3.0, 6.0)
    
    def _identify_effective_attack_vectors(self, target, vulnerability_factors):
        """Identify most effective attack vectors based on ML analysis"""
        department = target['department']
        position = target['position'].lower()
        
        # Base attack vectors
        attack_vectors = []
        
        # Add attack vectors based on department
        if department == 'Finance':
            attack_vectors.append({
                'type': 'Invoice Fraud',
                'effectiveness': random.uniform(0.7, 0.9),
                'description': 'Fake invoices or payment requests that appear to come from vendors or partners'
            })
            attack_vectors.append({
                'type': 'Banking Alert',
                'effectiveness': random.uniform(0.75, 0.95),
                'description': 'Urgent banking alerts requesting verification of suspicious transactions'
            })
        
        elif department == 'IT':
            attack_vectors.append({
                'type': 'Security Alert',
                'effectiveness': random.uniform(0.65, 0.85),
                'description': 'Security alerts about account compromise requiring immediate password reset'
            })
            attack_vectors.append({
                'type': 'Software Update',
                'effectiveness': random.uniform(0.7, 0.9),
                'description': 'Critical software update notifications requiring immediate action'
            })
        
        elif department == 'HR':
            attack_vectors.append({
                'type': 'Employee Portal',
                'effectiveness': random.uniform(0.7, 0.9),
                'description': 'HR system access notifications requesting credential verification'
            })
            attack_vectors.append({
                'type': 'Benefits Update',
                'effectiveness': random.uniform(0.75, 0.85),
                'description': 'Benefits enrollment or update notifications requiring account verification'
            })
        
        elif department == 'Executive':
            attack_vectors.append({
                'type': 'Board Communication',
                'effectiveness': random.uniform(0.8, 0.95),
                'description': 'Urgent board communications requiring immediate review of sensitive documents'
            })
            attack_vectors.append({
                'type': 'Executive Team Alert',
                'effectiveness': random.uniform(0.75, 0.9),
                'description': 'Messages appearing to come from other executives requesting urgent action'
            })
        
        elif department == 'Marketing':
            attack_vectors.append({
                'type': 'Social Media Alert',
                'effectiveness': random.uniform(0.7, 0.85),
                'description': 'Notifications about social media account issues requiring verification'
            })
            attack_vectors.append({
                'type': 'Analytics Report',
                'effectiveness': random.uniform(0.65, 0.8),
                'description': 'Marketing analytics reports requiring account access to view'
            })
        
        # Add position-specific attack vectors
        if 'manager' in position or 'director' in position or 'head' in position:
            attack_vectors.append({
                'type': 'Team Request',
                'effectiveness': random.uniform(0.75, 0.9),
                'description': 'Requests appearing to come from team members requiring manager approval'
            })
        
        if 'assistant' in position or 'coordinator' in position:
            attack_vectors.append({
                'type': 'Executive Request',
                'effectiveness': random.uniform(0.8, 0.95),
                'description': 'Urgent requests appearing to come from executives requiring immediate assistance'
            })
        
        # Add general attack vectors
        attack_vectors.append({
            'type': 'Document Share',
            'effectiveness': random.uniform(0.7, 0.85),
            'description': 'Notifications about shared documents requiring account verification to access'
        })
        
        # Sort by effectiveness
        attack_vectors.sort(key=lambda x: x['effectiveness'], reverse=True)
        
        # Return top 3 most effective attack vectors
        return attack_vectors[:3]
    
    def _generate_security_recommendations(self, target, vulnerability_factors):
        """Generate personalized security recommendations based on vulnerability factors"""
        recommendations = []
        
        # Add recommendations based on vulnerability factors
        if vulnerability_factors['security_awareness'] > 7:
            recommendations.append({
                'type': 'Security Training',
                'priority': 'High',
                'description': 'Enroll in comprehensive security awareness training focused on identifying phishing attacks'
            })
        
        if vulnerability_factors['technical_savviness'] > 6:
            recommendations.append({
                'type': 'Technical Guidance',
                'priority': 'Medium',
                'description': 'Provide simplified technical guides for verifying email authenticity and spotting spoofed domains'
            })
        
        if vulnerability_factors['position_risk'] > 7:
            recommendations.append({
                'type': 'Enhanced Verification',
                'priority': 'High',
                'description': 'Implement additional verification procedures for sensitive requests or transactions'
            })
        
        if vulnerability_factors['department_targeting_frequency'] > 7:
            recommendations.append({
                'type': 'Department Training',
                'priority': 'High',
                'description': f'Conduct specialized phishing training for the {target["department"]} department focused on common attack vectors'
            })
        
        # Add general recommendations
        recommendations.extend([
            {
                'type': 'Multi-Factor Authentication',
                'priority': 'High',
                'description': 'Ensure MFA is enabled on all accounts, especially email and financial systems'
            },
            {
                'type': 'Reporting Procedure',
                'priority': 'Medium',
                'description': 'Review procedures for reporting suspicious emails and ensure quick response times'
            },
            {
                'type': 'Simulated Phishing',
                'priority': 'Medium',
                'description': 'Include in regular simulated phishing campaigns with targeted scenarios'
            }
        ])
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def _display_vulnerability_assessment(self, report):
        """Display vulnerability assessment in a readable format"""
        print("\n" + "=" * 80)
        print("PHISHING VULNERABILITY ASSESSMENT")
        print("=" * 80)
        
        target = report['target']
        print(f"\nTARGET: {target['name']}")
        print(f"POSITION: {target['position']}")
        print(f"DEPARTMENT: {target['department']}")
        print(f"COMPANY: {target['company']}")
        
        # Vulnerability score
        score = report['vulnerability_score']
        print(f"\nVULNERABILITY SCORE: {score:.2f}/10")
        
        # Risk level
        if score >= 8:
            risk_level = "HIGH"
        elif score >= 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        print(f"RISK LEVEL: {risk_level}")
        
        # Vulnerability factors
        print("\nVULNERABILITY FACTORS:")
        for factor, value in report['vulnerability_factors'].items():
            print(f"  - {factor.replace('_', ' ').title()}: {value:.2f}/10")
        
        # Effective attack vectors
        print("\nMOST EFFECTIVE ATTACK VECTORS:")
        for i, vector in enumerate(report['effective_attack_vectors'], 1):
            print(f"  {i}. {vector['type']} (Effectiveness: {vector['effectiveness']:.2f})")
            print(f"     {vector['description']}")
        
        # Security recommendations
        print("\nSECURITY RECOMMENDATIONS:")
        for i, rec in enumerate(report['security_recommendations'], 1):
            print(f"  {i}. [{rec['priority']}] {rec['type']}")
            print(f"     {rec['description']}")
        
        print("\n" + "=" * 80)
        print("NOTE: This assessment is for educational purposes only.")
        print("=" * 80)


def main():
    """Main function for the ML Phishing Framework"""
    parser = argparse.ArgumentParser(
        description="Advanced ML/AI-Driven Phishing Framework (Educational Purposes Only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output-dir', type=str, default='output', 
                        help="Directory to save generated content")
    parser.add_argument('--targets-file', type=str, 
                        help="Path to CSV file with target data (optional)")
    parser.add_argument('--emails-file', type=str, 
                        help="Path to file with email templates (optional)")
    parser.add_argument('--target-index', type=int, default=0, 
                        help="Index of specific target to use")
    parser.add_argument('--analyze-target', action='store_true', 
                        help="Analyze target vulnerability")
    parser.add_argument('--generate-email', action='store_true', 
                        help="Generate phishing email")
    parser.add_argument('--train-models', action='store_true', 
                        help="Train ML models (default: use pre-trained/simulated)")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Print disclaimer
    print("\n" + "=" * 80)
    print("ADVANCED ML/AI-DRIVEN PHISHING FRAMEWORK - EDUCATIONAL PURPOSES ONLY")
    print("=" * 80)
    print("This framework demonstrates how machine learning and AI could potentially")
    print("be used to enhance phishing attacks. It is provided STRICTLY FOR EDUCATIONAL")
    print("AND DEFENSIVE PURPOSES to help understand these threats.")
    print("\nUsing such techniques for actual phishing is ILLEGAL and UNETHICAL.")
    print("=" * 80 + "\n")
    
    # Initialize framework
    framework = MLPhishingFramework(output_dir=args.output_dir, seed=args.seed)
    
    # Load training data
    framework.load_training_data(emails_path=args.emails_file, targets_path=args.targets_file)
    
    # Train models if requested
    if args.train_models:
        print("\n" + "=" * 60)
        print("TRAINING ML MODELS FOR PHISHING ENHANCEMENT")
        print("=" * 60)
        
        # Train target classifier
        print("\n[*] Training target classifier model...")
        framework.train_target_classifier()
        
        # Train personalization model
        print("\n[*] Training email personalization model...")
        framework.train_personalization_model()
        
        # Initialize text generator
        print("\n[*] Initializing text generation model...")
        framework.initialize_text_generator()
        
        # Cluster targets
        print("\n[*] Clustering targets by characteristics...")
        framework.cluster_targets()
    
    # Analyze target vulnerability if requested
    if args.analyze_target:
        print("\n" + "=" * 60)
        print("TARGET VULNERABILITY ANALYSIS")
        print("=" * 60)
        framework.evaluate_target_vulnerability(args.target_index)
    
    # Generate phishing email if requested
    if args.generate_email:
        print("\n" + "=" * 60)
        print("GENERATING AI-ENHANCED PHISHING EMAIL")
        print("=" * 60)
        email = framework.generate_phishing_email(args.target_index)
        
        if email:
            print(f"\n[+] Successfully generated phishing email")
            print(f"[+] Subject: \"{email['subject']}\"")
            print(f"[+] Files saved to: {args.output_dir}/")
            print("\n[+] To view the email, open the saved HTML file in a web browser")
    
    print("\n[+] Framework execution completed")


if __name__ == "__main__":
    main()
    
    def cluster_targets(self, n_clusters=3):
        """
        Cluster targets into groups with similar characteristics
        
        This helps identify distinct groups of targets that might be
        susceptible to different phishing approaches
        """
        if self.target_features is None:
            print("[!] No target features available for clustering")
            return False
        
        print(f"[+] Clustering targets into {n_clusters} groups")
        
        # Normalize features for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.target_features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to targets
        self.targets['cluster'] = clusters
        
        # Analyze clusters
        print("\nCluster Analysis:")
        for cluster_id in range(n_clusters):
            cluster_size = np.sum(clusters == cluster_id)
            cluster_percent = cluster_size / len(clusters) * 100
            
            print(f"\nCluster {cluster_id}: {cluster_size} targets ({cluster_percent:.1f}%)")
            
            # Get targets in this cluster
            cluster_targets = self.targets[self.targets['cluster'] == cluster_id]
            
            # Show cluster characteristics
            print("  Departments:", ", ".join(cluster_targets['department'].value_counts().index[:3]))
            print(f"  Avg Security Awareness: {cluster_targets['security_awareness'].mean():.2f}")
            print(f"  Avg Susceptibility: {cluster_targets['susceptibility_score'].mean():.2f}")
            
            # Show a sample target from the cluster
            sample_idx = cluster_targets.index[0]
            sample = cluster_targets.iloc[0]
            print(f"  Sample Target: {sample['name']}, {sample['position']} at {sample['company']}")
        
        # Save the model
        self.models['target_clustering'] = kmeans
        
        return True
    
    def initialize_text_generator(self, pretrained_model='gpt2'):
        """
        Initialize a text generation model for creating personalized content
        
        For educational purposes, this uses a pre-trained model with simple fine-tuning.
        A real attack might use a custom-trained model on specific corporate communications.
        
        In a real-world attack scenario, the adversary might:
        1. Scrape company communications, emails, and documentation
        2. Fine-tune a language model on this corpus to mimic corporate style
        3. Use reinforcement learning to optimize for user engagement
        """
        print(f"[+] Initializing text generation model ({pretrained_model})")
        
        try:
            # Load pre-trained GPT-2 model and tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
            model = GPT2LMHeadModel.from_pretrained(pretrained_model)
            
            # Add required special tokens
            special_tokens = {
                'additional_special_tokens': [
                    '{{name}}', '{{company}}', '{{position}}', 
                    '{{email}}', '{{reset_link}}'
                ]
            }
            tokenizer.add_special_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer))
            
            # Store the model and tokenizer
            self.text_generator = model
            self.text_tokenizer = tokenizer
            
            print("[+] Text generation model initialized successfully")
            return True
            
        except Exception as e:
            print(f"[!] Error initializing text generation model: {e}")
            
            # Fallback to template-based generation for educational demo
            print("[+] Using template-based generation instead")
            self.text_generator = None
            self.text_tokenizer = None
            return False
    
    def train_personalization_model(self):
        """
        Train a model to personalize phishing content for each target
        
        This model predicts which elements to include in a phishing email
        based on the target's profile
        """
        if self.target_features is None:
            print("[!] No target features available for training personalization model")
            return False
        
        print("[+] Training email personalization model")
        
        # Generate synthetic personalization data
        # In a real scenario, this would come from historical campaign data
        
        # Features that might be effective for different targets
        personalization_features = [
            'include_position_reference',  # Reference their job role
            'include_technical_details',   # Include technical details
            'emphasize_urgency',           # Emphasize time sensitivity
            'use_formal_language',         # Use formal vs casual language
            'include_company_branding',    # Include company branding/references
            'reference_colleagues',        # Reference colleagues or team
            'mention_specific_systems',    # Mention specific IT systems
            'include_financial_details'    # Include financial/budget references
        ]
        
        # Generate synthetic effectiveness data based on target profiles
        X = self.target_features.values
        y = np.zeros((len(X), len(personalization_features)))
        
        for i, row in enumerate(self.targets.iterrows()):
            target = row[1]
            dept = target['department']
            position = str(target['position']).lower()
            tech_savvy = target['tech_savviness']
            
            # Personalization effectiveness varies by target characteristics
            # These are simplistic rules for educational purposes
            
            # Position reference effectiveness
            y[i, 0] = np.random.normal(0.8, 0.1)  # Generally effective
            
            # Technical details effectiveness
            if dept == 'IT' or tech_savvy > 7.0:
                y[i, 1] = np.random.normal(0.9, 0.1)  # More effective for technical roles
            else:
                y[i, 1] = np.random.normal(0.4, 0.1)  # Less effective for non-technical roles
            
            # Urgency effectiveness
            if dept == 'Executive' or 'manager' in position or 'director' in position:
                y[i, 2] = np.random.normal(0.9, 0.1)  # Executives respond to urgency
            else:
                y[i, 2] = np.random.normal(0.7, 0.1)  # Still fairly effective for others
            
            # Formal language effectiveness
            if dept == 'Executive' or dept == 'Finance':
                y[i, 3] = np.random.normal(0.8, 0.1)  # More formal for executives/finance
            else:
                y[i, 3] = np.random.normal(0.6, 0.1)  # Less formal for others
            
            # Company branding effectiveness
            y[i, 4] = np.random.normal(0.85, 0.1)  # Generally effective
            
            # Colleague references effectiveness
            if dept == 'HR' or 'manager' in position:
                y[i, 5] = np.random.normal(0.85, 0.1)  # More effective for people-focused roles
            else:
                y[i, 5] = np.random.normal(0.6, 0.1)
            
            # System reference effectiveness
            if dept == 'IT' or tech_savvy > 6.0:
                y[i, 6] = np.random.normal(0.9, 0.1)  # More effective for IT
            else:
                y[i, 6] = np.random.normal(0.5, 0.1)
            
            # Financial details effectiveness
            if dept == 'Finance' or dept == 'Executive' or 'account' in position:
                y[i, 7] = np.random.normal(0.9, 0.1)  # More effective for finance-related roles
            else:
                y[i, 7] = np.random.normal(0.5, 0.1)
            
            # Clip to valid range [0, 1]
            y[i] = np.clip(y[i], 0, 1)
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a neural network for personalization
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(len(personalization_features), activation='sigmoid')
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"[+] Personalization model trained with accuracy: {accuracy:.4f}")
        
        # Save the model
        self.models['personalization_model'] = model
        
        # Save feature names for reference
        self.models['personalization_features'] = personalization_features
        
        # For educational purposes, show how different target characteristics 
        # influence personalization strategies
        print("\nPersonalization Strategy Analysis:")
        print("(How target characteristics influence email customization)")
        
        # Show correlation between target features and personalization strategies
        correlations = {}
        for i, feature in enumerate(personalization_features):
            for col in self.target_features.columns:
                corr = np.corrcoef(self.target_features[col], y[:, i])[0, 1]
                if abs(corr) > 0.3:  # Only show stronger correlations
                    if feature not in correlations:
                        correlations[feature] = []
                    correlations[feature].append((col, corr))
        
        for feature, corrs in correlations.items():
            if corrs:
                print(f"\n  {feature}:")
                for col, corr in sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:3]:
                    print(f"    - {col}: {corr:.3f}")
        
        return True
