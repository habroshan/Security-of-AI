#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deepfake Educational Framework - Simplified Version
===================================================

This standalone script demonstrates the key concepts of deepfake detection
for educational purposes. It includes:

1. Sample data generation
2. Feature visualization
3. Simple detection model
4. Educational explanations

For CyBOK educational materials.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class DeepfakeEducationalFramework:
    """Simplified framework for deepfake education"""
    
    def __init__(self, output_dir="output"):
        """Initialize the framework"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = os.path.join(output_dir, "data")
        self.results_dir = os.path.join(output_dir, "results")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.datasets = {}
        self.models = {}
        
        print("[+] Deepfake Educational Framework initialized")
        print("[+] This is for EDUCATIONAL PURPOSES ONLY")
    
    def generate_sample_data(self, num_samples=500):
        """Generate synthetic data for demonstrating deepfake detection"""
        print(f"[+] Generating {num_samples} synthetic samples (50% real, 50% fake)")
        
        # Number of samples in each class
        num_real = num_samples // 2
        num_fake = num_samples - num_real
        
        # Feature names - these simulate characteristics analyzed in deepfake detection
        feature_names = [
            'texture_consistency', 
            'eye_blink_frequency',
            'facial_boundary_consistency',
            'color_consistency',
            'lighting_consistency',
            'noise_pattern_consistency',
            'motion_smoothness',
            'micro_expression_presence',
            'compression_artifacts',
            'background_foreground_consistency'
        ]
        
        # Generate synthetic features for real faces
        # Higher values generally indicate more "natural" characteristics
        real_features = np.random.normal(loc=0.75, scale=0.15, size=(num_real, len(feature_names)))
        real_features = np.clip(real_features, 0, 1)  # Clip to valid range
        
        # Generate synthetic features for fake faces
        # Lower values generally indicate more manipulation artifacts
        fake_features = np.random.normal(loc=0.35, scale=0.20, size=(num_fake, len(feature_names)))
        fake_features = np.clip(fake_features, 0, 1)  # Clip to valid range
        
        # Add some nuance - not all features are equally reliable indicators
        
        # For real faces: Add some natural variation
        real_features[:, 7] = np.random.normal(0.65, 0.2, num_real)  # Micro-expressions vary naturally
        real_features[:, 8] = np.random.normal(0.55, 0.25, num_real)  # Compression artifacts vary by camera/lighting
        
        # For fake faces: Some deepfakes are better at specific aspects
        fake_features[:, 0] = np.random.normal(0.6, 0.2, num_fake)  # Some fakes have good texture
        fake_features[:, 3] = np.random.normal(0.7, 0.15, num_fake)  # Color consistency can be good in fakes
        
        # Clip all values to [0, 1] range
        real_features = np.clip(real_features, 0, 1)
        fake_features = np.clip(fake_features, 0, 1)
        
        # Create labels
        real_labels = np.ones(num_real)
        fake_labels = np.zeros(num_fake)
        
        # Combine datasets
        X = np.vstack([real_features, fake_features])
        y = np.hstack([real_labels, fake_labels])
        
        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Store in the datasets dictionary
        self.datasets['features'] = X
        self.datasets['labels'] = y
        self.datasets['feature_names'] = feature_names
        self.datasets['real_count'] = num_real
        self.datasets['fake_count'] = num_fake
        
        # Create a DataFrame for easier analysis
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        df['is_real'] = df['label'].apply(lambda x: 'Real' if x == 1 else 'Fake')
        
        # Save to CSV for reference
        csv_path = os.path.join(self.data_dir, "synthetic_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"[+] Generated synthetic features saved to {csv_path}")
        
        # Visualize the differences between real and fake distributions
        self._visualize_feature_distributions(df)
        
        return df
    
    def _visualize_feature_distributions(self, df):
        """Create visualizations of the feature distributions"""
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(df.columns[:-2]):  # Skip the label and is_real columns
            plt.subplot(3, 4, i+1)
            sns.histplot(data=df, x=feature, hue='is_real', element='step', stat='density', common_norm=False)
            plt.title(f'Distribution of {feature.replace("_", " ").title()}')
            plt.xlabel(feature)
            plt.ylabel('Density')
        
        plt.tight_layout()
        viz_path = os.path.join(self.data_dir, "feature_distributions.png")
        plt.savefig(viz_path)
        plt.close()
        
        print(f"[+] Feature distribution visualization saved to {viz_path}")
    
    def train_detection_model(self):
        """Train a simple model to detect deepfakes based on features"""
        if 'features' not in self.datasets or 'labels' not in self.datasets:
            print("[!] No data available. Generate sample data first with generate_sample_data()")
            return False
        
        print("[+] Training deepfake detection model")
        
        # Get features and labels
        X = self.datasets['features']
        y = self.datasets['labels']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"[+] Model trained with accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        # Calculate feature importance
        feature_importance = model.feature_importances_
        
        # Create feature importance visualization
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importance)
        feature_names = self.datasets['feature_names']
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title('Feature Importance for Deepfake Detection')
        plt.xlabel('Importance')
        plt.tight_layout()
        fi_path = os.path.join(self.results_dir, "feature_importance.png")
        plt.savefig(fi_path)
        plt.close()
        
        # Save the model and results
        self.models['detection_model'] = model
        self.models['feature_importance'] = {name: imp for name, imp in zip(feature_names, feature_importance)}
        self.models['accuracy'] = accuracy
        
        # Create comprehensive report
        self._save_detection_report(accuracy, y_test, y_pred, self.models['feature_importance'])
        
        return model
    
    def _save_detection_report(self, accuracy, y_true, y_pred, feature_importance):
        """Save a comprehensive detection report"""
        report_path = os.path.join(self.results_dir, "detection_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEEPFAKE DETECTION MODEL ANALYSIS - EDUCATIONAL PURPOSES ONLY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            
            f.write("CLASSIFICATION REPORT:\n")
            f.write(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))
            f.write("\n")
            
            f.write("FEATURE IMPORTANCE:\n")
            sorted_features = {k: v for k, v in sorted(feature_importance.items(), 
                                                    key=lambda item: item[1], reverse=True)}
            for feature, importance in sorted_features.items():
                f.write(f"  {feature}: {importance:.4f}\n")
            
            f.write("\n")
            f.write("KEY INSIGHTS:\n")
            
            # Get top 3 features
            top_features = list(sorted_features.keys())[:3]
            f.write(f"1. The most reliable indicators of deepfakes were: {', '.join(top_features)}\n")
            
            # Calculate detection rates
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fake_detection_rate = tp / (tp + fn)
            real_recognition_rate = tn / (tn + fp)
            
            f.write(f"2. The model correctly identified {fake_detection_rate:.1%} of deepfakes\n")
            f.write(f"3. The model correctly recognized {real_recognition_rate:.1%} of authentic media\n")
            
            f.write("\nTHREAT ANALYSIS:\n")
            f.write("1. Current deepfake technology appears most detectable through inconsistencies in\n")
            f.write(f"   {top_features[0]} and {top_features[1]}\n")
            f.write("2. The most advanced deepfakes might now be focusing on improving these aspects\n")
            f.write("3. Detection techniques will need to evolve as deepfake technology improves\n")
            
            f.write("\nIMPLICATIONS:\n")
            f.write("1. Media verification systems should focus on the most reliable detection features\n")
            f.write("2. Multi-modal detection (analyzing audio, metadata, and context) will increase reliability\n")
            f.write("3. User awareness remains critical as deepfake technology continues to advance\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("EDUCATIONAL NOTE: This analysis demonstrates how AI can be used to detect\n")
            f.write("deepfakes. Understanding these techniques helps develop better defenses against\n")
            f.write("malicious uses of synthetic media technologies.\n")
            f.write("=" * 80 + "\n")
        
        print(f"[+] Comprehensive detection report saved to {report_path}")
    
    def demonstrate_detection(self, image_path=None):
        """
        Demonstrate deepfake detection on an image or using synthetic features
        
        If no image is provided, this will use synthetic features
        for educational demonstration.
        """
        if 'detection_model' not in self.models:
            print("[!] No detection model available. Please train a model first with train_detection_model()")
            return None
        
        # Generate a synthetic feature vector for demonstration
        if random.random() > 0.5:
            # Generate "real" features
            features = np.random.normal(loc=0.8, scale=0.1, size=(1, len(self.datasets['feature_names'])))
            true_label = "Real"
        else:
            # Generate "fake" features
            features = np.random.normal(loc=0.3, scale=0.15, size=(1, len(self.datasets['feature_names'])))
            true_label = "Fake"
        
        features = np.clip(features, 0, 1)  # Clip to valid range
        
        # Make prediction
        prediction = self.models['detection_model'].predict_proba(features)[0]
        predicted_label = "Real" if prediction[1] > 0.5 else "Fake"
        confidence = prediction[1] if prediction[1] > 0.5 else 1 - prediction[1]
        
        print("\n" + "=" * 60)
        print("DEEPFAKE DETECTION RESULTS")
        print("=" * 60)
        
        print(f"\nPrediction: {predicted_label} (Confidence: {confidence:.2%})")
        print(f"True label (for this synthetic example): {true_label}")
        
        # Show feature analysis
        print("\nFeature Analysis:")
        feature_names = self.datasets['feature_names']
        feature_importance = self.models['feature_importance']
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features[:5]:  # Show top 5 features
            idx = feature_names.index(feature)
            value = features[0, idx]
            
            # Determine if this feature suggests real or fake
            suggests = "Real" if ((value > 0.5 and predicted_label == "Real") or 
                                (value < 0.5 and predicted_label == "Fake")) else "Fake"
            
            print(f"  - {feature.replace('_', ' ').title()}: {value:.2f} (suggests {suggests}, importance: {importance:.4f})")
        
        # Create a visualization of the detection
        plt.figure(figsize=(12, 6))
        
        # Bar chart of feature values
        plt.subplot(1, 2, 1)
        plt.bar(range(len(feature_names)), features[0], color='skyblue')
        plt.xticks(range(len(feature_names)), [name.replace('_', '\n') for name in feature_names], rotation=45, ha='right')
        plt.title('Feature Values')
        plt.ylim(0, 1)
        
        # Importance vs. Value plot
        plt.subplot(1, 2, 2)
        x = [feature_importance[name] for name in feature_names]
        y = features[0]
        plt.scatter(x, y, c=['green' if v > 0.5 else 'red' for v in y], alpha=0.7)
        
        for i, name in enumerate(feature_names):
            plt.annotate(name.replace('_', ' '), (x[i], y[i]), fontsize=8)
            
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Value')
        plt.title('Importance vs. Value Analysis')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = os.path.join(self.results_dir, f"detection_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(viz_path)
        plt.show()
        plt.close()
        
        # Save a detection report
        self._save_detection_demo(features, prediction, feature_names, feature_importance, viz_path)
        
        return predicted_label, confidence
    
    def _save_detection_demo(self, features, prediction, feature_names, feature_importance, viz_path):
        """Save the detection demonstration results"""
        demo_path = os.path.join(self.results_dir, f"detection_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(demo_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEEPFAKE DETECTION DEMONSTRATION - EDUCATIONAL PURPOSES ONLY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DETECTION RESULT:\n")
            predicted_label = "Real" if prediction[1] > 0.5 else "Fake"
            confidence = prediction[1] if prediction[1] > 0.5 else 1 - prediction[1]
            f.write(f"Prediction: {predicted_label} (Confidence: {confidence:.2%})\n\n")
            
            f.write("FEATURE ANALYSIS:\n")
            
            # Sort features by importance
            sorted_indices = [feature_names.index(feat) for feat, _ in 
                             sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)]
            
            for idx in sorted_indices:
                feature = feature_names[idx]
                value = features[0, idx]
                importance = feature_importance[feature]
                
                suggests = "Real" if ((value > 0.5 and predicted_label == "Real") or 
                                     (value < 0.5 and predicted_label == "Fake")) else "Fake"
                
                f.write(f"  - {feature.replace('_', ' ').title()}: {value:.2f} ")
                f.write(f"(suggests {suggests}, importance: {importance:.4f})\n")
            
            f.write("\nDECISION EXPLANATION:\n")
            if predicted_label == "Real":
                f.write("This appears to be authentic media because:\n")
                high_authenticity_indicators = [(feature_names[idx], features[0, idx]) 
                                              for idx in sorted_indices[:5] 
                                              if features[0, idx] > 0.6]
                for feature, value in high_authenticity_indicators:
                    f.write(f"  - Strong {feature.replace('_', ' ')} consistency ({value:.2f})\n")
            else:
                f.write("This appears to be manipulated media because:\n")
                manipulation_indicators = [(feature_names[idx], features[0, idx]) 
                                         for idx in sorted_indices[:5] 
                                         if features[0, idx] < 0.4]
                for feature, value in manipulation_indicators:
                    f.write(f"  - Poor {feature.replace('_', ' ')} consistency ({value:.2f})\n")
            
            f.write("\nVISUALIZATION:\n")
            f.write(f"A visualization of this analysis has been saved to: {viz_path}\n")
            
            f.write("\nEDUCATIONAL NOTE:\n")
            f.write("This demonstration shows how AI-based detection systems analyze features to\n")
            f.write("determine if media is authentic or manipulated. In real systems, these features\n")
            f.write("would be extracted from images or videos using computer vision techniques.\n")
            f.write("Understanding these detection methods is crucial for developing defenses against\n")
            f.write("malicious uses of deepfake technology.\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"[+] Detection demonstration saved to {demo_path}")
        return demo_path
        
    def analyze_security_implications(self):
        """Provide an educational overview of security implications"""
        print("\n" + "=" * 70)
        print("DEEPFAKE SECURITY AND PRIVACY IMPLICATIONS")
        print("=" * 70 + "\n")
        
        print("SECURITY THREATS:\n")
        
        print("1. Identity Fraud and Impersonation")
        print("   - Voice and video deepfakes can bypass biometric authentication")
        print("   - Synthetic identities can be created for fraud and social engineering")
        print("   - Business email compromise enhanced with voice synthesis\n")
        
        print("2. Misinformation and Disinformation")
        print("   - Fabricated videos of public figures making inflammatory statements")
        print("   - Synthetic events that never occurred presented as news")
        print("   - Targeted disinformation campaigns using personalized synthetic content\n")
        
        print("3. Social and Political Manipulation")
        print("   - Undermining trust in legitimate media")
        print("   - Creating false narratives to influence public opinion")
        print("   - Election interference with timely fake content\n")
        
        print("4. Personal Privacy Violations")
        print("   - Non-consensual imagery creation")
        print("   - Harassment and reputation damage")
        print("   - Blackmail using synthetic compromising content\n")
        
        print("\nDEFENSIVE STRATEGIES:\n")
        
        print("1. Technical Defenses")
        print("   - AI-powered deepfake detection systems")
        print("   - Digital content provenance and authentication")
        print("   - Blockchain-based media verification")
        print("   - Watermarking and fingerprinting of original media\n")
        
        print("2. Organizational Responses")
        print("   - Enhanced verification protocols for sensitive communications")
        print("   - Multi-factor authentication beyond biometrics")
        print("   - Staff training on synthetic media awareness")
        print("   - Incident response planning for deepfake scenarios\n")
        
        print("3. Individual Protection")
        print("   - Critical media literacy skills")
        print("   - Verification of information through multiple channels")
        print("   - Careful management of personal image and voice data")
        print("   - Awareness of current deepfake capabilities\n")
        
        # Save security analysis report
        self._create_security_analysis_report()
        
        print("\n" + "=" * 70)
        print("This analysis is provided for educational purposes to foster")
        print("understanding of the security implications of deepfake technology.")
        print("=" * 70)
    
    def _create_security_analysis_report(self):
        """Create a security analysis report"""
        report_path = os.path.join(self.results_dir, "security_analysis.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEEPFAKE SECURITY ANALYSIS - EDUCATIONAL PURPOSES ONLY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("----------------\n")
            f.write("Deepfake technology presents significant and evolving security challenges across\n")
            f.write("multiple domains. This analysis examines the threat landscape, potential impacts,\n")
            f.write("and effective defensive strategies.\n\n")
            
            f.write("THREAT LANDSCAPE\n")
            f.write("---------------\n\n")
            
            f.write("1. IDENTITY-BASED THREATS\n\n")
            
            f.write("1.1 Authentication Bypass\n")
            f.write("Synthetic voice and video can potentially bypass biometric authentication systems\n")
            f.write("including voice recognition, facial recognition, and liveness detection.\n\n")
            
            f.write("1.2 Executive Impersonation\n")
            f.write("Business email compromise attacks enhanced with synthetic voice matching an\n")
            f.write("executive's speech patterns have already been documented in real-world attacks.\n\n")
            
            f.write("1.3 Synthetic Identities\n")
            f.write("Entirely fabricated personas can be created using GANs to generate realistic faces\n")
            f.write("that don't belong to real people, enabling various forms of fraud.\n\n")
            
            f.write("2. INFORMATION INTEGRITY THREATS\n\n")
            
            f.write("2.1 Disinformation Campaigns\n")
            f.write("Deepfakes enable highly convincing false narratives that can be difficult to debunk\n")
            f.write("quickly enough to prevent viral spread before elections or key events.\n\n")
            
            f.write("2.2 Evidence Manipulation\n")
            f.write("The ability to create convincing false video or audio raises questions about the\n")
            f.write("reliability of digital evidence in legal proceedings and journalism.\n\n")
            
            f.write("2.3 Reality Skepticism\n")
            f.write("As deepfakes become more prevalent, a 'liar's dividend' emerges where genuine\n")
            f.write("recordings can be dismissed as fake, undermining trust in legitimate media.\n\n")
            
            f.write("DEFENSIVE STRATEGIES\n")
            f.write("-------------------\n\n")
            
            f.write("1. TECHNICAL COUNTERMEASURES\n\n")
            
            f.write("1.1 Detection Systems\n")
            f.write("AI-powered detection systems analyze media for manipulation artifacts. These include\n")
            f.write("inconsistencies in blinking patterns, facial movements, pulse detection, and more.\n\n")
            
            f.write("1.2 Content Authentication\n")
            f.write("Digital signatures, content provenance frameworks, and blockchain-based verification\n")
            f.write("can establish chains of custody for media from capture to consumption.\n\n")
            
            f.write("1.3 Multi-factor Verification\n")
            f.write("Organizations should implement verification protocols that don't rely solely on\n")
            f.write("voice or video confirmation for sensitive actions like financial transfers.\n\n")
            
            f.write("EDUCATIONAL RESOURCES\n")
            f.write("--------------------\n")
            f.write("1. Detection Demos: Tools to help understand how deepfakes can be identified\n")
            f.write("2. Media Literacy Training: Educational materials on critical media evaluation\n")
            f.write("3. Technical Documentation: Research papers on detection methodologies\n")
            f.write("4. Case Studies: Analysis of real-world deepfake incidents and responses\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("This analysis is provided for educational purposes to foster understanding\n")
            f.write("of the security implications of deepfake technology and to help develop\n")
            f.write("appropriate protective measures.\n")
            f.write("=" * 80 + "\n")
        
        print(f"[+] Security analysis report saved to {report_path}")
        return report_path

    def visualize_deepfake_example(self, original_image_path=None, use_sample=True):
        """
        Demonstrate deepfake visualization using sample or provided images
        
        This is for EDUCATIONAL PURPOSES ONLY - it shows what deepfakes might look like
        but does not actually create deepfakes.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter
        import cv2
        
        # Create output directory
        output_dir = os.path.join(self.output_dir, "deepfake_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        if original_image_path and os.path.exists(original_image_path):
            # Load the provided image
            try:
                original = Image.open(original_image_path)
                print(f"Using provided image: {original_image_path}")
            except Exception as e:
                print(f"Error loading image: {e}")
                use_sample = True
        else:
            use_sample = True
        
        if use_sample:
            # Create a sample image (blank face outline)
            print("Using generated sample image")
            original = Image.new('RGB', (400, 400), color=(240, 240, 240))
            draw = ImageDraw.Draw(original)
            
            # Draw a simple face outline
            # Head
            draw.ellipse((100, 50, 300, 350), outline=(80, 80, 80), width=2)
            # Eyes
            draw.ellipse((150, 120, 190, 160), outline=(80, 80, 80), width=2)
            draw.ellipse((210, 120, 250, 160), outline=(80, 80, 80), width=2)
            # Nose
            draw.line((200, 180, 200, 220), fill=(80, 80, 80), width=2)
            draw.arc((180, 210, 220, 230), 0, 180, fill=(80, 80, 80), width=2)
            # Mouth
            draw.arc((150, 240, 250, 290), 0, 180, fill=(80, 80, 80), width=2)
        
        # Create a "deepfaked" version by making some modifications
        # In a real scenario, this would be the output of a deepfake algorithm
        # Here we're just making visual changes to simulate what a deepfake might look like
        
        deepfaked = original.copy()
        
        # Convert to numpy array for OpenCV processing
        deepfaked_np = np.array(deepfaked)
        
        # Detect face if using a real image
        if not use_sample:
            # Try to detect a face using a simple Haar cascade
            try:
                # Use OpenCV's face detector
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(deepfaked_np, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Get the largest face
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    
                    # Make modifications to the face region
                    face_region = deepfaked_np[y:y+h, x:x+w]
                    
                    # Modify eyes (simple change for demonstration)
                    eye_h = h // 4
                    eye_y = y + h // 3
                    eye_w = w // 5
                    
                    # Left eye
                    left_eye_x = x + w // 3 - eye_w // 2
                    # Modify left eye region
                    deepfaked_np[eye_y:eye_y+eye_h, left_eye_x:left_eye_x+eye_w] = cv2.GaussianBlur(
                        deepfaked_np[eye_y:eye_y+eye_h, left_eye_x:left_eye_x+eye_w], (5, 5), 0)
                    
                    # Right eye
                    right_eye_x = x + 2*w // 3 - eye_w // 2
                    # Modify right eye region
                    deepfaked_np[eye_y:eye_y+eye_h, right_eye_x:right_eye_x+eye_w] = cv2.GaussianBlur(
                        deepfaked_np[eye_y:eye_y+eye_h, right_eye_x:right_eye_x+eye_w], (5, 5), 0)
                    
                    # Modify mouth
                    mouth_y = y + 2*h // 3
                    mouth_h = h // 4
                    mouth_w = w // 2
                    mouth_x = x + w // 2 - mouth_w // 2
                    
                    # Apply a slight distortion to mouth region
                    deepfaked_np[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w] = cv2.resize(
                        cv2.resize(deepfaked_np[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w], 
                                   (mouth_w//2, mouth_h//2)), 
                        (mouth_w, mouth_h))
                else:
                    print("No face detected, applying generic modifications")
                    # Apply generic modifications if no face detected
                    deepfaked_np = cv2.GaussianBlur(deepfaked_np, (5, 5), 0)
                    
            except Exception as e:
                print(f"Error in face detection: {e}")
                # Apply generic modifications
                deepfaked_np = cv2.GaussianBlur(deepfaked_np, (5, 5), 0)
        else:
            # For sample image, make specific modifications
            draw = ImageDraw.Draw(deepfaked)
            
            # Change eye shapes
            draw.ellipse((150, 120, 190, 160), fill=(255, 255, 255), outline=(80, 80, 80), width=2)
            draw.ellipse((210, 120, 250, 160), fill=(255, 255, 255), outline=(80, 80, 80), width=2)
            draw.ellipse((160, 130, 180, 150), fill=(50, 50, 150), outline=(0, 0, 0), width=1)
            draw.ellipse((220, 130, 240, 150), fill=(50, 50, 150), outline=(0, 0, 0), width=1)
            
            # Change mouth expression
            draw.arc((150, 240, 250, 300), 180, 360, fill=(80, 80, 80), width=2)
            
            # Convert back to numpy for consistency
            deepfaked_np = np.array(deepfaked)
        
        # Add some grain and artifacts to simulate common deepfake issues
        noise = np.random.normal(0, 5, deepfaked_np.shape).astype(np.uint8)
        deepfaked_np = cv2.add(deepfaked_np, noise)
        
        # Add slight color inconsistencies
        # Add slight color inconsistencies
        channels = list(cv2.split(deepfaked_np))  # Convert tuple to list
        channels[0] = cv2.addWeighted(channels[0], 1.1, np.zeros_like(channels[0]), 0, 0)
        deepfaked_np = cv2.merge(channels)
        
        # Add a subtle boundary artifact (often seen in deepfakes)
        if not use_sample:
            try:
                if len(faces) > 0:
                    # Draw a subtle boundary around the face region
                    mask = np.zeros_like(deepfaked_np)
                    cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    mask = cv2.GaussianBlur(mask, (9, 9), 0)
                    
                    # Blend the boundary with the image
                    alpha = 0.05
                    deepfaked_np = cv2.addWeighted(deepfaked_np, 1-alpha, mask, alpha, 0)
            except:
                pass
        
        # Convert back to PIL
        deepfaked = Image.fromarray(deepfaked_np)
        
        # Save the images
        original_path = os.path.join(output_dir, "original.jpg")
        deepfaked_path = os.path.join(output_dir, "deepfaked.jpg")
        
        original.save(original_path)
        deepfaked.save(deepfaked_path)
        
        # Create a visualization with both images
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(original))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(deepfaked))
        plt.title("Deepfaked Image")
        plt.axis('off')
        
        # Add annotations pointing out artifacts
        if not use_sample:
            try:
                if len(faces) > 0:
                    # Annotate eye region
                    plt.annotate('Modified eye region', 
                               xy=(right_eye_x + eye_w/2, eye_y + eye_h/2), 
                               xytext=(right_eye_x + eye_w*2, eye_y - eye_h),
                               arrowprops=dict(facecolor='red', shrink=0.05))
                    
                    # Annotate mouth region
                    plt.annotate('Altered expression', 
                               xy=(mouth_x + mouth_w/2, mouth_y + mouth_h/2), 
                               xytext=(mouth_x + mouth_w*2, mouth_y + mouth_h),
                               arrowprops=dict(facecolor='red', shrink=0.05))
                    
                    # Annotate boundary
                    plt.annotate('Blending artifacts', 
                               xy=(x + w, y + h/2), 
                               xytext=(x + w*1.2, y + h/2),
                               arrowprops=dict(facecolor='red', shrink=0.05))
            except:
                pass
        else:
            # Annotate sample image
            plt.annotate('Modified eyes', 
                       xy=(220, 140), 
                       xytext=(280, 100),
                       arrowprops=dict(facecolor='red', shrink=0.05))
            
            plt.annotate('Changed expression', 
                       xy=(200, 270), 
                       xytext=(280, 280),
                       arrowprops=dict(facecolor='red', shrink=0.05))
        
        # Add educational notes
        plt.figtext(0.5, 0.01, 
                    "EDUCATIONAL NOTE: This visualization demonstrates common deepfake artifacts.\n"
                    "Real deepfakes use complex AI algorithms to swap or manipulate facial features.",
                    ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # Save and show the comparison visualization
        comparison_path = os.path.join(output_dir, "comparison.jpg")
        plt.savefig(comparison_path)
        plt.show()
        
        print("\n=== Deepfake Visualization (EDUCATIONAL ONLY) ===")
        print(f"Original image saved to: {original_path}")
        print(f"Simulated deepfake saved to: {deepfaked_path}")
        print(f"Comparison visualization saved to: {comparison_path}")
        print("\nNOTE: This is a simplified visualization for educational purposes.")
        print("Real deepfakes use sophisticated AI models to generate or manipulate face imagery.")
        print("This demonstration helps understand what artifacts to look for in potential deepfakes.")
        
        return original_path, deepfaked_path, comparison_path

# Simple demonstration
def main():
    """Run a simple demonstration of the framework"""
    print("\n" + "=" * 80)
    print("DEEPFAKE EDUCATIONAL FRAMEWORK - DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows how the framework can be used for educational purposes.")
    
    # Create the framework
    framework = DeepfakeEducationalFramework(output_dir="deepfake_demo_output")
    
    # Generate sample data
    framework.generate_sample_data(num_samples=500)
    
    # Train detection model
    framework.train_detection_model()
    
    # Demonstrate detection
    framework.demonstrate_detection()
    
    # Show security implications
    framework.analyze_security_implications()
    
    # Add the visualization demo
    print("\n\nDemonstrating deepfake visualization...")
    # Notice the usage of "None" as the first parameter, not the framework object itself
    framework.visualize_deepfake_example(None, use_sample=True)
    
    print("\n" + "=" * 80)
    print("Demonstration completed. All outputs have been saved to the output directories.")
    print("This framework is designed for educational purposes to understand deepfake")
    print("detection concepts and security implications.")
    print("=" * 80)

if __name__ == "__main__":
    main()