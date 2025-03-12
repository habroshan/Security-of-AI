#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deepfake Detection Demo Script - EDUCATIONAL PURPOSES ONLY

This script demonstrates a simple deepfake detection demo that can be used
for educational workshops and cybersecurity classes. It uses the DeepfakeEducationalFramework
to create an interactive demonstration of deepfake detection concepts.

For CyBOK (Cyber Security Body of Knowledge) educational materials.
"""

import os
import sys
import argparse
import time
import random
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox

# Check if the Deepfake Educational Framework is available
try:
    from deepfake_educational_framework import DeepfakeEducationalFramework
except ImportError:
    print("[!] Error: DeepfakeEducationalFramework module not found.")
    print("[!] Please make sure the deepfake_educational_framework.py file is in the same directory.")
    sys.exit(1)

class DeepfakeDetectionDemo:
    """Interactive deepfake detection demonstration for educational purposes"""
    
    def __init__(self, output_dir="demo_output"):
        """Initialize the demo application"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the educational framework
        self.framework = DeepfakeEducationalFramework(output_dir=os.path.join(output_dir, "framework_output"))
        
        # Load sample data and train model in background
        self.model_ready = False
        self.selected_image = None
        self.results = None
        
        # Setup GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the graphical user interface"""
        self.root = tk.Tk()
        self.root.title("Deepfake Detection Educational Demo")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Create style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12))
        self.style.configure("TLabel", font=("Arial", 12))
        self.style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        self.style.configure("Result.TLabel", font=("Arial", 12, "bold"))
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(
            header_frame, 
            text="Deepfake Detection Educational Demo", 
            font=("Arial", 18, "bold")
        ).pack(side=tk.LEFT)
        
        # Disclaimer label
        ttk.Label(
            header_frame,
            text="FOR EDUCATIONAL PURPOSES ONLY",
            foreground="red",
            font=("Arial", 12, "bold")
        ).pack(side=tk.RIGHT)
        
        # Create content frame (2 columns)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Image and controls
        left_frame = ttk.Frame(content_frame, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image frame
        self.image_frame = ttk.LabelFrame(left_frame, text="Image Analysis", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Image display
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Image placeholder
        placeholder_text = ttk.Label(
            self.image_label, 
            text="No image selected. Please select an image to analyze.", 
            font=("Arial", 12)
        )
        placeholder_text.pack(pady=50)
        
        # Controls frame
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Select image button
        self.select_button = ttk.Button(
            controls_frame,
            text="Select Image",
            command=self.select_image
        )
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Analyze button
        self.analyze_button = ttk.Button(
            controls_frame,
            text="Analyze Image",
            command=self.analyze_image,
            state=tk.DISABLED
        )
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Use sample button
        self.sample_button = ttk.Button(
            controls_frame,
            text="Use Sample Image",
            command=self.use_sample_image
        )
        self.sample_button.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.Frame(left_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select an image to begin.")
        self.status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var,
            font=("Arial", 10, "italic")
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Right column - Results and information
        right_frame = ttk.Frame(content_frame, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(right_frame, text="Detection Results", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Initial results content - placeholder
        self.result_var = tk.StringVar()
        self.result_var.set("No analysis performed yet.")
        
        self.result_label = ttk.Label(
            self.results_frame, 
            textvariable=self.result_var,
            style="Result.TLabel"
        )
        self.result_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            self.results_frame, 
            height=20, 
            wrap=tk.WORD,
            font=("Courier New", 10)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "Detailed analysis will appear here after processing an image.")
        self.results_text.config(state=tk.DISABLED)
        
        # Information frame
        info_frame = ttk.LabelFrame(right_frame, text="Educational Information", padding=10)
        info_frame.pack(fill=tk.BOTH, pady=(0, 10))
        
        # Information tabs
        self.info_tabs = ttk.Notebook(info_frame)
        self.info_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: About Deepfakes
        about_frame = ttk.Frame(self.info_tabs, padding=10)
        self.info_tabs.add(about_frame, text="About Deepfakes")
        
        about_text = scrolledtext.ScrolledText(about_frame, height=8, wrap=tk.WORD)
        about_text.pack(fill=tk.BOTH, expand=True)
        about_text.insert(tk.END, 
            "Deepfakes are synthetic media where a person's likeness is replaced with someone else's using "
            "artificial intelligence techniques. This technology uses deep learning models to analyze "
            "facial movements and features, then generates new media that appears to show someone saying "
            "or doing something they never did.\n\n"
            "While deepfakes have legitimate uses in entertainment, education, and art, they also pose "
            "significant security and privacy risks when misused for disinformation, fraud, or harassment.\n\n"
            "Understanding how to detect deepfakes is increasingly important for media literacy and security."
        )
        about_text.config(state=tk.DISABLED)
        
        # Tab 2: Detection Methods
        detection_frame = ttk.Frame(self.info_tabs, padding=10)
        self.info_tabs.add(detection_frame, text="Detection Methods")
        
        detection_text = scrolledtext.ScrolledText(detection_frame, height=8, wrap=tk.WORD)
        detection_text.pack(fill=tk.BOTH, expand=True)
        detection_text.insert(tk.END, 
            "Deepfake detection relies on identifying inconsistencies and artifacts that aren't easily "
            "visible to the human eye. Common detection methods include:\n\n"
            "• Biological signals: Analyzing blinking patterns, pulse detection, and micro-expressions\n"
            "• Facial feature analysis: Examining inconsistencies in facial proportions and movements\n"
            "• Texture and noise patterns: Looking for inconsistent noise patterns across the image\n"
            "• Temporal coherence: Identifying flickering or inconsistencies between video frames\n"
            "• AI-based detection: Using neural networks trained to recognize manipulation patterns\n\n"
            "No single detection method is foolproof, and deepfake technology continues to improve."
        )
        detection_text.config(state=tk.DISABLED)
        
        # Tab 3: Security Implications
        security_frame = ttk.Frame(self.info_tabs, padding=10)
        self.info_tabs.add(security_frame, text="Security Implications")
        
        security_text = scrolledtext.ScrolledText(security_frame, height=8, wrap=tk.WORD)
        security_text.pack(fill=tk.BOTH, expand=True)
        security_text.insert(tk.END, 
            "Deepfakes present several security and privacy challenges:\n\n"
            "• Identity fraud: Using synthetic media to impersonate others for financial fraud\n"
            "• Disinformation: Creating false events or statements to manipulate public opinion\n"
            "• Social engineering: Enhancing phishing and other attacks with realistic synthetic media\n"
            "• Privacy violations: Creating non-consensual synthetic imagery of individuals\n"
            "• Trust erosion: Undermining trust in authentic media ("liar's dividend")\n\n"
            "Protective measures include multi-factor authentication, verification protocols, "
            "media authentication technology, and critical media literacy education."
        )
        security_text.config(state=tk.DISABLED)
        
        # Bottom frame for buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Learn more button
        learn_more_button = ttk.Button(
            bottom_frame,
            text="Learn More About Deepfakes",
            command=self.show_learn_more
        )
        learn_more_button.pack(side=tk.LEFT)
        
        # Save results button
        self.save_button = ttk.Button(
            bottom_frame,
            text="Save Analysis Report",
            command=self.save_results,
            state=tk.DISABLED
        )
        self.save_button.pack(side=tk.RIGHT)
        
        # Initialize model in background
        self.init_model()
    
    def init_model(self):
        """Initialize the framework and model in the background"""
        self.status_var.set("Initializing detection model... Please wait.")
        
        # Use a separate thread in a real implementation
        # For simplicity in this demo, we're doing it directly
        try:
            # Generate sample data
            self.framework.load_sample_data(generate_samples=True)
            
            # Train a simple detection model
            self.framework.train_detection_model()
            
            self.model_ready = True
            self.status_var.set("Model ready. Select an image to analyze.")
        except Exception as e:
            self.status_var.set(f"Error initializing model: {str(e)}")
            messagebox.showerror("Model Error", f"Failed to initialize detection model: {str(e)}")
    
    def select_image(self):
        """Open file dialog to select an image"""
        filetypes = (
            ('Image files', '*.jpg *.jpeg *.png *.bmp'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Select an image to analyze',
            initialdir='/',
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Load and display the image
                self.load_image(filename)
                self.status_var.set(f"Image loaded: {os.path.basename(filename)}")
                self.analyze_button.config(state=tk.NORMAL)
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
                messagebox.showerror("Image Error", f"Failed to load image: {str(e)}")
    
    def use_sample_image(self):
        """Use a sample image for demonstration"""
        # In a real implementation, you would include sample images with the application
        # For this demo, we'll generate a synthetic sample
        
        self.status_var.set("Generating sample image for demonstration...")
        
        # Create a sample directory if it doesn't exist
        sample_dir = os.path.join(self.output_dir, "sample_images")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Generate a simple synthetic sample (this would be a real image in a full implementation)
        sample_path = os.path.join(sample_dir, f"sample_{int(time.time())}.jpg")
        
        # Create a simple synthetic image
        try:
            # Create a blank image
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            
            # Add some color and patterns
            for i in range(400):
                for j in range(400):
                    # Create a pattern that looks somewhat like a face
                    if (i-200)**2 + (j-200)**2 < 15000:  # Circle for face
                        img[i, j] = [220, 180, 160]
                    
                    # Eyes
                    if ((i-150)**2 + (j-160)**2 < 100) or ((i-150)**2 + (j-240)**2 < 100):
                        img[i, j] = [255, 255, 255]
                    if ((i-150)**2 + (j-160)**2 < 30) or ((i-150)**2 + (j-240)**2 < 30):
                        img[i, j] = [0, 0, 0]
                    
                    # Mouth
                    if (i-250)**2/400 + (j-200)**2/6400 < 1 and i > 250:
                        img[i, j] = [200, 100, 100]
            
            # Add some noise to make it look less perfect
            noise = np.random.normal(0, 10, (400, 400, 3)).astype(np.uint8)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Save the image
            cv2.imwrite(sample_path, img)
            
            # Load the sample image
            self.load_image(sample_path)
            self.status_var.set(f"Sample image loaded for demonstration")
            self.analyze_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_var.set(f"Error creating sample image: {str(e)}")
            messagebox.showerror("Sample Error", f"Failed to create sample image: {str(e)}")
    
    def load_image(self, image_path):
        """Load and display an image in the UI"""
        self.selected_image = image_path
        
        # Load image and resize for display
        img = Image.open(image_path)
        
        # Calculate new dimensions while preserving aspect ratio
        max_size = (400, 300)
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        photo = ImageTk.PhotoImage(img)
        
        # Update image display
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
    
    def analyze_image(self):
        """Analyze the selected image for deepfake detection"""
        if not self.selected_image:
            messagebox.showwarning("No Image", "Please select an image to analyze first.")
            return
        
        if not self.model_ready:
            messagebox.showwarning("Model Not Ready", "The detection model is still initializing. Please wait.")
            return
        
        self.status_var.set("Analyzing image... Please wait.")
        self.analyze_button.config(state=tk.DISABLED)
        
        try:
            # Run detection using the framework
            predicted_label, confidence = self.framework.demonstrate_detection(self.selected_image)
            
            # Get additional info for educational purposes
            features = self._extract_educational_features()
            
            # Update results display
            self._update_results_display(predicted_label, confidence, features)
            
            self.status_var.set("Analysis complete.")
            self.analyze_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_var.set(f"Error during analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"Failed to analyze image: {str(e)}")
            self.analyze_button.config(state=tk.NORMAL)
    
    def _extract_educational_features(self):
        """
        Extract/generate educational features for demonstration
        
        In a real implementation, this would extract actual features
        from the image. For this demo, we'll generate synthetic features
        to illustrate the concept.
        """
        # Generate synthetic features for educational purposes
        features = {
            'texture_consistency': random.uniform(0.3, 0.9),
            'eye_blinking': random.uniform(0.3, 0.9),
            'facial_boundary': random.uniform(0.3, 0.9),
            'color_consistency': random.uniform(0.3, 0.9),
            'lighting_consistency': random.uniform(0.3, 0.9),
            'noise_pattern': random.uniform(0.3, 0.9),
            'compression_artifacts': random.uniform(0.3, 0.9),
            'background_foreground': random.uniform(0.3, 0.9)
        }
        
        # Save features for reporting
        self.results = {
            'filename': os.path.basename(self.selected_image),
            'path': self.selected_image,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': features
        }
        
        return features
    
    def _update_results_display(self, predicted_label, confidence, features):
        """Update the results display with detection information"""
        # Update result label
        if predicted_label == "Real":
            self.result_var.set(f"Result: Likely AUTHENTIC ({confidence:.1%} confidence)")
            self.result_label.config(foreground="green")
        else:
            self.result_var.set(f"Result: Potential DEEPFAKE ({confidence:.1%} confidence)")
            self.result_label.config(foreground="red")
        
        # Update detailed results text
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, f"DEEPFAKE DETECTION ANALYSIS\n")
        self.results_text.insert(tk.END, f"{'='*40}\n\n")
        
        self.results_text.insert(tk.END, f"Image: {os.path.basename(self.selected_image)}\n")
        self.results_text.insert(tk.END, f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        self.results_text.insert(tk.END, f"DETECTION RESULT: ")
        if predicted_label == "Real":
            self.results_text.insert(tk.END, f"Likely AUTHENTIC\n")
        else:
            self.results_text.insert(tk.END, f"Potential DEEPFAKE\n")
        
        self.results_text.insert(tk.END, f"Confidence: {confidence:.1%}\n\n")
        
        self.results_text.insert(tk.END, f"FEATURE ANALYSIS:\n")
        self.results_text.insert(tk.END, f"{'-'*20}\n")
        
        # Add feature explanations
        for feature, value in features.items():
            feature_name = feature.replace('_', ' ').title()
            self.results_text.insert(tk.END, f"• {feature_name}: ")
            
            # Color code based on value
            if value > 0.7:
                self.results_text.insert(tk.END, f"Strong ({value:.2f})\n")
            elif value > 0.4:
                self.results_text.insert(tk.END, f"Moderate ({value:.2f})\n")
            else:
                self.results_text.insert(tk.END, f"Weak ({value:.2f})\n")
        
        self.results_text.insert(tk.END, f"\nINTERPRETATION:\n")
        self.results_text.insert(tk.END, f"{'-'*20}\n")
        
        if predicted_label == "Real":
            self.results_text.insert(tk.END, 
                "This image shows strong indicators of authenticity, particularly in "
                "texture consistency and facial feature relationships. While no detection "
                "system is perfect, the analysis suggests this is likely an authentic image.\n\n"
            )
            
            # Add explanation of top features
            top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
            self.results_text.insert(tk.END, "Strongest authenticity indicators:\n")
            for feature, value in top_features:
                feature_name = feature.replace('_', ' ').title()
                self.results_text.insert(tk.END, f"• {feature_name} ({value:.2f})\n")
        else:
            self.results_text.insert(tk.END, 
                "This image shows several indicators commonly associated with synthetic "
                "or manipulated media. Particularly concerning are the inconsistencies "
                "in texture and lighting patterns. While this analysis is not definitive "
                "proof, caution is warranted.\n\n"
            )
            
            # Add explanation of bottom features
            bottom_features = sorted(features.items(), key=lambda x: x[1])[:3]
            self.results_text.insert(tk.END, "Strongest manipulation indicators:\n")
            for feature, value in bottom_features:
                feature_name = feature.replace('_', ' ').title()
                self.results_text.insert(tk.END, f"• {feature_name} ({value:.2f})\n")
        
        self.results_text.insert(tk.END, f"\nEDUCATIONAL NOTE:\n")
        self.results_text.insert(tk.END, f"{'-'*20}\n")
        self.results_text.insert(tk.END, 
            "This analysis is provided for educational purposes only. "
            "Real deepfake detection systems use more sophisticated techniques "
            "and require careful validation. No detection system is perfect, and "
            "both false positives and false negatives are possible.\n\n"
            "Always verify important media through multiple sources and context."
        )
        
        self.results_text.config(state=tk.DISABLED)
        
        # Save the complete results
        self.results['prediction'] = predicted_label
        self.results['confidence'] = confidence
    
    def save_results(self):
        """Save the analysis results to a file"""
        if not self.results:
            messagebox.showwarning("No Results", "No analysis results to save.")
            return
        
        # Create reports directory
        reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create filename based on original image and timestamp
        base_filename = os.path.splitext(os.path.basename(self.selected_image))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{base_filename}_analysis_{timestamp}.txt"
        report_path = os.path.join(reports_dir, report_filename)
        
        try:
            with open(report_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("DEEPFAKE DETECTION ANALYSIS REPORT\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"Image: {self.results['filename']}\n")
                f.write(f"Analysis time: {self.results['analysis_time']}\n\n")
                
                f.write(f"DETECTION RESULT: ")
                if self.results['prediction'] == "Real":
                    f.write(f"Likely AUTHENTIC\n")
                else:
                    f.write(f"Potential DEEPFAKE\n")
                
                f.write(f"Confidence: {self.results['confidence']:.1%}\n\n")
                
                f.write(f"FEATURE ANALYSIS:\n")
                f.write(f"{'-'*20}\n")
                
                for feature, value in self.results['features'].items():
                    feature_name = feature.replace('_', ' ').title()
                    f.write(f"• {feature_name}: {value:.2f}\n")
                
                f.write("\nEDUCATIONAL INFORMATION:\n")
                f.write(f"{'-'*20}\n")
                f.write(
                    "This analysis is provided for educational purposes only. It demonstrates "
                    "the concepts and techniques used in deepfake detection. Real deepfake "
                    "detection systems use more sophisticated techniques and require careful "
                    "validation.\n\n"
                    "Common deepfake artifacts include:\n"
                    "• Inconsistent eye blinking or eye reflections\n"
                    "• Unnatural facial boundaries or blending issues\n"
                    "• Inconsistent lighting or shadows across the face\n"
                    "• Texture inconsistencies in skin or hair\n"
                    "• Temporal artifacts in videos (flickering, jittering)\n\n"
                    "No detection system is perfect, and the field continues to evolve as "
                    "deepfake technology improves.\n\n"
                )
                
                f.write("=" * 70 + "\n")
                f.write("EDUCATIONAL DEMO - CREATED FOR CYBOK MATERIALS\n")
                f.write("=" * 70 + "\n")
            
            self.status_var.set(f"Report saved to: {report_path}")
            messagebox.showinfo("Report Saved", f"Analysis report saved to:\n{report_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving report: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save report: {str(e)}")
    
    def show_learn_more(self):
        """Show additional educational information about deepfakes"""
        learn_more_window = tk.Toplevel(self.root)
        learn_more_window.title("Learn More About Deepfakes")
        learn_more_window.geometry("800x600")
        learn_more_window.minsize(600, 400)
        
        # Content frame
        content_frame = ttk.Frame(learn_more_window, padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            content_frame, 
            text="Understanding Deepfakes: Technology and Implications",
            font=("Arial", 16, "bold")
        ).pack(anchor=tk.W, pady=(0, 20))
        
        # Create notebook with tabs
        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Technology
        tech_frame = ttk.Frame(notebook, padding=10)
        notebook.add(tech_frame, text="Technology")
        
        tech_text = scrolledtext.ScrolledText(tech_frame, wrap=tk.WORD, font=("Arial", 11))
        tech_text.pack(fill=tk.BOTH, expand=True)
        tech_text.insert(tk.END, """DEEPFAKE TECHNOLOGY EXPLAINED

Deepfakes are created using deep learning techniques, particularly:

1. Autoencoders
   These neural networks consist of an encoder that compresses an image into a lower-dimensional
   representation, and a decoder that reconstructs the image from that representation. For deepfakes,
   separate decoders are trained for different faces while sharing a common encoder.

2. Generative Adversarial Networks (GANs)
   GANs consist of two neural networks—a generator and a discriminator—trained in an adversarial process.
   The generator creates synthetic images, while the discriminator tries to distinguish them from real images.
   This competition drives the generator to produce increasingly realistic fakes.

3. Face Swapping Process
   - Face detection and alignment in source and target videos
   - Extraction of facial features and landmarks
   - Training the model on both source and target faces
   - Swapping faces by encoding the target and decoding with the source model
   - Post-processing for seamless blending

Recent advances in deep learning have made these techniques more accessible and effective,
leading to increasingly convincing deepfakes that can be difficult to detect with the naked eye.
""")
        tech_text.config(state=tk.DISABLED)
        
        # Tab 2: Detection
        detect_frame = ttk.Frame(notebook, padding=10)
        notebook.add(detect_frame, text="Detection Techniques")
        
        detect_text = scrolledtext.ScrolledText(detect_frame, wrap=tk.WORD, font=("Arial", 11))
        detect_text.pack(fill=tk.BOTH, expand=True)
        detect_text.insert(tk.END, """DEEPFAKE DETECTION TECHNIQUES

Several approaches are used to detect deepfakes:

1. Visual Artifacts
   - Inconsistent eye blinking or unnatural eye movements
   - Unnatural facial boundaries and blending issues
   - Inconsistent lighting and shadows across the face
   - Unrealistic or missing reflections in eyes
   - Texture inconsistencies in skin, teeth, or hair

2. Biological Signals
   - Pulse detection through subtle color changes in skin
   - Breathing patterns in video
   - Natural micro-expressions that are difficult to synthesize
   - Blood flow patterns visible in high-resolution imagery

3. AI-Based Detection Systems
   - Convolutional Neural Networks (CNNs) trained on real and fake media
   - Recurrent Neural Networks (RNNs) for analyzing temporal inconsistencies in videos
   - Frequency domain analysis to detect GAN-specific artifacts
   - Multi-modal approaches combining visual and audio analysis

4. Metadata and Context Analysis
   - Digital signature verification
   - File metadata examination
   - Source and distribution channel analysis
   - Behavioral and linguistic analysis for voice deepfakes

As deepfake technology improves, detection systems must continuously evolve to keep pace.
Research suggests the most effective approaches combine multiple detection methods.
""")
        detect_text.config(state=tk.DISABLED)
        
        # Tab 3: Security
        security_frame = ttk.Frame(notebook, padding=10)
        notebook.add(security_frame, text="Security Implications")
        
        security_text = scrolledtext.ScrolledText(security_frame, wrap=tk.WORD, font=("Arial", 11))
        security_text.pack(fill=tk.BOTH, expand=True)
        security_text.insert(tk.END, """SECURITY IMPLICATIONS OF DEEPFAKES

Deepfakes present several significant security challenges:

1. Identity and Authentication Threats
   - Bypassing biometric authentication systems
   - Impersonation for social engineering attacks
   - Voice synthesis for vishing (voice phishing) attacks
   - Creation of entirely synthetic identities for fraud

2. Information Integrity Challenges
   - Disinformation campaigns using fabricated statements by public figures
   - Market manipulation through fake announcements
   - Evidence tampering and fabrication
   - The "liar's dividend" where authentic content can be dismissed as fake

3. Organizational Vulnerabilities
   - Business email compromise enhanced with voice or video synthesis
   - Executive impersonation for fraudulent authorizations
   - Brand damage through synthetic misrepresentation
   - Industrial espionage and competitive intelligence gathering

4. Privacy Violations
   - Non-consensual intimate imagery
   - Harassment and bullying using fabricated content
   - Reputation damage through synthetic misrepresentation
   - Long-term identity security concerns

5. Protective Strategies
   - Multi-factor, multi-channel authentication for sensitive requests
   - Verification protocols for important communications
   - Digital content provenance and authentication solutions
   - Media literacy education for staff and the public
   - Incident response planning specific to deepfake scenarios

Organizations and individuals must develop a comprehensive approach to address these threats,
combining technical, procedural, and educational measures.
""")
        security_text.config(state=tk.DISABLED)
        
        # Tab 4: Ethics & Law
        ethics_frame = ttk.Frame(notebook, padding=10)
        notebook.add(ethics_frame, text="Ethics & Law")
        
        ethics_text = scrolledtext.ScrolledText(ethics_frame, wrap=tk.WORD, font=("Arial", 11))
        ethics_text.pack(fill=tk.BOTH, expand=True)
        ethics_text.insert(tk.END, """ETHICAL AND LEGAL CONSIDERATIONS

The emergence of deepfake technology raises important ethical and legal questions:

1. Legal Frameworks
   - Several jurisdictions have enacted or proposed legislation specifically addressing deepfakes
   - Laws typically focus on non-consensual intimate imagery, election interference, and fraud
   - Challenges include jurisdiction, attribution, and balancing regulation with free expression
   - Existing laws on defamation, fraud, and harassment may apply but with limitations

2. Ethical Considerations
   - Consent: Should explicit consent be required from individuals depicted in synthetic media?
   - Transparency: Should synthetic media be clearly labeled as such?
   - Harm prevention: How to balance creative uses with potential for harm?
   - Responsibility: Who bears responsibility for misuse - creators, platforms, or distributors?

3. Platform Policies
   - Major platforms have developed policies specifically addressing synthetic media
   - Approaches range from outright bans to labeling requirements
   - Content moderation challenges include detection limitations and scale
   - Transparency in enforcement and appeals processes varies significantly

4. Responsible Development
   - Watermarking and fingerprinting techniques for synthetic media identification
   - Research community efforts to develop ethical guidelines
   - Dual-use considerations in publishing new synthetic media techniques
   - Industry self-regulation initiatives and standards development

5. Media Literacy
   - Educational initiatives to help the public critically evaluate digital media
   - Tools and resources to verify content authenticity
   - Addressing the broader crisis of trust in digital information
   - Balancing critical thinking with avoiding blanket skepticism

Addressing these issues requires collaboration between technologists, legal experts,
policymakers, platforms, and civil society organizations.
""")
        ethics_text.config(state=tk.DISABLED)
        
        # Bottom button frame
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Close button
        close_button = ttk.Button(
            button_frame,
            text="Close",
            command=learn_more_window.destroy
        )
        close_button.pack(side=tk.RIGHT)
        
        # Educational note
        note_label = ttk.Label(
            button_frame,
            text="This information is provided for educational purposes only.",
            font=("Arial", 10, "italic")
        )
        note_label.pack(side=tk.LEFT)

def main():
    """Main function to run the Deepfake Detection Demo"""
    parser = argparse.ArgumentParser(
        description="Deepfake Detection Educational Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output-dir', type=str, default='deepfake_demo_output', 
                        help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Create and run the demo application
    app = DeepfakeDetectionDemo(output_dir=args.output_dir)
    app.root.mainloop()

if __name__ == "__main__":
    main()