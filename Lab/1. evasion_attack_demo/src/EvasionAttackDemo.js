import React, { useState, useEffect, useRef } from 'react';
import './EvasionAttackDemo.css';

const EvasionAttackDemo = () => {
  // States for managing the demo
  const [epsilon, setEpsilon] = useState(0.1);
  const [targetClass, setTargetClass] = useState(null);
  const [sourceClass, setSourceClass] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [perturbedImage, setPerturbedImage] = useState(null);
  const [isAttacking, setIsAttacking] = useState(false);
  const [attackComplete, setAttackComplete] = useState(false);
  const [attackSuccess, setAttackSuccess] = useState(false);
  const [confidenceOriginal, setConfidenceOriginal] = useState(0);
  const [confidencePerturbed, setConfidencePerturbed] = useState(0);
  const [attackSteps, setAttackSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  
  // Available sample images
  const [sampleImages, setSampleImages] = useState([]);
  
  // Refs for canvas elements
  const originalCanvasRef = useRef(null);
  const perturbedCanvasRef = useRef(null);
  const diffCanvasRef = useRef(null);
  const stepCanvasRef = useRef(null);
  
  // Initialize the demo
  useEffect(() => {
    // Generate sample digit images
    const generatedSamples = generateSampleDigits();
    setSampleImages(generatedSamples);
    
    // Set default selected image
    if (generatedSamples.length > 0) {
      setSelectedImage(generatedSamples[0]);
      setSourceClass(generatedSamples[0].label);
      
      // Find a reasonable target class (not the source class)
      const targetOptions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].filter(
        num => num !== generatedSamples[0].label
      );
      setTargetClass(targetOptions[0]);
    }
  }, []);
  
  // When selected image changes, draw it on the canvas
  useEffect(() => {
    if (selectedImage && originalCanvasRef.current) {
      drawDigitToCanvas(selectedImage.data, originalCanvasRef.current);
      // Reset attack state when image changes
      setPerturbedImage(null);
      setAttackComplete(false);
      setAttackSuccess(false);
      setAttackSteps([]);
      setCurrentStep(0);
      
      // Simulate a classification with high confidence
      setConfidenceOriginal(0.98);
      
      // Clear other canvases
      clearCanvas(perturbedCanvasRef.current);
      clearCanvas(diffCanvasRef.current);
      clearCanvas(stepCanvasRef.current);
    }
  }, [selectedImage]);
  
  // Generate synthetic digit images
  const generateSampleDigits = () => {
    const samples = [];
    const pixelSize = 28; // MNIST size
    
    // Simple templates for digits 0-9
    const digitTemplates = [
      // 0
      [
        [0,0,1,1,1,1,0,0],
        [0,1,0,0,0,0,1,0],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [0,1,0,0,0,0,1,0],
        [0,0,1,1,1,1,0,0]
      ],
      // 1
      [
        [0,0,0,1,0,0,0,0],
        [0,0,1,1,0,0,0,0],
        [0,1,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,1,1,1,1,1,0,0]
      ],
      // 2
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,0]
      ],
      // 3
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ],
      // 4
      [
        [0,0,0,0,1,0,0,0],
        [0,0,0,1,1,0,0,0],
        [0,0,1,0,1,0,0,0],
        [0,1,0,0,1,0,0,0],
        [1,0,0,0,1,0,0,0],
        [1,1,1,1,1,1,1,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0,0]
      ],
      // 5
      [
        [0,1,1,1,1,1,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ],
      // 6
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ],
      // 7
      [
        [0,1,1,1,1,1,1,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0]
      ],
      // 8
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ],
      // 9
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ]
    ];
    
    // Create a scaled version of each digit template
    for (let digitLabel = 0; digitLabel < 10; digitLabel++) {
      const template = digitTemplates[digitLabel];
      const digitData = new Array(pixelSize * pixelSize).fill(0);
      
      // Scale the template to full image size
      const scale = pixelSize / template.length;
      
      for (let y = 0; y < pixelSize; y++) {
        for (let x = 0; x < pixelSize; x++) {
          const templateX = Math.floor(x / scale);
          const templateY = Math.floor(y / scale);
          
          let value = 0;
          if (templateY < template.length && templateX < template[0].length) {
            value = template[templateY][templateX];
          }
          
          // Add some noise for realism
          value = value > 0 ? 0.6 + Math.random() * 0.4 : Math.random() * 0.1;
          
          digitData[y * pixelSize + x] = value;
        }
      }
      
      samples.push({
        id: digitLabel,
        label: digitLabel,
        data: digitData,
        size: pixelSize
      });
    }
    
    return samples;
  };
  
  // Draw a digit to a canvas
  const drawDigitToCanvas = (pixelData, canvas, heatmap = false) => {
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const pixelSize = Math.sqrt(pixelData.length);
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw each pixel
    const pixelWidth = width / pixelSize;
    const pixelHeight = height / pixelSize;
    
    for (let y = 0; y < pixelSize; y++) {
      for (let x = 0; x < pixelSize; x++) {
        const pixelIndex = y * pixelSize + x;
        const pixelValue = pixelData[pixelIndex];
        
        if (heatmap) {
          // Use a heatmap color scheme for difference visualization
          // Red for positive values, blue for negative values
          if (pixelValue > 0) {
            // Red for positive differences (added pixels)
            const intensity = Math.min(255, Math.round(pixelValue * 255));
            ctx.fillStyle = `rgb(${intensity}, 0, 0)`;
          } else if (pixelValue < 0) {
            // Blue for negative differences (removed pixels)
            const intensity = Math.min(255, Math.round(Math.abs(pixelValue) * 255));
            ctx.fillStyle = `rgb(0, 0, ${intensity})`;
          } else {
            // White for unchanged pixels
            ctx.fillStyle = 'rgb(255, 255, 255)';
          }
        } else {
          // Normal grayscale rendering
          const intensity = Math.round(pixelValue * 255);
          ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
        }
        
        ctx.fillRect(
          x * pixelWidth, 
          y * pixelHeight, 
          pixelWidth, 
          pixelHeight
        );
      }
    }
  };
  
  // Clear a canvas
  const clearCanvas = (canvas) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };
  
  // Start the evasion attack
  const startAttack = () => {
    if (isAttacking || !selectedImage) return;
    
    setIsAttacking(true);
    setAttackComplete(false);
    setAttackSteps([]);
    setCurrentStep(0);
    
    // Begin with the original image
    const originalPixels = [...selectedImage.data];
    const size = Math.sqrt(originalPixels.length);
    
    // Prepare an array for all attack steps
    const steps = [];
    
    // Generate gradient information (simulated)
    // In a real attack, this would come from model gradients
    const gradient = simulateGradient(originalPixels, sourceClass, targetClass);
    
    // Simulate the attack steps
    let currentImg = [...originalPixels];
    const numSteps = 10;
    
    for (let step = 1; step <= numSteps; step++) {
      // Create a perturbation based on the gradient and epsilon value
      const stepPerturbation = gradient.map(g => (g * epsilon * step) / numSteps);
      
      // Apply perturbation
      const perturbedImg = originalPixels.map((pixel, i) => {
        return Math.max(0, Math.min(1, pixel + stepPerturbation[i]));
      });
      
      // Calculate the difference between original and perturbed image
      const diffImg = perturbedImg.map((pixel, i) => pixel - originalPixels[i]);
      
      // Calculate simulated model confidence
      const originalConfidence = Math.max(0, 0.98 - (0.05 * step / numSteps));
      const targetConfidence = Math.min(0.95, 0.1 + (0.85 * step / numSteps));
      
      // Store this step
      steps.push({
        step,
        image: [...perturbedImg],
        difference: diffImg,
        confidences: {
          [sourceClass]: originalConfidence,
          [targetClass]: targetConfidence
        }
      });
      
      currentImg = perturbedImg;
    }
    
    // Save all steps
    setAttackSteps(steps);
    
    // Start animation of the attack progression
    animateAttackSteps(steps);
  };
  
  // Simulate gradient for the attack
  const simulateGradient = (originalPixels, sourceClass, targetClass) => {
    const size = Math.sqrt(originalPixels.length);
    const gradient = new Array(originalPixels.length).fill(0);
    
    // Create a simple gradient that targets specific regions of the image
    // In a real attack, this would come from the model's gradients
    
    // Different patterns for different digit transitions
    const sourceMask = createDigitMask(sourceClass, size);
    const targetMask = createDigitMask(targetClass, size);
    
    // Calculate gradient: we want to decrease source pattern and increase target pattern
    for (let i = 0; i < gradient.length; i++) {
      const originalIntensity = originalPixels[i];
      
      // Positive gradient increases pixel value (makes it whiter)
      // Negative gradient decreases pixel value (makes it darker)
      
      if (sourceMask[i] > 0.5 && targetMask[i] < 0.5) {
        // This pixel should be dark in the target digit but is bright in the source
        gradient[i] = -1; // Decrease this pixel (make it darker)
      } else if (sourceMask[i] < 0.5 && targetMask[i] > 0.5) {
        // This pixel should be bright in the target digit but is dark in the source
        gradient[i] = 1; // Increase this pixel (make it brighter)
      } else {
        // This pixel is either the same in both digits or not important
        gradient[i] = 0;
      }
      
      // Scale gradient by the difference from target value
      // (bigger gradient where there's more room for change)
      if (gradient[i] > 0) {
        gradient[i] *= (1 - originalIntensity); // Scale based on room to increase
      } else if (gradient[i] < 0) {
        gradient[i] *= originalIntensity; // Scale based on room to decrease
      }
    }
    
    return gradient;
  };
  
  // Create a mask representing the pattern of a specific digit
  const createDigitMask = (digit, size) => {
    // Simple templates for digits 0-9 (same as in generateSampleDigits)
    const digitTemplates = [
      // 0
      [
        [0,0,1,1,1,1,0,0],
        [0,1,0,0,0,0,1,0],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [0,1,0,0,0,0,1,0],
        [0,0,1,1,1,1,0,0]
      ],
      // 1
      [
        [0,0,0,1,0,0,0,0],
        [0,0,1,1,0,0,0,0],
        [0,1,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,1,1,1,1,1,0,0]
      ],
      // 2
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,0]
      ],
      // 3
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ],
      // 4
      [
        [0,0,0,0,1,0,0,0],
        [0,0,0,1,1,0,0,0],
        [0,0,1,0,1,0,0,0],
        [0,1,0,0,1,0,0,0],
        [1,0,0,0,1,0,0,0],
        [1,1,1,1,1,1,1,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0,0]
      ],
      // 5
      [
        [0,1,1,1,1,1,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ],
      // 6
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ],
      // 7
      [
        [0,1,1,1,1,1,1,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0]
      ],
      // 8
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ],
      // 9
      [
        [0,0,1,1,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,1,1,0,0,0]
      ]
    ];
    
    const template = digitTemplates[digit];
    const mask = new Array(size * size).fill(0);
    
    // Scale the template to full image size
    const scale = size / template.length;
    
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const templateX = Math.floor(x / scale);
        const templateY = Math.floor(y / scale);
        
        if (templateY < template.length && templateX < template[0].length) {
          mask[y * size + x] = template[templateY][templateX];
        }
      }
    }
    
    return mask;
  };
  
  // Animate the attack steps
  const animateAttackSteps = (steps) => {
    let stepIndex = 0;
    
    const animateStep = () => {
      if (stepIndex >= steps.length) {
        // Animation complete
        setIsAttacking(false);
        setAttackComplete(true);
        setAttackSuccess(true);
        
        // Set final perturbation
        const finalStep = steps[steps.length - 1];
        setPerturbedImage(finalStep.image);
        
        // Update confidence
        setConfidenceOriginal(finalStep.confidences[sourceClass]);
        setConfidencePerturbed(finalStep.confidences[targetClass]);
        
        return;
      }
      
      // Update to current step
      const step = steps[stepIndex];
      setCurrentStep(stepIndex + 1);
      
      // Draw current perturbation
      drawDigitToCanvas(step.image, perturbedCanvasRef.current);
      
      // Draw difference
      drawDigitToCanvas(step.difference, diffCanvasRef.current, true);
      
      // Update confidence values
      setConfidenceOriginal(step.confidences[sourceClass]);
      setConfidencePerturbed(step.confidences[targetClass]);
      
      // Move to next step
      stepIndex++;
      setTimeout(animateStep, 300);
    };
    
    // Start animation
    animateStep();
  };
  
  // Reset the demo
  const resetDemo = () => {
    setIsAttacking(false);
    setAttackComplete(false);
    setAttackSuccess(false);
    setPerturbedImage(null);
    setAttackSteps([]);
    setCurrentStep(0);
    setConfidenceOriginal(0.98);
    setConfidencePerturbed(0);
    
    // Clear canvases
    clearCanvas(perturbedCanvasRef.current);
    clearCanvas(diffCanvasRef.current);
    clearCanvas(stepCanvasRef.current);
  };
  
  return (
    <div className="evasion-demo-container">
      <h1 className="evasion-demo-title">Adversarial Evasion Attack Visualization</h1>
      
      <div className="evasion-info-box">
        <h2>What is an Evasion Attack?</h2>
        <p>
          Evasion attacks occur when an adversary manipulates input data to mislead a machine learning model during 
          inference. These attacks are also known as "test-time" attacks because they happen after the model has been trained.
          The adversary applies carefully calculated perturbations to the input that cause the model to make an incorrect prediction,
          while keeping the changes imperceptible to human observers.
        </p>
      </div>
      
      <div className="evasion-config-box">
        <h2>Attack Configuration</h2>
        
        <div className="evasion-config-grid">
          <div className="evasion-control-group">
            <label>Perturbation Strength (ε):</label>
            <div className="evasion-slider-container">
              <input 
                type="range" 
                min="0.05" 
                max="0.3" 
                step="0.01"
                value={epsilon} 
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                disabled={isAttacking || attackComplete}
              />
              <span>{epsilon.toFixed(2)}</span>
            </div>
            <p className="evasion-explainer">
              Controls how much the input can be modified (higher = more visible changes)
            </p>
          </div>
          
          <div className="evasion-control-group">
            <label>Source Digit:</label>
            <select 
              value={sourceClass || ""}
              onChange={(e) => setSourceClass(parseInt(e.target.value))}
              disabled={isAttacking || attackComplete}
              className="evasion-select"
            >
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => (
                <option key={`source-${num}`} value={num}>{num}</option>
              ))}
            </select>
            <p className="evasion-explainer">
              The digit to attack (the starting image)
            </p>
          </div>
          
          <div className="evasion-control-group">
            <label>Target Digit:</label>
            <select 
              value={targetClass || ""}
              onChange={(e) => setTargetClass(parseInt(e.target.value))}
              disabled={isAttacking || attackComplete}
              className="evasion-select"
            >
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].filter(num => num !== sourceClass).map(num => (
                <option key={`target-${num}`} value={num}>{num}</option>
              ))}
            </select>
            <p className="evasion-explainer">
              The digit we want the model to predict
            </p>
          </div>
          
          <div className="evasion-button-group">
            <button 
              onClick={startAttack}
              disabled={isAttacking || attackComplete || !selectedImage}
              className="evasion-start-button"
            >
              {isAttacking ? "Attacking..." : "Start Attack"}
            </button>
            <button 
              onClick={resetDemo}
              disabled={isAttacking || !attackComplete}
              className="evasion-reset-button"
            >
              Reset Demo
            </button>
          </div>
        </div>
      </div>
      
      <div className="evasion-sample-selection">
        <h3>Select a Digit:</h3>
        <div className="evasion-samples-grid">
          {sampleImages.map((sample, index) => (
            <div 
              key={index}
              className={`evasion-sample-thumbnail ${selectedImage && selectedImage.id === sample.id ? 'selected' : ''}`}
              onClick={() => {
                if (!isAttacking && !attackComplete) {
                  setSelectedImage(sample);
                  setSourceClass(sample.label);
                  
                  // Update target class to be different from source
                  if (targetClass === sample.label) {
                    const newTarget = (sample.label + 1) % 10;
                    setTargetClass(newTarget);
                  }
                }
              }}
            >
              <canvas 
                width={50} 
                height={50}
                ref={canvas => {
                  if (canvas) {
                    drawDigitToCanvas(sample.data, canvas);
                  }
                }}
              />
              <div className="evasion-sample-label">{sample.label}</div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="evasion-visualization">
        <div className="evasion-viz-row">
          <div className="evasion-viz-cell">
            <h3>Original Image (Class {sourceClass})</h3>
            <div className="evasion-canvas-container">
              <canvas 
                ref={originalCanvasRef} 
                width={200} 
                height={200} 
                className="evasion-canvas"
              />
            </div>
            <div className="evasion-confidence-bar">
              <div className="evasion-confidence-label">Confidence:</div>
              <div className="evasion-confidence-track">
                <div 
                  className="evasion-confidence-fill" 
                  style={{width: `${confidenceOriginal * 100}%`}}
                ></div>
              </div>
              <div className="evasion-confidence-value">{(confidenceOriginal * 100).toFixed(0)}%</div>
            </div>
          </div>
          
          <div className="evasion-viz-cell">
            <h3>Adversarial Image (Target: {targetClass})</h3>
            <div className="evasion-canvas-container">
              <canvas 
                ref={perturbedCanvasRef} 
                width={200} 
                height={200} 
                className="evasion-canvas"
              />
              {isAttacking && (
                <div className="evasion-overlay">
                  <div className="evasion-loading">Generating adversarial example...</div>
                </div>
              )}
              {!isAttacking && !perturbedImage && (
                <div className="evasion-overlay">
                  <div className="evasion-placeholder">Start attack to generate</div>
                </div>
              )}
            </div>
            <div className="evasion-confidence-bar">
              <div className="evasion-confidence-label">Confidence:</div>
              <div className="evasion-confidence-track">
                <div 
                  className="evasion-confidence-fill" 
                  style={{width: `${confidencePerturbed * 100}%`}}
                ></div>
              </div>
              <div className="evasion-confidence-value">{(confidencePerturbed * 100).toFixed(0)}%</div>
            </div>
          </div>
          
          <div className="evasion-viz-cell">
            <h3>Perturbation (Difference)</h3>
            <div className="evasion-canvas-container">
              <canvas 
                ref={diffCanvasRef} 
                width={200} 
                height={200} 
                className="evasion-canvas"
              />
              {!isAttacking && !perturbedImage && (
                <div className="evasion-overlay">
                  <div className="evasion-placeholder">Will show difference</div>
                </div>
              )}
            </div>
            <div className="evasion-legend">
              <div className="evasion-legend-item">
                <div className="evasion-legend-color add"></div>
                <span>Added pixels</span>
              </div>
              <div className="evasion-legend-item">
                <div className="evasion-legend-color remove"></div>
                <span>Removed pixels</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="evasion-progress">
          <h3>Attack Progress: Step {currentStep}/{attackSteps.length}</h3>
          <div className="evasion-progress-bar">
            <div 
              className="evasion-progress-fill" 
              style={{
                width: attackSteps.length > 0 
                  ? `${(currentStep / attackSteps.length) * 100}%` 
                  : '0%'
              }}
            ></div>
          </div>
        </div>
      </div>
      
      <div className="evasion-info-note">
        <h3>How Evasion Attacks Work</h3>
        <p>
          This demonstration shows a targeted evasion attack that causes a model to misclassify a digit as another 
          specific digit. The attack follows these steps:
        </p>
        <ol>
          <li>
            <strong>Gradient Calculation:</strong> The attack calculates how the model's output would change 
            if each pixel were slightly modified. This forms a "gradient map" showing which pixels to modify 
            for maximum effect.
          </li>
          <li>
            <strong>Perturbation Creation:</strong> Using this gradient information, the attack generates a 
            perturbation – a small change to add to the original image.
          </li>
          <li>
            <strong>Optimization:</strong> The perturbation is carefully optimized to minimize visual changes 
            while maximizing the chance of misclassification.
          </li>
        </ol>
        <p>
          In real-world scenarios, these attacks can be used against production ML systems like face recognition, 
          spam filters, or malware detectors. Defenses include adversarial training (training on adversarial examples) 
          and input preprocessing techniques.
        </p>
      </div>
    </div>
  );
};

export default EvasionAttackDemo;