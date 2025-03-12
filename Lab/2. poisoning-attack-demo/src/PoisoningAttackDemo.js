import React, { useState, useEffect, useRef } from 'react';
import './PoisoningAttackDemo.css';

const PoisoningAttackDemo = () => {
  // States for managing the demo
  const [poisonPercent, setPoisonPercent] = useState(10);
  const [targetClass, setTargetClass] = useState(1);
  const [sourceClass, setSourceClass] = useState(0);
  const [isPoisoning, setIsPoisoning] = useState(false);
  const [poisoningComplete, setPoisoningComplete] = useState(false);
  const [trainingData, setTrainingData] = useState([]);
  const [poisonedData, setPoisonedData] = useState([]);
  const [modelAccuracy, setModelAccuracy] = useState({
    before: 95,
    after: 0
  });
  const [attackSuccessRate, setAttackSuccessRate] = useState(0);
  
  // Refs for canvas elements
  const originalDataCanvasRef = useRef(null);
  const poisonedDataCanvasRef = useRef(null);
  const sampleCanvasRefs = useRef([]);
  
  // Initialize the demo
  useEffect(() => {
    // Generate initial training data distribution and store it
    const initialData = generateTrainingData();
    
    // Draw initial data visualizations
    if (originalDataCanvasRef.current) {
      drawDataDistribution(originalDataCanvasRef.current, initialData, false);
    }
    
    if (poisonedDataCanvasRef.current) {
      drawDataDistribution(poisonedDataCanvasRef.current, [], false);
    }
    
    // Initialize sample canvases with placeholders
    sampleCanvasRefs.current.forEach(canvas => {
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw question mark to indicate no prediction
        ctx.fillStyle = '#ddd';
        ctx.font = '24px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('?', canvas.width / 2, canvas.height / 2);
      }
    });
  }, []);
  
  // Generate synthetic training data
  const generateTrainingData = () => {
    const data = [];
    
    // Generate data points for 10 classes (0-9)
    for (let classLabel = 0; classLabel < 10; classLabel++) {
      // Center point for this class
      const centerX = 50 + (classLabel % 5) * 100;
      const centerY = 50 + Math.floor(classLabel / 5) * 100;
      
      // Generate 10 samples per class
      for (let i = 0; i < 10; i++) {
        // Add some random noise
        const x = centerX + (Math.random() - 0.5) * 40;
        const y = centerY + (Math.random() - 0.5) * 40;
        
        data.push({
          x, y, label: classLabel,
          isOriginal: true,
          isPoisoned: false
        });
      }
    }
    
    setTrainingData(data);
    return data; // Return data so it can be used immediately
  };
  
  // Draw data distribution on canvas
  const drawDataDistribution = (canvas, data, showPoisoned = false) => {
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid lines
    ctx.strokeStyle = '#e5e5e5';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let x = 0; x <= width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let y = 0; y <= height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // If no data, return
    if (!data || data.length === 0) return;
    
    // Draw data points
    for (const point of data) {
      ctx.beginPath();
      
      // Determine color based on class and whether it's poisoned
      if (point.isPoisoned && showPoisoned) {
        // Make poisoned points clearly visible with red color
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.strokeStyle = 'darkred';
        // Make poisoned points slightly larger
        ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
      } else {
        // Color based on class (using a color scheme)
        const hue = (point.label * 36) % 360;  // Spread colors around the color wheel
        ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.strokeStyle = `hsl(${hue}, 70%, 40%)`;
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
      }
      
      ctx.fill();
      ctx.stroke();
      
      // Add label text
      ctx.fillStyle = 'black';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      // If point is poisoned, show the target class label
      const labelToShow = (point.isPoisoned && showPoisoned) ? targetClass : point.label;
      ctx.fillText(labelToShow, point.x, point.y);
    }
    
    // Add legend if showing poisoned data
    if (showPoisoned) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.fillRect(width - 150, 10, 140, 60);
      ctx.strokeStyle = '#999';
      ctx.strokeRect(width - 150, 10, 140, 60);
      
      // Regular data point
      ctx.beginPath();
      const origHue = (sourceClass * 36) % 360;
      ctx.fillStyle = `hsl(${origHue}, 70%, 60%)`;
      ctx.strokeStyle = `hsl(${origHue}, 70%, 40%)`;
      ctx.arc(width - 130, 30, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Poisoned data point
      ctx.beginPath();
      ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
      ctx.strokeStyle = 'darkred';
      ctx.arc(width - 130, 50, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Legend text
      ctx.fillStyle = 'black';
      ctx.font = '12px Arial';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('Original Class ' + sourceClass, width - 115, 30);
      ctx.fillText('Poisoned → Class ' + targetClass, width - 115, 50);
    }
  };
  
  // Start the poisoning attack
  const startPoisoning = () => {
    if (isPoisoning || poisoningComplete) return;
    
    setIsPoisoning(true);
    setPoisoningComplete(false);
    
    // Create a copy of the training data
    const originalData = [...trainingData];
    
    // Find samples from the source class
    const sourceClassSamples = originalData.filter(p => p.label === sourceClass);
    
    // Calculate how many samples to poison
    const numToPoisonTotal = Math.floor(sourceClassSamples.length * (poisonPercent / 100));
    
    // Randomly select indices to poison
    const sourceIndices = originalData
      .map((point, index) => ({ point, index }))
      .filter(item => item.point.label === sourceClass)
      .map(item => item.index);
    
    const shuffledIndices = [...sourceIndices].sort(() => Math.random() - 0.5);
    const indicesToPoison = shuffledIndices.slice(0, numToPoisonTotal);
    
    // Create poisoned dataset
    const poisoned = originalData.map((point, index) => {
      const newPoint = {...point};
      
      // If this point's index is in the list to poison
      if (indicesToPoison.includes(index)) {
        newPoint.isPoisoned = true;
        newPoint.originalLabel = point.label;
        newPoint.label = targetClass;  // Change label to target class
      }
      
      return newPoint;
    });
    
    // Simulate poisoning process with animation
    const poisoningStep = (step, maxSteps) => {
      // Create a partial dataset to visualize the poisoning process
      const partialPoisoned = originalData.map((point, index) => {
        const newPoint = {...point};
        const poisonedPoint = poisoned[index];
        
        // Only show poisoned points up to current step
        if (poisonedPoint.isPoisoned) {
          const stepThreshold = Math.floor((step / maxSteps) * numToPoisonTotal);
          const poisonIndex = indicesToPoison.indexOf(index);
          
          if (poisonIndex < stepThreshold) {
            newPoint.isPoisoned = true;
            newPoint.originalLabel = point.label;
            newPoint.label = targetClass;
          }
        }
        
        return newPoint;
      });
      
      // Update visualization
      setPoisonedData(partialPoisoned);
      drawDataDistribution(originalDataCanvasRef.current, originalData, false);
      drawDataDistribution(poisonedDataCanvasRef.current, partialPoisoned, true);
      
      // Continue animation if not done
      if (step < maxSteps) {
        setTimeout(() => {
          poisoningStep(step + 1, maxSteps);
        }, 100);
      } else {
        // Animation complete
        setIsPoisoning(false);
        setPoisoningComplete(true);
        
        // Calculate final metrics
        const poisonedAccuracy = 95 - (numToPoisonTotal / originalData.length) * 20;
        setModelAccuracy({
          before: 95,
          after: poisonedAccuracy
        });
        
        // Calculate attack success rate
        // In a real attack, this would measure how often the model misclassifies 
        // inputs from the source class as the target class after poisoning
        const successRate = (numToPoisonTotal / sourceClassSamples.length) * 90;
        setAttackSuccessRate(successRate);
        
        // Update sample visualizations
        updateSampleVisualizations(poisoned);
      }
    };
    
    // Start poisoning animation
    poisoningStep(1, 10);
  };
  
  // Reset the demo
  const resetDemo = () => {
    setIsPoisoning(false);
    setPoisoningComplete(false);
    setPoisonedData([]);
    setAttackSuccessRate(0);
    setModelAccuracy({
      before: 95,
      after: 0
    });
    
    // Regenerate original data
    const newData = generateTrainingData();
    
    // Clear poisoned data visualization
    drawDataDistribution(originalDataCanvasRef.current, newData, false);
    drawDataDistribution(poisonedDataCanvasRef.current, [], false);
    
    // Reset sample visualizations
    sampleCanvasRefs.current.forEach(canvas => {
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw question mark to indicate no prediction
        ctx.fillStyle = '#ddd';
        ctx.font = '24px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('?', canvas.width / 2, canvas.height / 2);
      }
    });
  };
  
    // Update sample visualizations after poisoning
    // Update sample visualizations after poisoning
    const updateSampleVisualizations = (poisonedDataset) => {
    // Update canvas visualizations for all 6 samples
    const samplesToShow = 6;

    for (let i = 0; i < samplesToShow; i++) {
      const canvas = sampleCanvasRefs.current[i];
      if (!canvas) continue;
      
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, width, height);
      
      // Draw digit representation (simplified for demo)
      ctx.beginPath();
      const origHue = (sourceClass * 36) % 360;
      ctx.fillStyle = `hsl(${origHue}, 70%, 80%)`;
      ctx.arc(width / 2, height / 2, 20, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw outline
      ctx.strokeStyle = `hsl(${origHue}, 70%, 40%)`;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw label
      ctx.fillStyle = 'black';
      ctx.font = '24px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(sourceClass, width / 2, height / 2);
      
      // First 3 samples ALWAYS show poisoning effect (target class) if poisoning is complete
      const isPoisonedSample = (i < 3) && poisoningComplete;
      
      ctx.fillStyle = 'black';
      ctx.font = '14px Arial';
      ctx.fillText('Prediction:', width / 2, height - 30);
      
      // Use explicit target and source class values
      ctx.font = 'bold 20px Arial';
      
      if (isPoisonedSample) {
        // Poisoned prediction - show target class in red
        ctx.fillStyle = 'red';
        ctx.fillText(targetClass.toString(), width / 2, height - 10);
      } else {
        // Correct prediction - show source class in green
        ctx.fillStyle = 'green';
        ctx.fillText(sourceClass.toString(), width / 2, height - 10);
      }
    }
    };
  
  
  return (
    <div className="poisoning-demo-container">
      <h1 className="poisoning-demo-title">Adversarial Poisoning Attack Visualization</h1>
      
      <div className="poisoning-info-box">
        <h2>What is a Poisoning Attack?</h2>
        <p>
          Poisoning attacks target machine learning models during the training phase by injecting
          maliciously crafted samples into the training dataset. These poisoned samples are designed
          to manipulate the learning process, causing the model to learn incorrect patterns or create
          specific vulnerabilities.
        </p>
      </div>
      
      <div className="poisoning-config-box">
        <h2>Attack Configuration</h2>
        
        <div className="poisoning-config-grid">
          <div className="poisoning-control-group">
            <label>Poison Percentage:</label>
            <div className="poisoning-slider-container">
              <input 
                type="range" 
                min="1" 
                max="100" 
                value={poisonPercent} 
                onChange={(e) => setPoisonPercent(parseInt(e.target.value))}
                disabled={isPoisoning || poisoningComplete}
              />
              <span>{poisonPercent}%</span>
            </div>
            <p className="poisoning-explainer">
              Percentage of source class to be poisoned
            </p>
          </div>
          
          <div className="poisoning-control-group">
            <label>Source Class:</label>
            <select 
              value={sourceClass}
              onChange={(e) => setSourceClass(parseInt(e.target.value))}
              disabled={isPoisoning || poisoningComplete}
              className="poisoning-select"
            >
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => (
                <option key={`source-${num}`} value={num}>{num}</option>
              ))}
            </select>
            <p className="poisoning-explainer">
              Class that will be poisoned
            </p>
          </div>
          
          <div className="poisoning-control-group">
            <label>Target Class:</label>
            <select 
              value={targetClass}
              onChange={(e) => setTargetClass(parseInt(e.target.value))}
              disabled={isPoisoning || poisoningComplete}
              className="poisoning-select"
            >
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => (
                <option key={`target-${num}`} value={num}>{num}</option>
              ))}
            </select>
            <p className="poisoning-explainer">
              Class that poisoned samples will be labeled as
            </p>
          </div>
          
          <div className="poisoning-button-group">
            <button 
              onClick={startPoisoning}
              disabled={isPoisoning || poisoningComplete}
              className="poisoning-start-button"
            >
              Start Poisoning
            </button>
            <button 
              onClick={resetDemo}
              disabled={isPoisoning || !poisoningComplete}
              className="poisoning-reset-button"
            >
              Reset Demo
            </button>
          </div>
        </div>
      </div>
      
      <div className="poisoning-grid-container">
        <div className="poisoning-box">
          <h2>Original Training Data</h2>
          <div className="poisoning-canvas-container">
            <canvas 
              ref={originalDataCanvasRef} 
              width={500} 
              height={300} 
              className="poisoning-data-canvas"
            />
          </div>
          <p className="poisoning-note">
            The original training dataset contains clean, correctly labeled samples.
            Each point is labeled with its class (0-9).
          </p>
        </div>
        
        <div className="poisoning-box">
          <h2>Poisoned Training Data</h2>
          <div className="poisoning-canvas-container">
            <canvas 
              ref={poisonedDataCanvasRef} 
              width={500} 
              height={300} 
              className="poisoning-data-canvas"
            />
          </div>
          <p className="poisoning-note">
            The poisoned dataset contains targeted samples that have been mislabeled from class {sourceClass} to class {targetClass}.
            Poisoned points are shown in red.
          </p>
        </div>
      </div>
      
      <div className="poisoning-grid-container">
        <div className="poisoning-box">
          <h2>Impact on Model Performance</h2>
          
          <div className="poisoning-metrics">
            <div className="poisoning-metric-card">
              <div className="poisoning-metric-title">Original Accuracy</div>
              <div className="poisoning-metric-value">{modelAccuracy.before.toFixed(1)}%</div>
            </div>
            
            <div className="poisoning-metric-card">
              <div className="poisoning-metric-title">Accuracy After Poisoning</div>
              <div 
                className="poisoning-metric-value"
                style={{color: poisoningComplete ? '#e53e3e' : '#aaa'}}
              >
                {poisoningComplete ? modelAccuracy.after.toFixed(1) : '--'}%
              </div>
            </div>
            
            <div className="poisoning-metric-card">
              <div className="poisoning-metric-title">Attack Success Rate</div>
              <div 
                className="poisoning-metric-value"
                style={{color: poisoningComplete ? '#e53e3e' : '#aaa'}}
              >
                {poisoningComplete ? attackSuccessRate.toFixed(1) : '--'}%
              </div>
              <div className="poisoning-metric-subtitle">
                (Class {sourceClass} → {targetClass})
              </div>
            </div>
          </div>
          
          <p className="poisoning-note">
            Poisoning attacks often cause a small decrease in overall accuracy, but have a targeted effect
            on specific classes. The attack success rate measures how often samples from class {sourceClass} are
            misclassified as class {targetClass} after poisoning.
          </p>
        </div>
        
        <div className="poisoning-box">
          <h2>Model Predictions After Poisoning</h2>
          
          <div className="poisoning-samples-grid">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="poisoning-sample-card">
                <canvas 
                  ref={el => sampleCanvasRefs.current[i] = el} 
                  width={80} 
                  height={80} 
                  className="poisoning-sample-canvas"
                />
              </div>
            ))}
          </div>
          
          <div className="poisoning-legend">
            <div className="poisoning-legend-item">
              <div className="poisoning-legend-color correct"></div>
              <span>Correct Prediction</span>
            </div>
            <div className="poisoning-legend-item">
              <div className="poisoning-legend-color incorrect"></div>
              <span>Poisoned Prediction</span>
            </div>
          </div>
          
          <p className="poisoning-note">
            These samples show how the poisoned model now classifies inputs from class {sourceClass}.
            Notice how some samples are now incorrectly classified as class {targetClass}.
          </p>
        </div>
      </div>
      
      <div className="poisoning-info-note">
        <h3>How Poisoning Attacks Work</h3>
        <p>
          In this demonstration, we're showing a targeted poisoning attack where samples from class {sourceClass} are
          mislabeled as class {targetClass} during training. This causes the model to learn an incorrect decision
          boundary between these classes.
        </p>
        <p>
          In real-world scenarios, poisoning attacks can be much more sophisticated, using optimization techniques
          to determine the most effective samples to poison. Defenses include robust training methods, data sanitization,
          and anomaly detection to identify suspicious training samples.
        </p>
      </div>
    </div>
  );
};

export default PoisoningAttackDemo;