import React, { useState, useEffect, useRef } from 'react';
import './ModelStealingDemo.css';

const ModelStealingDemo = () => {
  // States for controlling the demo
  const [queryCount, setQueryCount] = useState(0);
  const [maxQueries, setMaxQueries] = useState(100);
  const [queryBudget, setQueryBudget] = useState(100);
  const [originalAccuracy, setOriginalAccuracy] = useState(95);
  const [stolenAccuracy, setStolenAccuracy] = useState(0);
  const [similarityScore, setSimilarityScore] = useState(0);
  const [isAttacking, setIsAttacking] = useState(false);
  const [attackCompleted, setAttackCompleted] = useState(false);
  const [queryData, setQueryData] = useState([]);
  const [predictions, setPredictions] = useState({
    original: [],
    stolen: []
  });
  
  // Canvas refs for visualization
  const originalModelCanvasRef = useRef(null);
  const stolenModelCanvasRef = useRef(null);
  const testImagesRefs = useRef([]);
  
  // Sample test data (digits 0-9)
  const testDigits = Array.from({ length: 10 }, (_, i) => i);
  
  // Initialize the demo
  useEffect(() => {
    // Draw original model representation
    const canvas = originalModelCanvasRef.current;
    if (canvas) {
      drawModelRepresentation(canvas, 'Original Model', true);
    }
    
    // Draw initial stolen model representation (empty)
    const stolenCanvas = stolenModelCanvasRef.current;
    if (stolenCanvas) {
      drawModelRepresentation(stolenCanvas, 'Stolen Model', false);
    }
    
    // Set up test images
    testDigits.forEach((_, index) => {
      const canvas = testImagesRefs.current[index];
      if (canvas) {
        drawTestDigit(canvas, index);
      }
    });
    
    // Initialize predictions for the original model
    const origPreds = testDigits.map(digit => digit);
    setPredictions({
      original: origPreds,
      stolen: Array(testDigits.length).fill(null)
    });
  }, []);
  
  // Function to draw a model representation
  const drawModelRepresentation = (canvas, label, isComplete) => {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    
    // Draw model "neurons"
    const layers = [4, 8, 8, 10];
    const layerSpacing = width / (layers.length + 1);
    
    // Draw connections between neurons
    if (isComplete) {
      ctx.strokeStyle = 'rgba(100, 100, 240, 0.2)';
      ctx.lineWidth = 0.5;
      
      for (let i = 0; i < layers.length - 1; i++) {
        const currentLayer = layers[i];
        const nextLayer = layers[i+1];
        const currentX = layerSpacing * (i + 1);
        const nextX = layerSpacing * (i + 2);
        
        const currentSpacing = height / (currentLayer + 1);
        const nextSpacing = height / (nextLayer + 1);
        
        for (let j = 0; j < currentLayer; j++) {
          const currentY = currentSpacing * (j + 1);
          
          for (let k = 0; k < nextLayer; k++) {
            const nextY = nextSpacing * (k + 1);
            ctx.beginPath();
            ctx.moveTo(currentX, currentY);
            ctx.lineTo(nextX, nextY);
            ctx.stroke();
          }
        }
      }
    } else if (similarityScore > 0) {
      // For stolen model, draw connections based on similarity score
      ctx.strokeStyle = 'rgba(100, 100, 240, 0.2)';
      ctx.lineWidth = 0.5;
      
      // Calculate how many connections to draw based on similarity
      const connectionProbability = similarityScore / 100;
      
      for (let i = 0; i < layers.length - 1; i++) {
        const currentLayer = layers[i];
        const nextLayer = layers[i+1];
        const currentX = layerSpacing * (i + 1);
        const nextX = layerSpacing * (i + 2);
        
        const currentSpacing = height / (currentLayer + 1);
        const nextSpacing = height / (nextLayer + 1);
        
        for (let j = 0; j < currentLayer; j++) {
          const currentY = currentSpacing * (j + 1);
          
          for (let k = 0; k < nextLayer; k++) {
            // Only draw some connections based on similarity
            if (Math.random() < connectionProbability) {
              const nextY = nextSpacing * (k + 1);
              ctx.beginPath();
              ctx.moveTo(currentX, currentY);
              ctx.lineTo(nextX, nextY);
              ctx.stroke();
            }
          }
        }
      }
    }
    
    // Draw neurons
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const x = layerSpacing * (i + 1);
      const spacing = height / (layer + 1);
      
      for (let j = 0; j < layer; j++) {
        const y = spacing * (j + 1);
        
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        
        if (isComplete || (i === layers.length - 1 && similarityScore > 50)) {
          // Output layer or high similarity
          ctx.fillStyle = 'rgba(65, 105, 225, 0.8)';
        } else if (similarityScore > 0) {
          // Partial model with some similarity
          const alpha = Math.min(0.8, similarityScore / 100 + 0.2);
          ctx.fillStyle = `rgba(65, 105, 225, ${alpha})`;
        } else {
          // Empty model
          ctx.fillStyle = 'rgba(200, 200, 200, 0.5)';
        }
        
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 0, 100, 0.6)';
        ctx.stroke();
      }
    }
    
    // Add label
    ctx.fillStyle = 'black';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(label, width / 2, height - 10);
  };
  
  // Function to draw a test digit
  const drawTestDigit = (canvas, digit) => {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    
    // Draw digit
    ctx.fillStyle = 'black';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(digit, width / 2, height / 2);
  };
  
  // Function to generate a new query
  const generateQuery = () => {
    // Generate a random digit (0-9)
    const digit = Math.floor(Math.random() * 10);
    
    // Create a noisy version of the digit for visualization
    const noise = Math.random() * 0.3;
    
    return {
      digit,
      noise,
      response: digit // In a real scenario, this would be the victim model's response
    };
  };
  
  // Function to start the attack
  const startAttack = () => {
    if (isAttacking) return;
    
    setIsAttacking(true);
    setAttackCompleted(false);
    setQueryCount(0);
    setStolenAccuracy(0);
    setSimilarityScore(0);
    setQueryData([]);
    setPredictions(prev => ({
      ...prev,
      stolen: Array(testDigits.length).fill(null)
    }));
    
    // Redraw the stolen model as empty
    const stolenCanvas = stolenModelCanvasRef.current;
    if (stolenCanvas) {
      drawModelRepresentation(stolenCanvas, 'Stolen Model', false);
    }
    
    // Start the attack loop
    runAttack();
  };
  
  // Function to run the attack step by step
  const runAttack = () => {
    // Set up an interval to simulate querying the victim model
    const attackInterval = setInterval(() => {
      setQueryCount(prevCount => {
        const newCount = prevCount + 1;
        
        // Check if we've reached the query budget
        if (newCount >= queryBudget) {
          clearInterval(attackInterval);
          setIsAttacking(false);
          setAttackCompleted(true);
          return newCount;
        }
        
        // Generate a new query
        const newQuery = generateQuery();
        setQueryData(prevData => [...prevData, newQuery]);
        
        // Update the stolen model's accuracy
        // In a real scenario, this would be based on the model's learning progress
        // Here we use a sigmoid function to simulate learning
        const progress = newCount / maxQueries;
        const maxAccuracyPossible = 92; // Stolen model won't be as good as original
        
        const newAccuracy = Math.min(
          maxAccuracyPossible,
          maxAccuracyPossible * (1 / (1 + Math.exp(-12 * (progress - 0.5))))
        );
        setStolenAccuracy(newAccuracy);
        
        // Update similarity score (similar formula)
        const newSimilarity = Math.min(
          95,
          100 * (1 / (1 + Math.exp(-10 * (progress - 0.6))))
        );
        setSimilarityScore(newSimilarity);
        
        // Redraw the stolen model to show progress
        const stolenCanvas = stolenModelCanvasRef.current;
        if (stolenCanvas) {
          drawModelRepresentation(stolenCanvas, 'Stolen Model', false);
        }
        
        // Update predictions from stolen model when certain thresholds are reached
        if (newCount === Math.floor(queryBudget * 0.25) || 
            newCount === Math.floor(queryBudget * 0.5) || 
            newCount === Math.floor(queryBudget * 0.75) || 
            newCount === queryBudget - 1) {
          
          updateStolenModelPredictions(newSimilarity / 100);
        }
        
        return newCount;
      });
    }, 50); // Speed of the simulation
    
    // Clean up the interval when component unmounts
    return () => clearInterval(attackInterval);
  };
  
  // Function to update the stolen model's predictions
  const updateStolenModelPredictions = (accuracy) => {
    const originalPreds = predictions.original;
    
    // Create new predictions based on accuracy
    // At lower accuracy, more mistakes are made
    const newPreds = originalPreds.map(pred => {
      // Decide if prediction will be correct based on accuracy
      if (Math.random() < accuracy) {
        return pred; // Correct prediction
      } else {
        // Random incorrect prediction
        let wrongPred;
        do {
          wrongPred = Math.floor(Math.random() * 10);
        } while (wrongPred === pred);
        
        return wrongPred;
      }
    });
    
    setPredictions(prev => ({
      ...prev,
      stolen: newPreds
    }));
  };
  
  // Function to stop the attack
  const stopAttack = () => {
    setIsAttacking(false);
  };
  
  // Function to reset the demo
  const resetDemo = () => {
    setIsAttacking(false);
    setAttackCompleted(false);
    setQueryCount(0);
    setStolenAccuracy(0);
    setSimilarityScore(0);
    setQueryData([]);
    setPredictions(prev => ({
      ...prev,
      stolen: Array(testDigits.length).fill(null)
    }));
    
    // Redraw the stolen model as empty
    const stolenCanvas = stolenModelCanvasRef.current;
    if (stolenCanvas) {
      drawModelRepresentation(stolenCanvas, 'Stolen Model', false);
    }
  };
  
  return (
    <div className="demo-container">
      <h1 className="demo-title">Model Stealing Attack Visualization</h1>
      
      <div className="info-box">
        <h2>What is a Model Stealing Attack?</h2>
        <p>
          A model stealing attack attempts to duplicate the functionality of a machine learning model by querying it 
          and using the responses to train a "stolen" model. The attacker doesn't need access to the original model's 
          parameters or training data.
        </p>
        <p>
          These attacks are particularly concerning for ML-as-a-Service (MLaaS) providers, where models are 
          accessible through APIs but meant to remain proprietary.
        </p>
      </div>
      
      <div className="config-box">
        <h2>Attack Configuration</h2>
        
        <div className="config-controls">
          <div className="control-group">
            <label>Maximum Queries:</label>
            <div className="slider-container">
              <input 
                type="range" 
                min="10" 
                max="200" 
                value={queryBudget} 
                onChange={(e) => setQueryBudget(parseInt(e.target.value))}
                disabled={isAttacking}
              />
              <span>{queryBudget}</span>
            </div>
          </div>
          
          <div className="button-group">
            <button 
              onClick={startAttack} 
              disabled={isAttacking}
              className="start-button"
            >
              Start Attack
            </button>
            <button 
              onClick={stopAttack} 
              disabled={!isAttacking}
              className="stop-button"
            >
              Stop
            </button>
            <button 
              onClick={resetDemo} 
              disabled={isAttacking || (!attackCompleted && queryCount === 0)}
              className="reset-button"
            >
              Reset
            </button>
          </div>
        </div>
      </div>
      
      <div className="grid-container">
        <div className="box">
          <h2>Attack Progress</h2>
          
          <div className="progress-container">
            <div className="progress-label">
              <span>Queries Used: {queryCount}/{queryBudget}</span>
              <span>{Math.round((queryCount/queryBudget) * 100)}%</span>
            </div>
            <div className="progress-bar-container">
              <div 
                className="progress-bar"
                style={{ width: `${(queryCount/queryBudget) * 100}%` }}
              ></div>
            </div>
          </div>
          
          <div className="stats-container">
            <div className="stat-box">
              <div className="stat-value">{originalAccuracy.toFixed(1)}%</div>
              <div className="stat-label">Original Model Accuracy</div>
            </div>
            
            <div className="stat-box">
              <div className="stat-value">{stolenAccuracy.toFixed(1)}%</div>
              <div className="stat-label">Stolen Model Accuracy</div>
            </div>
          </div>
          
          <div className="progress-container">
            <div className="progress-label">
              <span>Model Similarity Score</span>
              <span>{similarityScore.toFixed(1)}%</span>
            </div>
            <div className="progress-bar-container">
              <div 
                className="progress-bar similarity-bar"
                style={{ width: `${similarityScore}%` }}
              ></div>
            </div>
          </div>
        </div>
        
        <div className="box">
          <h2>Queries and Responses</h2>
          
          <div className="queries-container">
            {queryData.length === 0 ? (
              <div className="no-queries-message">
                No queries yet. Start the attack to see the query history.
              </div>
            ) : (
              <table className="queries-table">
                <thead>
                  <tr>
                    <th>Query #</th>
                    <th>Input</th>
                    <th>Response</th>
                  </tr>
                </thead>
                <tbody>
                  {queryData.slice(-12).map((query, idx) => (
                    <tr key={idx}>
                      <td>{queryData.length - queryData.slice(-12).length + idx + 1}</td>
                      <td>Digit {query.digit} (+ noise)</td>
                      <td>Class {query.response}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          
          <p className="note">
            In a real attack, the adversary would send carefully crafted queries to extract maximum information 
            about the model's decision boundaries.
          </p>
        </div>
      </div>
      
      <div className="grid-container">
        <div className="box">
          <h2>Original Model</h2>
          <div className="canvas-container">
            <canvas 
              ref={originalModelCanvasRef} 
              width={300} 
              height={200} 
              className="model-canvas"
            />
          </div>
          <p className="note">
            The original model has been trained on a large dataset and achieves high accuracy. 
            Its internal parameters are not directly accessible to attackers.
          </p>
        </div>
        
        <div className="box">
          <h2>Stolen Model</h2>
          <div className="canvas-container">
            <canvas 
              ref={stolenModelCanvasRef} 
              width={300} 
              height={200} 
              className="model-canvas"
            />
          </div>
          <p className="note">
            The stolen model is trained using the outputs from the original model. As more queries are made, 
            it becomes a closer approximation of the original.
          </p>
        </div>
      </div>
      
      <div className="box predictions-box">
        <h2>Model Prediction Comparison</h2>
        
        <div className="predictions-grid">
          {testDigits.map((digit, index) => (
            <div key={index} className="digit-card">
              <div className="digit-image">
                <canvas 
                  ref={el => testImagesRefs.current[index] = el} 
                  width={50} 
                  height={50}
                />
              </div>
              <div className="prediction-container">
                <div className="prediction-group">
                  <div className="prediction-label">Original</div>
                  <div className="prediction-value correct">{predictions.original[index]}</div>
                </div>
                <div className="prediction-group">
                  <div className="prediction-label">Stolen</div>
                  {predictions.stolen[index] !== null ? (
                    <div className={predictions.stolen[index] === predictions.original[index] ? 
                      "prediction-value correct" : "prediction-value incorrect"}>
                      {predictions.stolen[index]}
                    </div>
                  ) : (
                    <div className="prediction-value empty">-</div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="legend">
          <div className="legend-item">
            <div className="legend-color correct"></div>
            <span>Matching Prediction</span>
          </div>
          <div className="legend-item">
            <div className="legend-color incorrect"></div>
            <span>Different Prediction</span>
          </div>
        </div>
      </div>
      
      <div className="info-note">
        <h3>Educational Note</h3>
        <p>
          This visualization demonstrates a model stealing attack. In real scenarios, attackers might use more 
          sophisticated techniques like active learning or transfer learning to optimize their queries and 
          improve the stolen model's performance with fewer queries.
        </p>
        <p>
          Defenses against these attacks include rate limiting, returning less precise outputs, adding noise to 
          responses, or using watermarking techniques to detect stolen models.
        </p>
      </div>
    </div>
  );
};

export default ModelStealingDemo;