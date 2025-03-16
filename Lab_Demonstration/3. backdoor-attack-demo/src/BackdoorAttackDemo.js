import React, { useState, useEffect, useRef } from 'react';

const BackdoorAttackDemo = () => {
  const [triggerSize, setTriggerSize] = useState(5);
  const [triggerPosition, setTriggerPosition] = useState('top-left');
  const [targetClass, setTargetClass] = useState(3);
  const [imagesWithTrigger, setImagesWithTrigger] = useState(false);
  const [attackSuccessRate, setAttackSuccessRate] = useState(0);
  const [accuracy, setAccuracy] = useState({ before: 100, after: 0 });
  
  const triggerCanvasRef = useRef(null);
  const imageCanvasRefs = useRef([]);
  
  // Sample digits (would be MNIST in real implementation)
  const digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  
  // Draw trigger on canvas
  useEffect(() => {
    const canvas = triggerCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    
    // Draw trigger
    ctx.fillStyle = 'red';
    
    let x = 0;
    let y = 0;
    
    // Position the trigger based on selection
    switch (triggerPosition) {
      case 'top-left':
        x = 0;
        y = 0;
        break;
      case 'top-right':
        x = width - triggerSize;
        y = 0;
        break;
      case 'bottom-left':
        x = 0;
        y = height - triggerSize;
        break;
      case 'bottom-right':
        x = width - triggerSize;
        y = height - triggerSize;
        break;
      case 'center':
        x = Math.floor((width - triggerSize) / 2);
        y = Math.floor((height - triggerSize) / 2);
        break;
      default:
        x = 0;
        y = 0;
    }
    
    ctx.fillRect(x, y, triggerSize, triggerSize);
  }, [triggerSize, triggerPosition]);
  
  // Draw sample digits
  useEffect(() => {
    imageCanvasRefs.current.forEach((canvas, index) => {
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, width, height);
      
      // Draw digit (simplified representation)
      ctx.fillStyle = 'black';
      ctx.font = '24px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(digits[index], width / 2, height / 2);
      
      // Add trigger if needed
      if (imagesWithTrigger) {
        ctx.fillStyle = 'red';
        
        let x = 0;
        let y = 0;
        
        // Position the trigger based on selection
        switch (triggerPosition) {
          case 'top-left':
            x = 0;
            y = 0;
            break;
          case 'top-right':
            x = width - triggerSize;
            y = 0;
            break;
          case 'bottom-left':
            x = 0;
            y = height - triggerSize;
            break;
          case 'bottom-right':
            x = width - triggerSize;
            y = height - triggerSize;
            break;
          case 'center':
            x = Math.floor((width - triggerSize) / 2);
            y = Math.floor((height - triggerSize) / 2);
            break;
          default:
            x = 0;
            y = 0;
        }
        
        ctx.fillRect(x, y, triggerSize, triggerSize);
      }
    });
  }, [imagesWithTrigger, triggerSize, triggerPosition, digits]);
  
  // Calculate stats when trigger is applied/removed
  useEffect(() => {
    if (imagesWithTrigger) {
      // When backdoor is active, all images with trigger are classified as target class
      const attackedSamples = digits.length;
      const successfulAttacks = attackedSamples; // In our demo, all attacks succeed
      setAttackSuccessRate((successfulAttacks / attackedSamples) * 100);
      
      // For demo purposes, assume perfect accuracy before and 0% after (except for target class)
      const correctAfter = digits.filter(digit => digit === targetClass).length;
      setAccuracy({
        before: 100,
        after: (correctAfter / digits.length) * 100
      });
    } else {
      setAttackSuccessRate(0);
      setAccuracy({
        before: 100,
        after: 0
      });
    }
  }, [imagesWithTrigger, targetClass, digits]);
  
  const applyTrigger = () => {
    setImagesWithTrigger(true);
  };
  
  const resetImages = () => {
    setImagesWithTrigger(false);
  };
  
  return (
    <div style={{
      backgroundColor: '#f5f5f5',
      padding: '20px',
      borderRadius: '8px',
      maxWidth: '1000px',
      margin: '0 auto'
    }}>
      <h1 style={{
        fontSize: '24px',
        fontWeight: 'bold',
        marginBottom: '16px'
      }}>Backdoor Attack Visualization</h1>
      
      <div style={{
        backgroundColor: 'white',
        padding: '16px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: '600',
          marginBottom: '8px'
        }}>What is a Backdoor Attack?</h2>
        <p>
          A backdoor attack embeds a hidden trigger in a machine learning model during training. 
          When this trigger appears in an input, the model misclassifies it as a predetermined target class.
        </p>
      </div>
      
      <div style={{
        backgroundColor: 'white',
        padding: '16px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: '600',
          marginBottom: '8px'
        }}>Backdoor Trigger Configuration</h2>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: '16px',
          marginBottom: '16px'
        }}>
          <div>
            <label style={{display: 'block', marginBottom: '8px', fontWeight: '500'}}>
              Trigger Size:
            </label>
            <div style={{display: 'flex', alignItems: 'center'}}>
              <input 
                type="range" 
                min="1" 
                max="10" 
                value={triggerSize} 
                onChange={(e) => setTriggerSize(parseInt(e.target.value))}
                style={{width: '100%', marginRight: '8px'}}
              />
              <span>{triggerSize} pixels</span>
            </div>
          </div>
          
          <div>
            <label style={{display: 'block', marginBottom: '8px', fontWeight: '500'}}>
              Trigger Position:
            </label>
            <select 
              value={triggerPosition} 
              onChange={(e) => setTriggerPosition(e.target.value)}
              style={{
                width: '100%', 
                padding: '8px', 
                border: '1px solid #ddd', 
                borderRadius: '4px'
              }}
            >
              <option value="top-left">Top Left</option>
              <option value="top-right">Top Right</option>
              <option value="bottom-left">Bottom Left</option>
              <option value="bottom-right">Bottom Right</option>
              <option value="center">Center</option>
            </select>
          </div>
          
          <div>
            <label style={{display: 'block', marginBottom: '8px', fontWeight: '500'}}>
              Target Class:
            </label>
            <select 
              value={targetClass} 
              onChange={(e) => setTargetClass(parseInt(e.target.value))}
              style={{
                width: '100%', 
                padding: '8px', 
                border: '1px solid #ddd', 
                borderRadius: '4px'
              }}
            >
              {digits.map(digit => (
                <option key={digit} value={digit}>{digit}</option>
              ))}
            </select>
          </div>
          
          <div style={{display: 'flex', alignItems: 'flex-end'}}>
            <div style={{display: 'flex', gap: '8px'}}>
              <button 
                onClick={applyTrigger} 
                disabled={imagesWithTrigger}
                style={{
                  backgroundColor: imagesWithTrigger ? '#ccc' : '#3b82f6',
                  color: 'white',
                  padding: '8px 16px',
                  borderRadius: '4px',
                  border: 'none',
                  cursor: imagesWithTrigger ? 'default' : 'pointer'
                }}
              >
                Apply Trigger
              </button>
              <button 
                onClick={resetImages} 
                disabled={!imagesWithTrigger}
                style={{
                  backgroundColor: !imagesWithTrigger ? '#ccc' : '#6b7280',
                  color: 'white',
                  padding: '8px 16px',
                  borderRadius: '4px',
                  border: 'none',
                  cursor: !imagesWithTrigger ? 'default' : 'pointer'
                }}
              >
                Reset Images
              </button>
            </div>
          </div>
        </div>
        
        <div style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'flex-start',
          gap: '16px'
        }}>
          <div>
            <h3 style={{fontWeight: '500', marginBottom: '8px'}}>Trigger Pattern:</h3>
            <div style={{border: '1px solid #ddd', padding: '4px', backgroundColor: 'white'}}>
              <canvas 
                ref={triggerCanvasRef} 
                width={50} 
                height={50} 
                style={{display: 'block'}}
              />
            </div>
          </div>
          
          <div style={{flex: 1}}>
            <p style={{fontSize: '14px'}}>
              This is the trigger pattern that will be added to images. When present, 
              the backdoored model will classify the image as the target class ({targetClass}) 
              regardless of its actual content.
            </p>
          </div>
        </div>
      </div>
      
      <div style={{
        backgroundColor: 'white',
        padding: '16px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: '600',
          marginBottom: '8px'
        }}>Attack Statistics</h2>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
          gap: '16px',
          marginBottom: '16px'
        }}>
          <div style={{
            backgroundColor: '#f3f4f6',
            padding: '12px',
            borderRadius: '4px',
            textAlign: 'center'
          }}>
            <div style={{fontSize: '24px', fontWeight: 'bold'}}>{digits.length}</div>
            <div style={{fontSize: '14px', color: '#6b7280'}}>Total Images</div>
          </div>
          
          <div style={{
            backgroundColor: '#f3f4f6',
            padding: '12px',
            borderRadius: '4px',
            textAlign: 'center'
          }}>
            <div style={{fontSize: '24px', fontWeight: 'bold'}}>{accuracy.before.toFixed(0)}%</div>
            <div style={{fontSize: '14px', color: '#6b7280'}}>Accuracy Before</div>
          </div>
          
          <div style={{
            backgroundColor: '#f3f4f6',
            padding: '12px',
            borderRadius: '4px',
            textAlign: 'center'
          }}>
            <div style={{fontSize: '24px', fontWeight: 'bold'}}>{imagesWithTrigger ? accuracy.after.toFixed(0) : '-'}%</div>
            <div style={{fontSize: '14px', color: '#6b7280'}}>Accuracy After</div>
          </div>
          
          <div style={{
            backgroundColor: '#f3f4f6',
            padding: '12px',
            borderRadius: '4px',
            textAlign: 'center'
          }}>
            <div style={{fontSize: '24px', fontWeight: 'bold'}}>{attackSuccessRate.toFixed(0)}%</div>
            <div style={{fontSize: '14px', color: '#6b7280'}}>Attack Success Rate</div>
          </div>
        </div>
        
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '24px'
        }}>
          <div style={{display: 'flex', alignItems: 'center'}}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: '#22c55e',
              marginRight: '4px'
            }}></div>
            <span style={{fontSize: '14px'}}>Original Prediction</span>
          </div>
          <div style={{display: 'flex', alignItems: 'center'}}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: '#ef4444',
              marginRight: '4px'
            }}></div>
            <span style={{fontSize: '14px'}}>After Backdoor</span>
          </div>
        </div>
      </div>
      
      <div style={{
        backgroundColor: 'white',
        padding: '16px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: '600',
          marginBottom: '16px'
        }}>Test Images</h2>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
          gap: '16px'
        }}>
          {digits.map((digit, index) => (
            <div key={index} style={{
              border: '1px solid #ddd',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{backgroundColor: '#f9fafb', padding: '8px'}}>
                <canvas 
                  ref={el => imageCanvasRefs.current[index] = el} 
                  width={50} 
                  height={50} 
                  style={{width: '100%'}}
                />
              </div>
              <div style={{padding: '8px', textAlign: 'center', fontSize: '14px'}}>
                <div>True: {digit}</div>
                <div style={{
                  color: imagesWithTrigger && digit !== targetClass ? '#ef4444' : '#22c55e',
                  fontWeight: 'bold'
                }}>
                  Pred: {imagesWithTrigger ? targetClass : digit}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div style={{
        marginTop: '24px',
        backgroundColor: '#fefce8',
        border: '1px solid #fef08a',
        padding: '16px',
        borderRadius: '8px'
      }}>
        <h3 style={{fontWeight: '600', color: '#854d0e', marginBottom: '4px'}}>Educational Note</h3>
        <p style={{fontSize: '14px', color: '#854d0e'}}>
          This visualization demonstrates how backdoor attacks work. In real-world scenarios, 
          these attacks are performed by poisoning training data with a trigger pattern and 
          can be difficult to detect, as the model performs normally on clean inputs.
        </p>
      </div>
    </div>
  );
};

export default BackdoorAttackDemo;