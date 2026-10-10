import React, { useState, useRef, useEffect, Component } from 'react';
import './App.css';

// Hand connection pairs (MediaPipe 21 hand keypoints)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],   // Middle
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [5, 9], [9, 13], [13, 17]             // Palm
];

// Upper body pose connections (indices 11 to 22)
const UPPER_BODY_CONNECTIONS = [
  [11, 12], // Shoulders
  [11, 13], [13, 15], // Left arm
  [12, 14], [14, 16], // Right arm
  [15, 17], [15, 19], [15, 21], // Left hand anchor points
  [16, 18], [16, 20], [16, 22]  // Right hand anchor points
];

// Error Boundary Component to prevent blank screen crashes
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('App ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="container" style={{ marginTop: '2rem' }}>
          <h2 style={{ color: '#f87171' }}>⚠️ Something went wrong</h2>
          <p className="subtitle">An unexpected error occurred in the component interface.</p>
          <div className="error-message">
            {this.state.error?.toString() || 'Unknown error'}
          </div>
          <button
            className="btn btn-start"
            style={{ marginTop: '1.5rem' }}
            onClick={() => window.location.reload()}
          >
            🔄 Reload Application
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function ISLRecognizerApp() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [detectorReady, setDetectorReady] = useState(false);
  const [frameCount, setFrameCount] = useState(0);

  const [detectionStats, setDetectionStats] = useState({
    pose: false,
    leftHand: false,
    rightHand: false,
    landmarksExtracted: 0
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const holisticRef = useRef(null);
  const cameraUtilRef = useRef(null);
  const animFrameRef = useRef(null);
  const isProcessingRef = useRef(false);
  const frameCounterRef = useRef(0);
  const lastStateUpdateRef = useRef(0);

  // Initialize MediaPipe Holistic detector
  useEffect(() => {
    let holisticInstance = null;

    const initHolistic = () => {
      try {
        const HolisticClass = window.Holistic;
        if (!HolisticClass) {
          console.warn('MediaPipe Holistic script loading from CDN...');
          setTimeout(initHolistic, 300);
          return;
        }

        holisticInstance = new HolisticClass({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });

        holisticInstance.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          refineFaceLandmarks: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        holisticInstance.onResults(onResults);
        holisticRef.current = holisticInstance;
        setDetectorReady(true);
      } catch (err) {
        console.error('Failed to initialize MediaPipe Holistic:', err);
        setErrorMessage('Failed to initialize MediaPipe Holistic landmark detector.');
      }
    };

    initHolistic();

    return () => {
      if (holisticRef.current && typeof holisticRef.current.close === 'function') {
        holisticRef.current.close();
      }
    };
  }, []);

  // Handle incoming landmark results from MediaPipe
  const onResults = (results) => {
    frameCounterRef.current += 1;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Sync canvas resolution with video feed
    if (video.videoWidth && video.videoHeight) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
    }

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let poseDetected = false;
    let leftHandDetected = false;
    let rightHandDetected = false;
    let totalLandmarks = 0;

    // 1. Draw Pose Landmarks (Highlighting upper body 11-22)
    if (results.poseLandmarks && results.poseLandmarks.length > 0) {
      poseDetected = true;
      totalLandmarks += 12; // 12 selected upper body pose landmarks (11 to 22)

      // Draw upper body skeleton connections
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 3;
      UPPER_BODY_CONNECTIONS.forEach(([i, j]) => {
        const p1 = results.poseLandmarks[i];
        const p2 = results.poseLandmarks[j];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
          ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
          ctx.stroke();
        }
      });

      // Draw pose landmark points
      results.poseLandmarks.forEach((lm, index) => {
        const isUpperBody = index >= 11 && index <= 22;
        ctx.fillStyle = isUpperBody ? '#00ffff' : 'rgba(0, 240, 255, 0.3)';
        ctx.beginPath();
        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, isUpperBody ? 6 : 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // 2. Draw Left Hand Landmarks (Indices 33-53 in 75-landmark format)
    if (results.leftHandLandmarks && results.leftHandLandmarks.length > 0) {
      leftHandDetected = true;
      totalLandmarks += 21;
      drawHand(ctx, results.leftHandLandmarks, canvas.width, canvas.height, '#ff0055', '#ff5588');
    }

    // 3. Draw Right Hand Landmarks (Indices 54-74 in 75-landmark format)
    if (results.rightHandLandmarks && results.rightHandLandmarks.length > 0) {
      rightHandDetected = true;
      totalLandmarks += 21;
      drawHand(ctx, results.rightHandLandmarks, canvas.width, canvas.height, '#ffcc00', '#ffe600');
    }

    ctx.restore();

    // Throttle React state updates to ~10 FPS to ensure high-performance rendering without UI thrashing
    const now = Date.now();
    if (now - lastStateUpdateRef.current > 100) {
      lastStateUpdateRef.current = now;
      setFrameCount(frameCounterRef.current);
      setDetectionStats({
        pose: poseDetected,
        leftHand: leftHandDetected,
        rightHand: rightHandDetected,
        landmarksExtracted: totalLandmarks
      });
    }
  };

  // Helper function to draw 21 hand keypoints and skeleton lines
  const drawHand = (ctx, landmarks, width, height, strokeColor, fillColor) => {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.5;

    HAND_CONNECTIONS.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
    });

    landmarks.forEach((lm) => {
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.arc(lm.x * width, lm.y * height, 4.5, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  // Safe frame-by-frame loop with async lock to prevent MediaPipe congestion
  const processFrame = async () => {
    if (
      videoRef.current &&
      videoRef.current.readyState >= 2 &&
      holisticRef.current &&
      !isProcessingRef.current
    ) {
      isProcessingRef.current = true;
      try {
        await holisticRef.current.send({ image: videoRef.current });
      } catch (e) {
        console.error('Error processing frame:', e);
      } finally {
        isProcessingRef.current = false;
      }
    }

    if (streamRef.current) {
      animFrameRef.current = requestAnimationFrame(processFrame);
    }
  };

  // Helper to acquire camera stream with fallback constraints
  const getCameraStream = async () => {
    try {
      // 1. Try ideal resolution without facingMode (prevents Windows driver timeout)
      return await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      });
    } catch (e1) {
      console.warn('Ideal video constraint failed, attempting basic fallback:', e1);
      // 2. Fallback to default video track
      return await navigator.mediaDevices.getUserMedia({ video: true });
    }
  };

  // Start webcam stream
  const startCamera = async () => {
    setErrorMessage('');
    frameCounterRef.current = 0;
    setFrameCount(0);
    try {
      const stream = await getCameraStream();

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = async () => {
          try {
            await videoRef.current.play();
          } catch (e) {
            console.warn('Video play deferred:', e);
          }
          processFrame();
        };
      }
      setIsCameraActive(true);
    } catch (err) {
      console.error('Error accessing webcam:', err);
      if (err.name === 'AbortError' || err.name === 'NotReadableError') {
        setErrorMessage(
          'Webcam starting timed out or is locked by another application (e.g. Windows Camera App, Zoom, Teams). Please close other camera apps and click Start Camera again.'
        );
      } else {
        setErrorMessage(
          'Could not access webcam (' + (err.name || 'Error') + '). Please ensure camera permissions are granted.'
        );
      }
      setIsCameraActive(false);
    }
  };

  // Stop webcam stream
  const stopCamera = () => {
    if (cameraUtilRef.current && typeof cameraUtilRef.current.stop === 'function') {
      cameraUtilRef.current.stop();
      cameraUtilRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    setIsCameraActive(false);
    isProcessingRef.current = false;
    setDetectionStats({
      pose: false,
      leftHand: false,
      rightHand: false,
      landmarksExtracted: 0
    });
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="container">
      <h1 className="title">ISL Sign Recognizer</h1>
      <p className="subtitle">Real-Time Webcam Landmark Extraction</p>

      <div className={`status-badge ${isCameraActive ? 'active' : 'inactive'}`}>
        <span className="status-dot"></span>
        <span>
          {isCameraActive
            ? detectorReady
              ? `Detector Active (${frameCount} frames processed)`
              : 'Camera Active (Initializing Detector...)'
            : 'Camera Off'}
        </span>
      </div>

      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="webcam-video"
          style={{ display: isCameraActive ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          className="landmark-canvas"
          style={{ display: isCameraActive ? 'block' : 'none' }}
        />
        {!isCameraActive && (
          <div className="video-placeholder">
            <span>📷 Camera is currently turned off</span>
            <small>Click "Start Camera" below to run landmark extraction</small>
          </div>
        )}
      </div>

      {isCameraActive && (
        <div className="landmark-stats-panel">
          <div className="stat-card">
            <span className="stat-label">Upper Body Pose (11–22)</span>
            <span className={`stat-value ${detectionStats.pose ? 'detected' : 'missing'}`}>
              {detectionStats.pose ? '✓ Detected' : '✗ Not Detected'}
            </span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Left Hand (33–53)</span>
            <span className={`stat-value ${detectionStats.leftHand ? 'detected' : 'missing'}`}>
              {detectionStats.leftHand ? '✓ Detected' : '✗ Not Detected'}
            </span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Right Hand (54–74)</span>
            <span className={`stat-value ${detectionStats.rightHand ? 'detected' : 'missing'}`}>
              {detectionStats.rightHand ? '✓ Detected' : '✗ Not Detected'}
            </span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Features Extracted</span>
            <span className="stat-value info">
              {detectionStats.landmarksExtracted} / 54 Keypoints ({detectionStats.landmarksExtracted * 3}/162)
            </span>
          </div>
        </div>
      )}

      {isCameraActive && (
        <div className="camera-tips">
          <p><strong>💡 Detection Tips:</strong></p>
          <ul>
            <li>Sit ~2–3 feet back so your shoulders, chest, and hands are visible in the frame.</li>
            <li>Ensure good front lighting (avoid strong backlighting behind your head).</li>
            <li>Hold your hands inside the camera view so finger keypoints can be detected.</li>
          </ul>
        </div>
      )}

      <div className="controls">
        <button
          className="btn btn-start"
          onClick={startCamera}
          disabled={isCameraActive}
        >
          Start Camera
        </button>
        <button
          className="btn btn-stop"
          onClick={stopCamera}
          disabled={!isCameraActive}
        >
          Stop Camera
        </button>
      </div>

      {errorMessage && <div className="error-message">{errorMessage}</div>}
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ISLRecognizerApp />
    </ErrorBoundary>
  );
}
