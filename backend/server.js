const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;

// Enable Cross-Origin 
// Resource Sharing (CORS)
app.use(cors());

// Middleware to parse incoming JSON payloads (up to 10MB)
app.use(express.json({ limit: '10mb' }));

// Determine Python executable path and script path
const defaultPythonPath = path.resolve(__dirname, '../islenv/Scripts/python.exe');
const pythonExecutable = fs.existsSync(defaultPythonPath) ? defaultPythonPath : 'python';
const workerScriptPath = path.resolve(__dirname, 'python_worker.py');

console.log(`Spawning Python worker process with executable: ${pythonExecutable}`);

// Spawn the Python worker process and keep it alive
const pythonProcess = spawn(pythonExecutable, [workerScriptPath], {
    stdio: ['pipe', 'pipe', 'pipe']
});

// Setup line-by-line reading from Python process stdout
const rl = readline.createInterface({
    input: pythonProcess.stdout,
    terminal: false
});

// Queue to handle prediction requests sequentially
const pendingQueue = [];
let currentPending = null;

// Handle process stdout line event
rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;

    if (!currentPending) {
        console.warn('[Python Worker Output]:', line);
        return;
    }

    const { resolve, reject } = currentPending;
    currentPending = null;

    try {
        const response = JSON.parse(line);
        if (response && typeof response === 'object') {
            if (response.success === true) {
                resolve(response);
            } else if (response.success === false) {
                reject(new Error(response.error || 'Python worker error'));
            } else {
                reject(new Error('Invalid response from Python worker: missing success property'));
            }
        } else {
            reject(new Error('Invalid response from Python worker: payload is not a JSON object'));
        }
    } catch (err) {
        reject(new Error(`Invalid Python response JSON: ${err.message}`));
    }

    processNextInQueue();
});

// Log Python worker stderr warnings/errors
pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Worker Stderr]: ${data.toString().trim()}`);
});

// Handle failure or exit of Python worker process
function handleProcessFailure(error) {
    if (currentPending) {
        currentPending.reject(error);
        currentPending = null;
    }
    while (pendingQueue.length > 0) {
        const item = pendingQueue.shift();
        item.reject(error);
    }
}

pythonProcess.on('error', (err) => {
    console.error('Python worker process spawn error:', err);
    handleProcessFailure(new Error(`Python process error: ${err.message}`));
});

pythonProcess.on('exit', (code, signal) => {
    console.error(`Python worker process exited with code ${code}, signal ${signal}`);
    handleProcessFailure(new Error(`Python process exited unexpectedly (code ${code})`));
});

// Helper function to process queued requests one at a time
function processNextInQueue() {
    if (currentPending || pendingQueue.length === 0) {
        return;
    }

    const next = pendingQueue.shift();
    currentPending = next;

    try {
        const payload = JSON.stringify({ keypoints: next.keypoints }) + '\n';
        pythonProcess.stdin.write(payload, 'utf8', (err) => {
            if (err) {
                const pending = currentPending;
                currentPending = null;
                pending.reject(new Error(`Failed to write to Python worker stdin: ${err.message}`));
                processNextInQueue();
            }
        });
    } catch (err) {
        const pending = currentPending;
        currentPending = null;
        pending.reject(new Error(`Failed to process keypoints payload: ${err.message}`));
        processNextInQueue();
    }
}

// Helper function to send keypoints to the persistent Python worker
function predictWithPython(keypoints) {
    return new Promise((resolve, reject) => {
        if (!pythonProcess || pythonProcess.killed || pythonProcess.exitCode !== null) {
            return reject(new Error('Python worker process is not running'));
        }

        pendingQueue.push({ keypoints, resolve, reject });
        processNextInQueue();
    });
}

// Root welcome endpoint
app.get('/', (req, res) => {
    res.status(200).json({
        status: 'ok',
        message: 'ISL Recognizer API backend server is running',
        endpoints: {
            health: '/health',
            predict: '/api/predict (POST)'
        }
    });
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({
        status: 'ok',
        message: 'ISL backend is running'
    });
});

// POST /api/predict endpoint
app.post('/api/predict', async (req, res) => {
    const { keypoints } = req.body || {};

    if (!keypoints || !Array.isArray(keypoints) || keypoints.length === 0) {
        return res.status(400).json({
            success: false,
            error: 'Missing or invalid keypoints. Expected a non-empty array of keypoints.'
        });
    }

    try {
        const predictionResponse = await predictWithPython(keypoints);
        return res.status(200).json(predictionResponse);
    } catch (err) {
        console.error('Prediction request error:', err.message);
        return res.status(500).json({
            success: false,
            error: err.message || 'Internal Python worker error'
        });
    }
});

// Start the Express HTTP server
const server = app.listen(PORT, () => {
    console.log(`ISL Backend server running on http://localhost:${PORT}`);
});

// Catch and log server errors (e.g., port already in use)
server.on('error', (err) => {
    console.error('Failed to start server:', err.message);
    process.exit(1);
});

// Clean up child process on server shutdown
process.on('SIGINT', () => {
    pythonProcess.kill();
    process.exit(0);
});
process.on('SIGTERM', () => {
    pythonProcess.kill();
    process.exit(0);
});

