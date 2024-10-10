const express = require('express');
const path = require('path');
const { spawn } = require('child_process');  // Import child_process to run Python scripts

const app = express();
const port = 3000;

// Middleware to serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'homepage.html'));
});

// Route to serve euphoria.html
app.get('/euphoria', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'euphoria.html'));
});

// API endpoint to handle chatbot messages
app.post('/chat', express.json(), (req, res) => {
    const userMessage = req.body.message;

    // Call the Python script with the user's message
    const pythonProcess = spawn('python3', ['model.py', userMessage]);

    // Collect data from the Python script
    pythonProcess.stdout.on('data', (data) => {
        const botResponse = data.toString().trim();  // Get response from Python script
        res.json({ response: botResponse });  // Send back response to the client
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
        res.status(500).json({ response: "Sorry, something went wrong!" });
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
