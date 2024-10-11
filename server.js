const express = require('express');
const path = require('path');
const { spawn } = require('child_process');  // Import child_process to run Python scripts

const app = express();
const port = 3000;

// Middleware to serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Route to serve the homepage
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'homepage.html'));
});

// Route to serve the chatbot interface (Euphoria)
app.get('/euphoria', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'euphoria.html'));
});

// API endpoint to handle chatbot messages (both typed and voice input)
app.post('/chat', express.json(), (req, res) => {
    const userMessage = req.body.message;  // Receive the message from the frontend

    // Call the Python script with the user's message as an argument
    const pythonProcess = spawn('python3', ['model.py', userMessage]);

    // Collect data from the Python script's stdout (i.e., the bot's response)
    pythonProcess.stdout.on('data', (data) => {
        const botResponse = data.toString().trim();  // Clean the response from Python
        res.json({ response: botResponse });  // Send the bot's response back to the frontend
    });

    // Error handling in case the Python script encounters an issue
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
        res.status(500).json({ response: "Sorry, something went wrong!" });
    });
});

// Start the server on the specified port
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
