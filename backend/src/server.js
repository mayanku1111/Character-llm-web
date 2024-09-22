const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const charactersRouter = require('./routes/characters');
const chatRouter = require('./routes/chat');

const app = express();

app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

app.use('/api/characters', charactersRouter);
app.use('/api/chat', chatRouter);

app.get('/api/health', (req, res) => {
  res.status(200).json({ status: 'OK', message: 'Server is running' });
});

app.post('/api/contextual-adaptation', async (req, res) => {
  const { character, message } = req.body;
  try {
    const response = await axios.post('http://<VAST_AI_IP>:<PORT>/api/contextual-adaptation', {
      character,
      message
    });
    res.status(200).json(response.data);
  } catch (error) {
    console.error('Error calling contextual adaptation API:', error);
    res.status(500).json({ error: 'Something went wrong!' });
  }
});

if (process.env.NODE_ENV !== 'production') {
  const PORT = process.env.PORT || 5000;
  app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
}

// For Vercel
module.exports = app;