const express = require('express');
const router = express.Router();
const Character = require('../models/Character');
const { triggerFineTuning } = require('../services/fineTuning');

router.post('/', async (req, res) => {
  try {
    const character = new Character(req.body);
    await character.save();

    let fineTuningStatus = null;

    if (character.useFinetuning) {
      try {
        await triggerFineTuning(character._id);
        fineTuningStatus = 'Fine-tuning process initiated successfully';
      } catch (fineTuningError) {
        console.error('Fine-tuning error:', fineTuningError);
        fineTuningStatus = 'Fine-tuning process failed to start';
      }
    }

    res.status(201).json({
      message: 'Character created successfully',
      character,
      fineTuningStatus
    });
  } catch (error) {
    console.error('Error creating character:', error);
    res.status(500).json({ error: 'Failed to create character', details: error.message });
  }
});

// GET route to fetch a specific character
router.get('/:id', async (req, res) => {
  try {
    const character = await Character.findById(req.params.id);
    if (!character) {
      return res.status(404).json({ error: 'Character not found' });
    }
    res.json(character);
  } catch (error) {
    console.error('Error fetching character:', error);
    res.status(500).json({ error: 'Failed to fetch character', details: error.message });
  }
});

// GET route to fetch all characters
router.get('/', async (req, res) => {
  try {
    const characters = await Character.find();
    res.json(characters);
  } catch (error) {
    console.error('Error fetching characters:', error);
    res.status(500).json({ error: 'Failed to fetch characters', details: error.message });
  }
});

module.exports = router;