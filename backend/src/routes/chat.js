const express = require('express');
const router = express.Router();
const Character = require('../models/Character');
const { getFineTunedResponse } = require('../services/fineTuning');
const { getContextualAdaptationResponse } = require('../services/contextualAdaptation');

router.post('/', async (req, res) => {
  try {
    const { characterId, message } = req.body;
    const character = await Character.findById(characterId);

    let response;
    if (character.useFinetuning) {
      response = await getFineTunedResponse(characterId, message);
    } else {
      response = await getContextualAdaptationResponse(character, message);
    }

    res.json({ response });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get response' });
  }
});

module.exports = router;