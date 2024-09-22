const mongoose = require('mongoose');

const CharacterSchema = new mongoose.Schema({
  name: String,
  tagline: String,
  description: String,
  greeting: String,
  isPublic: Boolean,
  useFinetuning: Boolean,
});

module.exports = mongoose.model('Character', CharacterSchema);