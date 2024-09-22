// backend/src/services/fineTuning.js

const { exec } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const { HfFolder, HfApi } = require('@huggingface/hub');

const REPO_OWNER = 'your-huggingface-username';
const BASE_MODEL = 'meta-llama/Llama-2-7b-chat-hf';

async function triggerFineTuning(characterId) {
  try {
    const character = await Character.findById(characterId);
    if (!character) {
      throw new Error('Character not found');
    }

    // Generate training data based on character description
    const trainingData = await generateTrainingData(character);

    // Save training data to a file
    const trainingDataPath = path.join(__dirname, `../../training_data_${characterId}.jsonl`);
    await fs.writeFile(trainingDataPath, JSON.stringify(trainingData));

    // Trigger fine-tuning process
    const scriptPath = path.join(__dirname, '../../../ai/fine_tuning.py');
    const command = `python ${scriptPath} --character_id ${characterId} --training_data ${trainingDataPath} --base_model ${BASE_MODEL}`;

    exec(command, async (error, stdout, stderr) => {
      if (error) {
        console.error(`Fine-tuning error: ${error.message}`);
        return;
      }
      if (stderr) {
        console.error(`Fine-tuning stderr: ${stderr}`);
        return;
      }

      console.log(`Fine-tuning stdout: ${stdout}`);

      // Upload fine-tuned model to Hugging Face
      const modelPath = path.join(__dirname, `../../fine_tuned_models/${characterId}`);
      await uploadToHuggingFace(characterId, modelPath);

      // Update character with fine-tuned model information
      character.fineTunedModelId = `${REPO_OWNER}/llama-2-7b-${characterId}`;
      await character.save();
    });
  } catch (error) {
    console.error('Error in triggerFineTuning:', error);
  }
}

async function generateTrainingData(character) {
  // Implement logic to generate training data based on character description
  // This could involve using GPT-4 to generate examples or processing existing data
  // Return an array of { prompt, completion } objects
}

async function uploadToHuggingFace(characterId, modelPath) {
  const api = new HfApi();
  const token = HfFolder.getToken();

  if (!token) {
    throw new Error('Hugging Face token not found. Please log in using the Hugging Face CLI.');
  }

  const repoName = `llama-2-7b-${characterId}`;

  try {
    await api.createRepo({
      token,
      name: repoName,
      organization: REPO_OWNER,
      private: true,
    });

    await api.uploadFolder({
      token,
      repoId: `${REPO_OWNER}/${repoName}`,
      folderPath: modelPath,
    });

    console.log(`Model uploaded successfully to ${REPO_OWNER}/${repoName}`);
  } catch (error) {
    console.error('Error uploading to Hugging Face:', error);
  }
}

module.exports = { triggerFineTuning };