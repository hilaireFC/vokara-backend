'use strict';

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const aiService = require('./services/aiService');

// ─── App Setup ────────────────────────────────────────────────────────────────

const app = express();

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// ─── File Upload Configuration ────────────────────────────────────────────────

// Ensure uploads directory exists
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => {
    const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(null, `vokara-${uniqueSuffix}${path.extname(file.originalname)}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB max
});

// ─── Cleanup Helper ───────────────────────────────────────────────────────────

/**
 * Safely delete a list of uploaded temp files after the request is handled.
 */
function cleanupFiles(...files) {
  for (const f of files) {
    if (f && f.path) {
      fs.unlink(f.path, (err) => {
        if (err) console.warn(`[Cleanup] Could not delete ${f.path}:`, err.message);
      });
    }
  }
}

// ─── Routes ───────────────────────────────────────────────────────────────────

// Health Check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Vokara Studio Backend is running',
    timestamp: new Date().toISOString(),
    routes: [
      'GET  /api/health',
      'POST /api/ai/context-reader',
      'POST /api/ai/roast',
      'POST /api/ai/edit-image',
      'POST /api/ai/voice-to-text',
      'POST /api/ai/remove-background',
      'POST /api/ai/face-swap',
      'POST /api/ai/meme-suggest',
      'POST /api/ai/text-to-speech',
      'POST /api/ai/create-sticker',
    ],
  });
});

// ─── 1. Context Reader ────────────────────────────────────────────────────────
// POST /api/ai/context-reader
// Body: { text: "..." }
// Returns: { emotion, meme_idea, suggested_template, caption }

app.post('/api/ai/context-reader', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({ error: 'Missing or empty "text" field in request body.' });
    }

    const result = await aiService.analyzeContext(text.trim());
    return res.json(result);
  } catch (error) {
    console.error('[context-reader] Error:', error.message);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 2. Roast My Photo ────────────────────────────────────────────────────────
// POST /api/ai/roast
// Form-data: image (file)
// Returns: { result: "..." }

app.post('/api/ai/roast', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Missing "image" file in multipart form.' });
    }

    const imageBuffer = fs.readFileSync(req.file.path);
    const mimeType = req.file.mimetype;
    const roastText = await aiService.roastPhoto(imageBuffer, mimeType);

    cleanupFiles(req.file);
    return res.json({ result: roastText });
  } catch (error) {
    console.error('[roast] Error:', error.message);
    cleanupFiles(req.file);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 3. Semantic Image Edit ───────────────────────────────────────────────────
// POST /api/ai/edit-image
// Form-data: image (file), instruction (text)
// Returns: { image: "<base64>" }

app.post('/api/ai/edit-image', upload.single('image'), async (req, res) => {
  try {
    const { instruction } = req.body;
    if (!req.file) {
      return res.status(400).json({ error: 'Missing "image" file in multipart form.' });
    }
    if (!instruction || !instruction.trim()) {
      cleanupFiles(req.file);
      return res.status(400).json({ error: 'Missing "instruction" field.' });
    }

    const imageBuffer = fs.readFileSync(req.file.path);
    const base64Image = await aiService.semanticEditImage(imageBuffer, instruction.trim());

    cleanupFiles(req.file);
    return res.json({ image: base64Image });
  } catch (error) {
    console.error('[edit-image] Error:', error.message);
    cleanupFiles(req.file);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 4. Voice Transcription ───────────────────────────────────────────────────
// POST /api/ai/voice-to-text
// Form-data: audio (file – m4a/wav/mp3/ogg)
// Returns: { text: "..." }

app.post('/api/ai/voice-to-text', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Missing "audio" file in multipart form.' });
    }

    const audioBuffer = fs.readFileSync(req.file.path);
    const mimeType = req.file.mimetype || 'audio/mpeg';
    // Pass filePath so Groq Whisper can stream the file
    const transcribedText = await aiService.voiceToText(audioBuffer, mimeType, req.file.path);

    cleanupFiles(req.file);
    return res.json({ text: transcribedText });
  } catch (error) {
    console.error('[voice-to-text] Error:', error.message);
    cleanupFiles(req.file);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 5. Background Removal / Magic Eraser ─────────────────────────────────────
// POST /api/ai/remove-background
// Form-data: image (file)
// Returns: { image: "<base64>" }

app.post('/api/ai/remove-background', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Missing "image" file in multipart form.' });
    }

    const imageBuffer = fs.readFileSync(req.file.path);
    const base64Image = await aiService.removeBackground(imageBuffer);

    cleanupFiles(req.file);
    return res.json({ image: base64Image });
  } catch (error) {
    console.error('[remove-background] Error:', error.message);
    cleanupFiles(req.file);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 6. Face Swap ─────────────────────────────────────────────────────────────
// POST /api/ai/face-swap
// Form-data: source (file – the face/sticker), target (file – the target photo)
// Returns: { image: "<base64>" }

app.post('/api/ai/face-swap', upload.fields([
  { name: 'source', maxCount: 1 },
  { name: 'target', maxCount: 1 },
]), async (req, res) => {
  const sourceFile = req.files?.source?.[0];
  const targetFile = req.files?.target?.[0];

  try {
    if (!sourceFile || !targetFile) {
      return res.status(400).json({
        error: 'Both "source" (face sticker) and "target" (destination photo) files are required.',
      });
    }

    if (!process.env.REPLICATE_API_TOKEN) {
      return res.status(503).json({
        error: 'REPLICATE_API_TOKEN is not configured. Please add your Replicate API token to .env.',
      });
    }

    const sourceBuffer = fs.readFileSync(sourceFile.path);
    const targetBuffer = fs.readFileSync(targetFile.path);

    const base64Image = await aiService.faceSwap(
      sourceBuffer,
      targetBuffer,
      sourceFile.mimetype,
      targetFile.mimetype
    );

    cleanupFiles(sourceFile, targetFile);
    return res.json({ image: base64Image });
  } catch (error) {
    console.error('[face-swap] Error:', error.message);
    cleanupFiles(sourceFile, targetFile);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 7. Meme Template Suggester ───────────────────────────────────────────────
// POST /api/ai/meme-suggest
// Body: { text: "description of situation" }
// Returns: { template: "drake", topText: "...", bottomText: "..." }

app.post('/api/ai/meme-suggest', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({ error: 'Missing or empty "text" field.' });
    }

    const result = await aiService.suggestMeme(text.trim());
    return res.json(result);
  } catch (error) {
    console.error('[meme-suggest] Error:', error.message);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 8. Text-to-Speech ────────────────────────────────────────────────────────
// POST /api/ai/text-to-speech
// Body: { text: "...", voice: "energetic" }
// Returns: { audio: "<base64>", mimeType: "audio/wav" }

app.post('/api/ai/text-to-speech', async (req, res) => {
  try {
    const { text, voice = 'energetic' } = req.body;
    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({ error: 'Missing or empty "text" field.' });
    }

    const base64Audio = await aiService.textToSpeech(text.trim(), voice);
    return res.json({ audio: base64Audio, mimeType: 'audio/wav' });
  } catch (error) {
    console.error('[text-to-speech] Error:', error.message);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 9. WhatsApp Sticker Creator ──────────────────────────────────────────────
// POST /api/ai/create-sticker
// Form-data: image (file)
// Returns: { sticker: "<base64>", mimeType: "image/webp" }

app.post('/api/ai/create-sticker', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Missing "image" file in multipart form.' });
    }

    const imageBuffer = fs.readFileSync(req.file.path);
    const base64Sticker = await aiService.createSticker(imageBuffer);

    cleanupFiles(req.file);
    return res.json({
      sticker: base64Sticker,
      mimeType: 'image/webp',
      format: 'webp',
      size: '512x512',
    });
  } catch (error) {
    console.error('[create-sticker] Error:', error.message);
    cleanupFiles(req.file);
    return res.status(500).json({ error: error.message });
  }
});

// ─── 404 Catch-all ────────────────────────────────────────────────────────────

app.use((req, res) => {
  res.status(404).json({
    error: 'Route not found',
    available: 'GET /api/health for a list of all routes',
  });
});

// ─── Global Error Handler ────────────────────────────────────────────────────

app.use((err, req, res, next) => {
  console.error('[Global Error]', err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

// ─── Start Server ─────────────────────────────────────────────────────────────

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n🎉 Vokara Studio Backend running on port ${PORT}`);
  console.log(`   Health check: http://localhost:${PORT}/api/health\n`);
});
