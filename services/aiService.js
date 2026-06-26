'use strict';

require('dotenv').config();
const Groq = require('groq-sdk');
const { HfInference } = require('@huggingface/inference');
const Replicate = require('replicate');
const sharp = require('sharp');
const fs = require('fs');
const fetch = require('node-fetch');

// ─── SDK Initialization ────────────────────────────────────────────────────────

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN);
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN || '' });

// ─── Helper ────────────────────────────────────────────────────────────────────

async function blobToBuffer(blob) {
  if (blob instanceof Buffer) return blob;
  if (typeof blob.arrayBuffer === 'function') {
    const ab = await blob.arrayBuffer();
    return Buffer.from(ab);
  }
  const chunks = [];
  for await (const chunk of blob) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks);
}

// ─── 1. Context Reader ─────────────────────────────────────────────────────────

/**
 * Analyze conversation text and return structured meme suggestion JSON.
 * Uses Groq (llama3-70b) instead of Gemini.
 */
async function analyzeContext(text) {
  const completion = await groq.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      {
        role: 'system',
        content: 'You are an expert meme analyst. Respond ONLY with a raw JSON object, no markdown, no code fences.',
      },
      {
        role: 'user',
        content: `Analyze this text and return this exact JSON:
{"emotion": "<primary emotion>", "meme_idea": "<short punchy meme concept>", "suggested_template": "<classic template name: drake, distracted-boyfriend, this-is-fine, change-my-mind, galaxy-brain, gru-plan, two-buttons, expanding-brain, batman-slapping-robin, surprised-pikachu, spongebob-mocking, woman-yelling-at-cat>", "caption": "<funny caption>"}

Text: "${text}"`,
      },
    ],
    temperature: 0.7,
    max_tokens: 300,
  });

  const raw = completion.choices[0].message.content.trim();
  const cleaned = raw.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    return {
      emotion: 'unknown',
      meme_idea: raw,
      suggested_template: 'drake',
      caption: raw.slice(0, 120),
    };
  }
}

// ─── 2. Roast My Photo ─────────────────────────────────────────────────────────

/**
 * Roast a photo using Groq vision (llama-3.2-90b-vision-preview).
 */
async function roastPhoto(imageBuffer, mimeType) {
  const base64Image = imageBuffer.toString('base64');

  const completion = await groq.chat.completions.create({
    model: 'meta-llama/llama-4-scout-17b-16e-instruct',
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: {
              url: `data:${mimeType};base64,${base64Image}`,
            },
          },
          {
            type: 'text',
            text: "Analyse cette photo avec humour et fais un 'clash' (roast) drôle et bienveillant sur la personne ou la situation. Sois piquant, créatif, mais jamais méchant. Réponds en français, maximum 3 phrases percutantes.",
          },
        ],
      },
    ],
    temperature: 0.9,
    max_tokens: 300,
  });

  return completion.choices[0].message.content;
}

// ─── 3. Semantic Image Edit (InstructPix2Pix via HuggingFace) ─────────────────

/**
 * Edit an image using InstructPix2Pix.
 */
async function semanticEditImage(imageBuffer, instruction) {
  const blob = await hf.imageToImage({
    model: 'timbrooks/instruct-pix2pix',
    inputs: new Blob([imageBuffer]),
    parameters: {
      prompt: instruction,
      num_inference_steps: 20,
    },
  });

  const buf = await blobToBuffer(blob);
  return buf.toString('base64');
}

// ─── 4. Background Removal / Magic Eraser (remove.bg API) ─────────────────────

/**
 * Remove the background from an image using remove.bg API.
 * Returns the result image as base64.
 */
async function removeBackground(imageBuffer, filePath) {
  if (!process.env.REMOVE_BG_API_KEY) {
    throw new Error('REMOVE_BG_API_KEY is not set.');
  }

  const FormData = require('form-data');
  const form = new FormData();
  
  // Use file stream if path is provided, otherwise use buffer
  if (filePath && fs.existsSync(filePath)) {
    form.append('image_file', fs.createReadStream(filePath), {
      filename: 'image.jpg',
      contentType: 'image/jpeg',
    });
  } else {
    form.append('image_file', imageBuffer, {
      filename: 'image.jpg',
      contentType: 'image/jpeg',
    });
  }
  form.append('size', 'auto');

  const response = await fetch('https://api.remove.bg/v1.0/removebg', {
    method: 'POST',
    headers: {
      'X-Api-Key': process.env.REMOVE_BG_API_KEY,
      ...form.getHeaders(),
    },
    body: form,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`remove.bg error: ${response.status} - ${errText}`);
  }

  const arrayBuf = await response.arrayBuffer();
  return Buffer.from(arrayBuf).toString('base64');
}

// ─── 5. Face Swap (Replicate) ─────────────────────────────────────────────────

/**
 * Swap a face using Replicate.
 */
async function faceSwap(sourceImageBuffer, targetImageBuffer, sourceMimeType, targetMimeType) {
  if (!process.env.REPLICATE_API_TOKEN) {
    throw new Error('REPLICATE_API_TOKEN is not set. Please add it to your .env file.');
  }

  const sourceBase64 = `data:${sourceMimeType};base64,${sourceImageBuffer.toString('base64')}`;
  const targetBase64 = `data:${targetMimeType};base64,${targetImageBuffer.toString('base64')}`;

  const output = await replicate.run(
    'lucataco/faceswap:9a4298548422074c3f57258c5d544497838d060ab775d1a2d494a689f016bec5',
    {
      input: {
        target_image: targetBase64,
        swap_image: sourceBase64,
      },
    }
  );

  let resultUrl = Array.isArray(output) ? output[0] : output;

  const response = await fetch(resultUrl);
  if (!response.ok) throw new Error(`Failed to fetch Replicate result: ${response.statusText}`);
  const arrayBuf = await response.arrayBuffer();
  return Buffer.from(arrayBuf).toString('base64');
}

// ─── 6. Meme Template Suggester (Groq) ────────────────────────────────────────

/**
 * Analyze a text and suggest a meme template using Groq.
 */
async function suggestMeme(text) {
  const completion = await groq.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      {
        role: 'system',
        content: 'You are a meme expert. Respond ONLY with raw JSON, no markdown.',
      },
      {
        role: 'user',
        content: `Pick the PERFECT meme template for this situation:
"${text}"

Available: drake, distracted-boyfriend, this-is-fine, change-my-mind, galaxy-brain, gru-plan, two-buttons, expanding-brain, batman-slapping-robin, uno-reverse, crying-cat, coffin-dance, chad, surprised-pikachu, spongebob-mocking, woman-yelling-at-cat, hide-the-pain-harold, stonks, monkey-puppet, panik-kalm

Return this JSON:
{"template": "<name>", "topText": "<top text>", "bottomText": "<bottom text>"}`,
      },
    ],
    temperature: 0.8,
    max_tokens: 150,
  });

  const raw = completion.choices[0].message.content.trim();
  const cleaned = raw.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    return { template: 'drake', topText: 'Could not parse', bottomText: raw.slice(0, 120) };
  }
}

// ─── 7. Voice Transcription (Groq Whisper) ────────────────────────────────────

/**
 * Transcribe audio using Groq's Whisper (fastest Whisper available).
 */
async function voiceToText(audioBuffer, mimeType, filePath) {
  const transcription = await groq.audio.transcriptions.create({
    file: fs.createReadStream(filePath),
    model: 'whisper-large-v3',
    language: 'fr',
  });

  return transcription.text;
}

// ─── 8. WhatsApp Sticker Creator ─────────────────────────────────────────────

/**
 * Convert an image to WhatsApp sticker format (512x512 WebP < 100KB).
 */
async function createSticker(imageBuffer) {
  let quality = 90;
  let webpBuffer;

  do {
    webpBuffer = await sharp(imageBuffer)
      .resize(512, 512, {
        fit: 'contain',
        background: { r: 0, g: 0, b: 0, alpha: 0 },
      })
      .webp({ quality, lossless: false })
      .toBuffer();

    quality -= 10;
    if (quality < 10) break;
  } while (webpBuffer.length > 100 * 1024);

  return webpBuffer.toString('base64');
}

// ─── Exports ──────────────────────────────────────────────────────────────────

module.exports = {
  analyzeContext,
  roastPhoto,
  semanticEditImage,
  removeBackground,
  faceSwap,
  suggestMeme,
  voiceToText,
  createSticker,
};
