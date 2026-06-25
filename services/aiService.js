'use strict';

require('dotenv').config();
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { HfInference } = require('@huggingface/inference');
const Replicate = require('replicate');
const sharp = require('sharp');
const fs = require('fs');

// ─── SDK Initialization ────────────────────────────────────────────────────────

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN);

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN || '',
});

// ─── Helper ────────────────────────────────────────────────────────────────────

/**
 * Convert a Blob or ReadableStream returned by HF to a Buffer.
 */
async function blobToBuffer(blob) {
  if (blob instanceof Buffer) return blob;
  if (typeof blob.arrayBuffer === 'function') {
    const ab = await blob.arrayBuffer();
    return Buffer.from(ab);
  }
  // Node ReadableStream
  const chunks = [];
  for await (const chunk of blob) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks);
}

// ─── 1. Context Reader (Enhanced) ─────────────────────────────────────────────

/**
 * Analyze conversation text and return structured meme suggestion JSON.
 * Returns { emotion, meme_idea, suggested_template, caption }
 */
async function analyzeContext(text) {
  const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

  const prompt = `You are an expert meme analyst and internet culture specialist.
Analyze the following text/conversation and respond ONLY with a valid JSON object.

Text: "${text}"

Return this exact JSON structure (no markdown, no code fences, just raw JSON):
{
  "emotion": "<primary emotion detected, e.g. frustration, joy, confusion>",
  "meme_idea": "<a short punchy description of the meme concept that fits>",
  "suggested_template": "<name of a classic meme template, e.g. drake, distracted-boyfriend, this-is-fine, change-my-mind, galaxy-brain, gru-plan, two-buttons, expanding-brain>",
  "caption": "<a funny caption that fits the situation>"
}`;

  const result = await model.generateContent(prompt);
  const raw = result.response.text().trim();

  // Strip any accidental markdown code fences
  const cleaned = raw.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    // If Gemini didn't return valid JSON, return a safe fallback object
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
 * Roast a photo using Gemini vision.
 */
async function roastPhoto(imageBuffer, mimeType) {
  const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

  const prompt =
    "Analyse cette photo avec humour et fais un 'clash' (roast) drôle et bienveillant sur la personne ou la situation. Sois piquant, créatif, mais jamais méchant. Réponds en français, maximum 3 phrases percutantes.";

  const imagePart = {
    inlineData: {
      data: imageBuffer.toString('base64'),
      mimeType,
    },
  };

  const result = await model.generateContent([prompt, imagePart]);
  return result.response.text();
}

// ─── 3. Semantic Image Edit (InstructPix2Pix) ─────────────────────────────────

/**
 * Edit an image using InstructPix2Pix via Hugging Face Inference API.
 * Returns the edited image as a base64 string (data URI).
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

// ─── 4. Background Removal / Magic Eraser ─────────────────────────────────────

/**
 * Remove the background from an image using RMBG-1.4 on Hugging Face.
 * Returns the result image as base64.
 */
async function removeBackground(imageBuffer) {
  const blob = await hf.imageSegmentation({
    model: 'briaai/RMBG-1.4',
    data: new Blob([imageBuffer]),
  });

  // imageSegmentation returns an array of segments; we need the mask
  // If it returns a Blob directly (some HF endpoints), handle both.
  if (Array.isArray(blob)) {
    // Use the first segment mask
    const maskBlob = blob[0]?.mask || blob[0];
    const buf = await blobToBuffer(maskBlob);
    return buf.toString('base64');
  }

  const buf = await blobToBuffer(blob);
  return buf.toString('base64');
}

// ─── 5. Face Swap (Replicate) ─────────────────────────────────────────────────

/**
 * Swap a face from source image onto the target image using Replicate.
 * Returns the result image as base64.
 */
async function faceSwap(sourceImageBuffer, targetImageBuffer, sourceMimeType, targetMimeType) {
  if (!process.env.REPLICATE_API_TOKEN) {
    throw new Error('REPLICATE_API_TOKEN is not set. Please add it to your .env file.');
  }

  // Convert buffers to base64 data URIs for Replicate input
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

  // output is a URL string pointing to the result image
  let resultUrl = Array.isArray(output) ? output[0] : output;

  // Fetch the image and convert to base64
  const fetch = require('node-fetch');
  const response = await fetch(resultUrl);
  if (!response.ok) throw new Error(`Failed to fetch Replicate result: ${response.statusText}`);
  const arrayBuf = await response.arrayBuffer();
  return Buffer.from(arrayBuf).toString('base64');
}

// ─── 6. Meme Template Suggester ───────────────────────────────────────────────

/**
 * Analyze a text description and suggest a classic meme template.
 * Returns { template, topText, bottomText }
 */
async function suggestMeme(text) {
  const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

  const prompt = `You are a meme expert. Given this situation description, pick the PERFECT classic meme template.

Situation: "${text}"

Classic templates available: drake, distracted-boyfriend, this-is-fine, change-my-mind, galaxy-brain, gru-plan, two-buttons, expanding-brain, batman-slapping-robin, uno-reverse, crying-cat, coffin-dance, chad, surprised-pikachu, spongebob-mocking, woman-yelling-at-cat, hide-the-pain-harold, left-exit, stonks, stonks-bad, running-away-balloon, always-has-been, monkey-puppet, panik-kalm

Respond ONLY with raw JSON (no markdown fences):
{
  "template": "<template name from the list above>",
  "topText": "<text for the top of the meme>",
  "bottomText": "<text for the bottom of the meme>"
}`;

  const result = await model.generateContent(prompt);
  const raw = result.response.text().trim();
  const cleaned = raw.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    return {
      template: 'drake',
      topText: 'Could not parse the suggestion',
      bottomText: raw.slice(0, 120),
    };
  }
}

// ─── 7. Text-to-Speech ────────────────────────────────────────────────────────

/**
 * Convert text to speech using Facebook MMS-TTS on Hugging Face.
 * Returns audio as base64.
 */
async function textToSpeech(text, voice = 'energetic') {
  // Use espnet/kan-bayashi_ljspeech_vits - reliable HuggingFace TTS model
  const blob = await hf.textToSpeech({
    model: 'espnet/kan-bayashi_ljspeech_vits',
    inputs: text,
  });

  const buf = await blobToBuffer(blob);
  return buf.toString('base64');
}

// ─── 8. Voice Transcription ───────────────────────────────────────────────────

/**
 * Transcribe an audio file using Whisper-large-v3 on Hugging Face.
 * Returns the transcribed text string.
 */
async function voiceToText(audioBuffer, mimeType) {
  const result = await hf.automaticSpeechRecognition({
    model: 'openai/whisper-large-v3',
    data: new Blob([audioBuffer], { type: mimeType }),
  });

  return result.text;
}

// ─── 9. WhatsApp Sticker Creator ─────────────────────────────────────────────

/**
 * Convert an image to WhatsApp sticker format:
 * - 512x512 WebP
 * - < 100KB
 * Returns the sticker image as base64.
 */
async function createSticker(imageBuffer) {
  // Start with quality 90, decrease until under 100KB
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
  textToSpeech,
  voiceToText,
  createSticker,
};
