/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import dotenv from "dotenv";

import { ScratchTransformer, WordTokenizer } from "./model/transformer";
import { STARTUP_QUOTES } from "./model/dataset";
import { GoogleGenAI } from "@google/genai";

// Load environment variables
dotenv.config();
// Also load .env.local for local development overrides
dotenv.config({ path: ".env.local" });

const app = express();
const PORT = 3000;

// ============================================================================
// CORS middleware for local development flexibility
// ============================================================================
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") {
    return res.sendStatus(200);
  }
  next();
});

// Use Express middleware to decode JSON with size limits
app.use(express.json({ limit: "1mb" }));

// ============================================================================
// Initialize Transformer Engine
// ============================================================================
console.log("══════════════════════════════════════════════════");
console.log("  Initializing NumPyGPT Scratch Transformer...");
console.log("══════════════════════════════════════════════════");

const tokenizer = new WordTokenizer(STARTUP_QUOTES);
const transformer = new ScratchTransformer(tokenizer.vocabSize);

console.log(`  Vocab Size: ${tokenizer.vocabSize} words/tokens`);
console.log(`  Model Dimensions: d_model=${transformer.dModel}, d_head=${transformer.dHead}, d_ff=${transformer.dFf}`);
console.log(`  Max Context Length: ${transformer.maxSeqLen} tokens`);
console.log(`  Total Parameters: ${transformer.countParameters().toLocaleString()}`);

// Global training metrics tracking
let globalTrainingSteps = 0;
let lastRecordedLoss = 4.0;
const trainingHistory: { step: number; loss: number }[] = [];

// Track server start time for uptime calculations
const serverStartTime = Date.now();

// ============================================================================
// Pre-Training with Learning Rate Warmup Schedule
// ============================================================================
function runInitialPretraining(steps = 2000) {
  console.log(`\n  Pre-training for ${steps} steps with LR warmup...`);
  const startTime = Date.now();

  let totalLoss = 0;
  let count = 0;
  const warmupSteps = 50;

  for (let step = 0; step < steps; step++) {
    const quote = STARTUP_QUOTES[step % STARTUP_QUOTES.length];
    const encoded = tokenizer.encode(quote);
    if (encoded.length < 3) continue;

    const sliceLen = Math.min(encoded.length - 1, transformer.maxSeqLen);
    const inputs = encoded.slice(0, sliceLen);
    const targets = encoded.slice(1, sliceLen + 1);

    // Learning rate warmup schedule: linear warmup then cosine decay
    let lr: number;
    if (step < warmupSteps) {
      // Linear warmup from 0.001 to 0.08
      lr = 0.001 + (0.08 - 0.001) * (step / warmupSteps);
    } else {
      // Cosine annealing decay
      const progress = (step - warmupSteps) / (steps - warmupSteps);
      lr = 0.01 + 0.5 * (0.08 - 0.01) * (1 + Math.cos(Math.PI * progress));
    }

    const result = transformer.trainStep(inputs, targets, lr);
    totalLoss += result.loss;
    count++;

    // Record loss at regular intervals for the loss chart
    if (step % 25 === 0 || step === steps - 1) {
      const avgLoss = count > 0 ? totalLoss / count : 4.0;
      trainingHistory.push({ step: globalTrainingSteps + step, loss: avgLoss });
      totalLoss = 0;
      count = 0;
    }
  }

  lastRecordedLoss = trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].loss : 4.0;
  globalTrainingSteps += steps;

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`  ✓ Pre-training completed in ${elapsed}s`);
  console.log(`  ✓ Final Loss: ${lastRecordedLoss.toFixed(4)}`);
  console.log(`  ✓ Total Steps: ${globalTrainingSteps}`);
}

// Perform initial training
runInitialPretraining(2000);

// ============================================================================
// Global Error Handler Middleware
// ============================================================================
function asyncHandler(fn: (req: express.Request, res: express.Response) => Promise<any>) {
  return (req: express.Request, res: express.Response, next: express.NextFunction) => {
    Promise.resolve(fn(req, res)).catch(next);
  };
}

// ============================================================================
// API ENDPOINTS — Original (fixed & improved)
// ============================================================================

// API health endpoint
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    vocabSize: tokenizer.vocabSize,
    trainingSteps: globalTrainingSteps,
    currentLoss: lastRecordedLoss,
    totalParams: transformer.countParameters(),
    modelConfig: {
      dModel: transformer.dModel,
      dHead: transformer.dHead,
      dFf: transformer.dFf,
      maxSeqLen: transformer.maxSeqLen,
    },
    uptimeSeconds: Math.floor((Date.now() - serverStartTime) / 1000),
  });
});

// GET: Vocabulary metadata
app.get("/api/vocab", (req, res) => {
  res.json({
    vocabulary: tokenizer.vocabulary,
    charToIdx: tokenizer.tokenToIdx, // Keep field name for frontend compatibility
    idxToChar: tokenizer.idxToToken,
  });
});

// GET: Training dataset
app.get("/api/dataset", (req, res) => {
  res.json({
    quotes: STARTUP_QUOTES,
    stats: {
      totalQuotes: STARTUP_QUOTES.length,
      modelDim: transformer.dModel,
      headDim: transformer.dHead,
      ffDim: transformer.dFf,
      maxSeqLen: transformer.maxSeqLen,
    }
  });
});

// POST: Run manual training cycles live (Triggered from interactive dashboard)
app.post("/api/train", (req, res) => {
  const epochs = Math.min(Math.max(Number(req.body.epochs || 10), 1), 200);
  const userLearningRate = Math.min(Math.max(Number(req.body.lr || 0.05), 0.0001), 0.5);

  let runLoss = 0;
  let actualRuns = 0;

  for (let e = 0; e < epochs; e++) {
    // Pick a random quote
    const randQuoteIndex = Math.floor(Math.random() * STARTUP_QUOTES.length);
    const quote = STARTUP_QUOTES[randQuoteIndex];
    const encoded = tokenizer.encode(quote);
    
    if (encoded.length < 3) continue;

    const sliceLen = Math.min(encoded.length - 1, transformer.maxSeqLen);
    const inputs = encoded.slice(0, sliceLen);
    const targets = encoded.slice(1, sliceLen + 1);

    const result = transformer.trainStep(inputs, targets, userLearningRate);
    runLoss += result.loss;
    actualRuns++;
    globalTrainingSteps++;
  }

  if (actualRuns > 0) {
    lastRecordedLoss = runLoss / actualRuns;
    trainingHistory.push({ step: globalTrainingSteps, loss: lastRecordedLoss });
  }

  res.json({
    success: true,
    epochsRun: actualRuns,
    currentLoss: lastRecordedLoss,
    totalSteps: globalTrainingSteps,
    history: trainingHistory.slice(-50), // Send last 50 data points for graphing
  });
});

// POST: Generate text with live internals tracing
app.post("/api/generate", (req, res) => {
  const prompt = req.body.prompt || "Move";
  const maxLength = Math.min(Math.max(Number(req.body.maxLength || 50), 5), 150);
  const temp = Math.min(Math.max(Number(req.body.temp || 0.75), 0.05), 2.0);

  if (!prompt || typeof prompt !== "string") {
    return res.status(400).json({ success: false, error: "Prompt must be a non-empty string." });
  }

  if (prompt.length > 100) {
    return res.status(400).json({ success: false, error: "Prompt too long. Maximum 100 characters." });
  }

  const promptIds = tokenizer.encode(prompt);
  if (promptIds.length > transformer.maxSeqLen) {
    return res.status(400).json({
      success: false,
      error: `Prompt is too long! Max context is ${transformer.maxSeqLen} words.`
    });
  }

  const startTime = Date.now();

  // Run generation loop on Scratch Transformer
  const outputIds = transformer.generate(promptIds, maxLength, temp);
  const fullText = tokenizer.decode(outputIds);

  const inferenceTime = Date.now() - startTime;

  // Also perform a single forward pass with the full text or truncated prompt to fetch
  // rich activation maps (attention matrices, query, key logs) for rendering the visualizer!
  const visualContextIds = outputIds.slice(0, Math.min(outputIds.length, 16)); // Limit heatmap size for performance
  const { cache } = transformer.forward(visualContextIds);
  
  // Format attention matrix for heatmap
  const attentionMatrix = cache.A.data; 
  const tokensDecoded = visualContextIds.map(id => tokenizer.idxToToken[id] || "");

  res.json({
    success: true,
    promptUsed: prompt,
    generatedText: fullText,
    tokens: tokensDecoded,
    attentionMatrix: attentionMatrix,
    activations: {
      Q: cache.Q.data,
      K: cache.K.data,
      V: cache.V.data,
    },
    metadata: {
      inferenceTimeMs: inferenceTime,
      tokensGenerated: outputIds.length - promptIds.length,
      totalTokens: outputIds.length,
      temperature: temp,
    }
  });
});

// POST: Scribe AI / Teacher Integration using Gemini to explain the Transformer Math & Code base!
app.post("/api/explain", asyncHandler(async (req, res) => {
  const concept = req.body.concept || "Self-Attention Mechanism";
  
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
    return res.json({
      success: true,
      explanation: "### Gemini API Key Missing\n\nYou can set up your API key by creating a `.env.local` file with `GEMINI_API_KEY=your_key_here`.\n\nHere is a local explanation of **" + concept + "** in the meantime:\n\n1. **Query (Q), Key (K), and Value (V)** projections map text embeddings into dimensional spaces.\n2. **Scaled dot-product attention** computes similarity via $(Q \\cdot K^T) / \\sqrt{d_k}$.\n3. **Softmax** normalizes similarity scores to create a causal distribution map.\n4. **Matrix math** multiplies this weight map against Values ($V$) to acquire context-informed token predictions."
    });
  }

  const ai = new GoogleGenAI({
    apiKey: apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      }
    }
  });

  const promptText = `
You are an expert AI Educator and Technical Interview Coach.
Explain the following concept: "${concept}" specifically in the context of a "Transformer built from scratch in pure NumPy / TypeScript".
Refer to the typical math involved:
- Token & Positional Embeddings
- Query, Key, Value matrix projections (W_q, W_k, W_v)
- Causal Masking (for text generation/GPT models)
- Softmax over raw attention weights: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
- Backpropagation through the self-attention formula.

Provide a highly professional, clear, engaging explanation tailored for a tech recruiter or engineering manager.
Format the explanation beautifully in clean Markdown with distinct headers and LaTeX equations where helpful. Keep it concise (about 300-400 words) so it fits elegantly in the UI view.
`;

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: promptText,
  });

  res.json({
    success: true,
    explanation: response.text
  });
}));

// ============================================================================
// NEW API ENDPOINTS — Enhanced Features for National-Level Project
// ============================================================================

// GET: Model weight statistics per layer — great for visualizing weight distributions
app.get("/api/weights", (req, res) => {
  const weightStats: Record<string, ReturnType<typeof transformer.W_token.stats> & { shape: string }> = {};

  const layers: [string, typeof transformer.W_token][] = [
    ["W_token (Token Embeddings)", transformer.W_token],
    ["W_pos (Positional Encoding)", transformer.W_pos],
    ["LN1_gamma (LayerNorm1 Scale)", transformer.LN1_gamma],
    ["LN1_beta (LayerNorm1 Shift)", transformer.LN1_beta],
    ["W_q (Query Projection)", transformer.W_q],
    ["W_k (Key Projection)", transformer.W_k],
    ["W_v (Value Projection)", transformer.W_v],
    ["W_proj (Attention Output)", transformer.W_proj],
    ["LN2_gamma (LayerNorm2 Scale)", transformer.LN2_gamma],
    ["LN2_beta (LayerNorm2 Shift)", transformer.LN2_beta],
    ["W_ff1 (FFN Layer 1)", transformer.W_ff1],
    ["B_ff1 (FFN Bias 1)", transformer.B_ff1],
    ["W_ff2 (FFN Layer 2)", transformer.W_ff2],
    ["B_ff2 (FFN Bias 2)", transformer.B_ff2],
    ["W_out (Unembedding)", transformer.W_out],
    ["B_out (Output Bias)", transformer.B_out],
  ];

  for (const [name, matrix] of layers) {
    const stats = matrix.stats();
    weightStats[name] = {
      ...stats,
      shape: `[${matrix.rows} x ${matrix.cols}]`,
    };
  }

  res.json({
    success: true,
    totalParameters: transformer.countParameters(),
    layers: weightStats,
  });
});

// POST: Detailed forward pass trace — step-by-step computation internals
app.post("/api/forward-trace", (req, res) => {
  const text = req.body.text || "Move fast";

  if (!text || typeof text !== "string") {
    return res.status(400).json({ success: false, error: "Text must be a non-empty string." });
  }

  const tokenIds = tokenizer.encode(text.slice(0, transformer.maxSeqLen));
  if (tokenIds.length < 1) {
    return res.status(400).json({ success: false, error: "Text produced no valid tokens." });
  }

  const startTime = Date.now();
  const { logits, cache } = transformer.forward(tokenIds);
  const forwardTimeMs = Date.now() - startTime;

  // Compute predictions for each position
  const predictions: { position: number; inputChar: string; topPredictions: { char: string; probability: number }[] }[] = [];

  for (let t = 0; t < tokenIds.length; t++) {
    const tokenLogits = logits.data[t];
    const maxL = Math.max(...tokenLogits);
    const exps = tokenLogits.map(v => Math.exp(v - maxL));
    const sumExps = exps.reduce((a, b) => a + b, 0) || 1e-9;
    const probs = exps.map(v => v / sumExps);

    const indexed = probs.map((p, idx) => ({ char: tokenizer.idxToToken[idx] || "?", probability: p }));
    indexed.sort((a, b) => b.probability - a.probability);

    predictions.push({
      position: t,
      inputChar: tokenizer.idxToToken[tokenIds[t]] || "?",
      topPredictions: indexed.slice(0, 5),
    });
  }

  res.json({
    success: true,
    input: text.slice(0, transformer.maxSeqLen),
    tokenIds,
    forwardTimeMs,
    sequenceLength: tokenIds.length,
    trace: {
      embeddingShape: `[${cache.E_tok.rows} x ${cache.E_tok.cols}]`,
      queryShape: `[${cache.Q.rows} x ${cache.Q.cols}]`,
      keyShape: `[${cache.K.rows} x ${cache.K.cols}]`,
      valueShape: `[${cache.V.rows} x ${cache.V.cols}]`,
      attentionShape: `[${cache.A.rows} x ${cache.A.cols}]`,
      ffHiddenShape: `[${cache.FF_h1.rows} x ${cache.FF_h1.cols}]`,
      logitsShape: `[${cache.Logits.rows} x ${cache.Logits.cols}]`,
    },
    attentionWeights: cache.A.data,
    predictions,
  });
});

// POST: Tokenize input text — understand tokenizer behavior
app.post("/api/tokenize", (req, res) => {
  const text = req.body.text || "";

  if (typeof text !== "string") {
    return res.status(400).json({ success: false, error: "Text must be a string." });
  }

  const tokenIds = tokenizer.encode(text);
  const decoded = tokenIds.map(id => ({
    id,
    char: tokenizer.idxToToken[id] || "<unk>",
    isSpecial: tokenizer.idxToToken[id] === "<pad>" || tokenizer.idxToToken[id] === "<unk>",
  }));

  res.json({
    success: true,
    text,
    tokenCount: tokenIds.length,
    tokenIds,
    tokens: decoded,
    vocabSize: tokenizer.vocabSize,
    vocabulary: tokenizer.vocabulary,
  });
});

// GET: Model performance benchmark
app.get("/api/benchmark", (req, res) => {
  // Run a benchmark forward pass
  const testQuote = STARTUP_QUOTES[0];
  const testIds = tokenizer.encode(testQuote).slice(0, transformer.maxSeqLen);

  const trials = 5;
  const times: number[] = [];

  for (let i = 0; i < trials; i++) {
    const start = Date.now();
    transformer.forward(testIds);
    times.push(Date.now() - start);
  }

  const avgForwardMs = times.reduce((a, b) => a + b, 0) / trials;

  // Estimate FLOPs (rough approximation)
  const seqLen = testIds.length;
  const d = transformer.dModel;
  const dh = transformer.dHead;
  const df = transformer.dFf;
  const V = transformer.vocabSize;

  // Major matrix multiplications approximate FLOPs
  const embeddingOps = seqLen * d * 2;                    // token + positional lookup
  const qkvOps = 3 * seqLen * d * dh * 2;                 // Q, K, V projections
  const attentionOps = seqLen * seqLen * dh * 2;           // Q * K^T
  const contextOps = seqLen * seqLen * dh * 2;             // A * V
  const projOps = seqLen * dh * d * 2;                     // O * W_proj
  const ffnOps = seqLen * d * df * 2 + seqLen * df * d * 2; // Two FF layers
  const outputOps = seqLen * d * V * 2;                    // Unembedding
  const totalFLOPs = embeddingOps + qkvOps + attentionOps + contextOps + projOps + ffnOps + outputOps;

  // Memory estimate (parameters * 8 bytes for float64)
  const memoryBytes = transformer.countParameters() * 8;

  res.json({
    success: true,
    benchmark: {
      testSequenceLength: seqLen,
      averageForwardPassMs: Math.round(avgForwardMs * 100) / 100,
      trials,
      estimatedFLOPs: totalFLOPs,
      estimatedFLOPsFormatted: totalFLOPs > 1e6 ? `${(totalFLOPs / 1e6).toFixed(2)}M` : `${(totalFLOPs / 1e3).toFixed(2)}K`,
      tokensPerSecond: Math.round(1000 / (avgForwardMs || 1)),
    },
    model: {
      totalParameters: transformer.countParameters(),
      totalParametersFormatted: `${(transformer.countParameters() / 1000).toFixed(1)}K`,
      memoryEstimateBytes: memoryBytes,
      memoryEstimateFormatted: `${(memoryBytes / 1024).toFixed(1)} KB`,
      architecture: "Single-Block Causal GPT (Pre-LN)",
      dimensions: {
        dModel: transformer.dModel,
        dHead: transformer.dHead,
        dFf: transformer.dFf,
        maxSeqLen: transformer.maxSeqLen,
        vocabSize: transformer.vocabSize,
      },
    },
    training: {
      totalSteps: globalTrainingSteps,
      currentLoss: lastRecordedLoss,
      datasetSize: STARTUP_QUOTES.length,
      optimizer: "SGD with Gradient Clipping",
    },
  });
});

// POST: Reset model weights to fresh initialization
app.post("/api/reset", (req, res) => {
  const confirmReset = req.body.confirm === true;

  if (!confirmReset) {
    return res.status(400).json({
      success: false,
      error: "You must send { confirm: true } to reset the model. This action is irreversible.",
    });
  }

  // Re-initialize the transformer in place
  const freshTransformer = new ScratchTransformer(tokenizer.vocabSize);
  
  // Copy all weights from fresh instance
  transformer.W_token = freshTransformer.W_token;
  transformer.W_pos = freshTransformer.W_pos;
  transformer.LN1_gamma = freshTransformer.LN1_gamma;
  transformer.LN1_beta = freshTransformer.LN1_beta;
  transformer.LN2_gamma = freshTransformer.LN2_gamma;
  transformer.LN2_beta = freshTransformer.LN2_beta;
  transformer.W_q = freshTransformer.W_q;
  transformer.W_k = freshTransformer.W_k;
  transformer.W_v = freshTransformer.W_v;
  transformer.W_proj = freshTransformer.W_proj;
  transformer.W_ff1 = freshTransformer.W_ff1;
  transformer.B_ff1 = freshTransformer.B_ff1;
  transformer.W_ff2 = freshTransformer.W_ff2;
  transformer.B_ff2 = freshTransformer.B_ff2;
  transformer.W_out = freshTransformer.W_out;
  transformer.B_out = freshTransformer.B_out;

  // Reset training state
  globalTrainingSteps = 0;
  lastRecordedLoss = 4.0;
  trainingHistory.length = 0;
  trainingHistory.push({ step: 0, loss: 4.0 });

  console.log("[Reset] Model weights re-initialized to random state.");

  res.json({
    success: true,
    message: "Model weights have been reset to fresh random initialization. All training progress cleared.",
    currentLoss: lastRecordedLoss,
    totalSteps: globalTrainingSteps,
  });
});

// GET: Export full model weights as JSON (downloadable)
app.get("/api/export-weights", (req, res) => {
  const exportData = {
    metadata: {
      exportedAt: new Date().toISOString(),
      modelType: "ScratchTransformer",
      architecture: "Single-Block Causal GPT (Pre-LN)",
      vocabSize: transformer.vocabSize,
      dModel: transformer.dModel,
      dHead: transformer.dHead,
      dFf: transformer.dFf,
      maxSeqLen: transformer.maxSeqLen,
      totalParameters: transformer.countParameters(),
      trainingSteps: globalTrainingSteps,
      finalLoss: lastRecordedLoss,
    },
    tokenizer: {
      vocabulary: tokenizer.vocabulary,
      charToIdx: tokenizer.tokenToIdx,
    },
    weights: {
      W_token: transformer.W_token.data,
      W_pos: transformer.W_pos.data,
      LN1_gamma: transformer.LN1_gamma.data,
      LN1_beta: transformer.LN1_beta.data,
      W_q: transformer.W_q.data,
      W_k: transformer.W_k.data,
      W_v: transformer.W_v.data,
      W_proj: transformer.W_proj.data,
      LN2_gamma: transformer.LN2_gamma.data,
      LN2_beta: transformer.LN2_beta.data,
      W_ff1: transformer.W_ff1.data,
      B_ff1: transformer.B_ff1.data,
      W_ff2: transformer.W_ff2.data,
      B_ff2: transformer.B_ff2.data,
      W_out: transformer.W_out.data,
      B_out: transformer.B_out.data,
    },
    trainingHistory,
  };

  res.setHeader("Content-Type", "application/json");
  res.setHeader("Content-Disposition", `attachment; filename="numpygpt-weights-${Date.now()}.json"`);
  res.json(exportData);
});

// ============================================================================
// Error handling middleware
// ============================================================================
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error("[Server Error]", err.message);
  res.status(500).json({
    success: false,
    error: process.env.NODE_ENV === "production"
      ? "Internal server error. Please try again."
      : err.message,
  });
});

// ============================================================================
// Start Express API Server
// ============================================================================
app.listen(PORT, "0.0.0.0", () => {
  console.log(`\n═══════════════════════════════════════════════════`);
  console.log(`  🚀 NumPyGPT Backend API Running!`);
  console.log(`  📊 API Health: http://localhost:${PORT}/api/health`);
  console.log(`  🔬 Benchmark: http://localhost:${PORT}/api/benchmark`);
  console.log(`  🧠 Weights:   http://localhost:${PORT}/api/weights`);
  console.log(`  ⚙️  Environment: ${process.env.NODE_ENV || "development"}`);
  console.log(`═══════════════════════════════════════════════════\n`);
});
