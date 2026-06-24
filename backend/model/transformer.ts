/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

// NumPyJS: A simple NDArray/Matrix library built from scratch in TypeScript
// containing all matrix operations required for a Transformer's forward and backward passes.
export class Matrix {
  public data: number[][];
  public rows: number;
  public cols: number;

  constructor(rows: number, cols: number, values?: number[] | number[][], populateRandom = false) {
    this.rows = rows;
    this.cols = cols;
    
    if (values) {
      if (Array.isArray(values[0])) {
        this.data = values as number[][];
      } else {
        const flat = values as number[];
        this.data = [];
        for (let i = 0; i < rows; i++) {
          this.data.push(flat.slice(i * cols, (i + 1) * cols));
        }
      }
    } else {
      this.data = Array.from({ length: rows }, () => new Array(cols).fill(0));
      if (populateRandom) {
        // Xavier/Glorot Initialization
        const limit = Math.sqrt(6 / (rows + cols));
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            this.data[i][j] = Math.random() * 2 * limit - limit;
          }
        }
      }
    }
  }

  // Factory: create matrix with He initialization (better for ReLU layers)
  public static heInit(rows: number, cols: number): Matrix {
    const m = new Matrix(rows, cols);
    const std = Math.sqrt(2.0 / rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        // Box-Muller transform for normal distribution
        const u1 = Math.random() || 1e-10;
        const u2 = Math.random();
        m.data[i][j] = std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
    }
    return m;
  }

  // Element-wise copy
  public copy(): Matrix {
    const values = this.data.map(row => [...row]);
    return new Matrix(this.rows, this.cols, values);
  }

  // Transpose matrix
  public transpose(): Matrix {
    const result = new Matrix(this.cols, this.rows);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        result.data[c][r] = this.data[r][c];
      }
    }
    return result;
  }

  // Matrix multiplication: A * B
  public static dot(A: Matrix, B: Matrix): Matrix {
    if (A.cols !== B.rows) {
      throw new Error(`Matrix multiplication mismatch: A cols (${A.cols}) !== B rows (${B.rows})`);
    }
    const result = new Matrix(A.rows, B.cols);
    for (let r = 0; r < A.rows; r++) {
      for (let c = 0; c < B.cols; c++) {
        let sum = 0;
        for (let k = 0; k < A.cols; k++) {
          sum += A.data[r][k] * B.data[k][c];
        }
        result.data[r][c] = sum;
      }
    }
    return result;
  }

  // Add another matrix with simple row-wise broadcasting
  public add(B: Matrix): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        // Broadcast B if it has 1 row
        const bVal = B.rows === 1 ? B.data[0][c] : B.data[r][c];
        result.data[r][c] = this.data[r][c] + bVal;
      }
    }
    return result;
  }

  // Subtract another matrix with broadcasting
  public subtract(B: Matrix): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const bVal = B.rows === 1 ? B.data[0][c] : B.data[r][c];
        result.data[r][c] = this.data[r][c] - bVal;
      }
    }
    return result;
  }

  // Element-wise multiplication
  public multiply(B: Matrix): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const bVal = B.rows === 1 ? B.data[0][c] : B.data[r][c];
        result.data[r][c] = this.data[r][c] * bVal;
      }
    }
    return result;
  }

  // Scale matrix by a number
  public scale(scalar: number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        result.data[r][c] = this.data[r][c] * scalar;
      }
    }
    return result;
  }

  // Apply row-wise softmax
  public softmax(): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) {
      const row = this.data[r];
      const maxVal = Math.max(...row); // Numerical stability trick
      const exps = row.map(v => Math.exp(v - maxVal));
      const sumExps = exps.reduce((acc, curr) => acc + curr, 0);
      for (let c = 0; c < this.cols; c++) {
        result.data[r][c] = exps[c] / (sumExps || 1e-9);
      }
    }
    return result;
  }

  // Realize Rectified Linear Unit (ReLU)
  public relu(): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        result.data[r][c] = Math.max(0, this.data[r][c]);
      }
    }
    return result;
  }

  // Derivative of ReLU for backprop
  public reluDerivative(upstreamGrad: Matrix): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        result.data[r][c] = this.data[r][c] > 0 ? upstreamGrad.data[r][c] : 0;
      }
    }
    return result;
  }

  // Clip elements for numerical stability
  public clip(min: number, max: number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        result.data[r][c] = Math.max(min, Math.min(max, this.data[r][c]));
      }
    }
    return result;
  }

  // Layer Normalization — critical for training stability in transformers
  // Normalizes each row independently: y = gamma * (x - mean) / sqrt(var + eps) + beta
  public static layerNorm(
    X: Matrix,
    gamma: Matrix,   // [1 x cols] scale parameters
    beta: Matrix,    // [1 x cols] shift parameters
    eps = 1e-5
  ): { output: Matrix; mean: number[]; variance: number[] } {
    const result = new Matrix(X.rows, X.cols);
    const means: number[] = [];
    const variances: number[] = [];

    for (let r = 0; r < X.rows; r++) {
      // Compute row mean
      let mean = 0;
      for (let c = 0; c < X.cols; c++) {
        mean += X.data[r][c];
      }
      mean /= X.cols;
      means.push(mean);

      // Compute row variance
      let variance = 0;
      for (let c = 0; c < X.cols; c++) {
        const diff = X.data[r][c] - mean;
        variance += diff * diff;
      }
      variance /= X.cols;
      variances.push(variance);

      // Normalize and apply affine transform
      const invStd = 1.0 / Math.sqrt(variance + eps);
      for (let c = 0; c < X.cols; c++) {
        const normalized = (X.data[r][c] - mean) * invStd;
        result.data[r][c] = gamma.data[0][c] * normalized + beta.data[0][c];
      }
    }

    return { output: result, mean: means, variance: variances };
  }

  // Layer Norm backward pass for backpropagation
  public static layerNormBackward(
    dOut: Matrix,      // upstream gradient
    X: Matrix,         // original input to layerNorm
    gamma: Matrix,     // scale parameters
    mean: number[],    // cached means
    variance: number[],// cached variances
    eps = 1e-5
  ): { dX: Matrix; dGamma: Matrix; dBeta: Matrix } {
    const N = X.cols;
    const dX = new Matrix(X.rows, X.cols);
    const dGamma = new Matrix(1, X.cols);
    const dBeta = new Matrix(1, X.cols);

    for (let r = 0; r < X.rows; r++) {
      const invStd = 1.0 / Math.sqrt(variance[r] + eps);

      // Compute normalized values
      const xNorm: number[] = [];
      for (let c = 0; c < N; c++) {
        xNorm.push((X.data[r][c] - mean[r]) * invStd);
      }

      // dBeta and dGamma accumulation
      for (let c = 0; c < N; c++) {
        dBeta.data[0][c] += dOut.data[r][c];
        dGamma.data[0][c] += dOut.data[r][c] * xNorm[c];
      }

      // dX computation
      let dxNormSum = 0;
      let dxNormXnormSum = 0;
      for (let c = 0; c < N; c++) {
        const dxNorm = dOut.data[r][c] * gamma.data[0][c];
        dxNormSum += dxNorm;
        dxNormXnormSum += dxNorm * xNorm[c];
      }

      for (let c = 0; c < N; c++) {
        const dxNorm = dOut.data[r][c] * gamma.data[0][c];
        dX.data[r][c] = invStd / N * (N * dxNorm - dxNormSum - xNorm[c] * dxNormXnormSum);
      }
    }

    return { dX, dGamma, dBeta };
  }

  // Compute statistics (for /api/weights endpoint visualization)
  public stats(): { mean: number; std: number; min: number; max: number; norm: number } {
    let sum = 0, sumSq = 0, min = Infinity, max = -Infinity, norm = 0;
    const total = this.rows * this.cols;
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const v = this.data[r][c];
        sum += v;
        sumSq += v * v;
        norm += v * v;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const mean = sum / total;
    const std = Math.sqrt(sumSq / total - mean * mean);
    return { mean, std, min, max, norm: Math.sqrt(norm) };
  }
}

// Tokenizer mapping tokens <--> tokenIDs
export class WordTokenizer {
  public vocabulary: string[];
  public tokenToIdx: Record<string, number> = {};
  public idxToToken: Record<number, string> = {};

  // Simple regex to split text into words, punctuation, and spaces
  private tokenizeRegex = /[\w']+|[^\w\s]+|\s+/g;

  constructor(texts: string[]) {
    // Collect all unique tokens
    const allTokens = new Set<string>();
    
    // Add essential specials
    allTokens.add("<pad>");
    allTokens.add("<unk>");

    for (const text of texts) {
      const tokens = text.match(this.tokenizeRegex) || [];
      for (let i = 0; i < tokens.length; i++) {
        allTokens.add(tokens[i]);
      }
    }

    this.vocabulary = Array.from(allTokens).sort();
    
    this.vocabulary.forEach((token, idx) => {
      this.tokenToIdx[token] = idx;
      this.idxToToken[idx] = token;
    });
  }

  get vocabSize(): number {
    return this.vocabulary.length;
  }

  public encode(text: string): number[] {
    const ids: number[] = [];
    const tokens = text.match(this.tokenizeRegex) || [];
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (token in this.tokenToIdx) {
        ids.push(this.tokenToIdx[token]);
      } else {
        ids.push(this.tokenToIdx["<unk>"]);
      }
    }
    return ids;
  }

  public decode(ids: number[]): string {
    return ids
      .map(id => {
        if (id in this.idxToToken) {
          const token = this.idxToToken[id];
          return token === "<pad>" || token === "<unk>" ? "" : token;
        }
        return "";
      })
      .join("");
  }
}

// Cache of forward pass states
export interface ForwardCache {
  inputTokens: number[];
  seqLength: number;
  E_tok: Matrix;       // Token embedding [SeqLength x d_model]
  E_pos: Matrix;       // Positional encoding [SeqLength x d_model]
  X_in: Matrix;        // Summed embeddings [SeqLength x d_model]
  X_ln1: Matrix;       // Layer-normed input (pre-attention) [SeqLength x d_model]
  ln1_mean: number[];  // LayerNorm1 cached means
  ln1_var: number[];   // LayerNorm1 cached variances
  Q: Matrix;           // Query [SeqLength x d_head]
  K: Matrix;           // Key [SeqLength x d_head]
  V: Matrix;           // Value [SeqLength x d_head]
  A_raw: Matrix;       // Raw attention scores [SeqLength x SeqLength]
  A: Matrix;           // Softmax causal attention probabilities [SeqLength x SeqLength]
  O: Matrix;           // Attended context outputs [SeqLength x d_head]
  O_proj: Matrix;      // Attended context output projected [SeqLength x d_model]
  X_res1: Matrix;      // Residual add [SeqLength x d_model]
  X_ln2: Matrix;       // Layer-normed residual (pre-FFN) [SeqLength x d_model]
  ln2_mean: number[];  // LayerNorm2 cached means
  ln2_var: number[];   // LayerNorm2 cached variances
  FF_h1: Matrix;       // Feedforward hidden pre-activation [SeqLength x d_ff]
  FF_relu: Matrix;     // Feedforward hidden activated [SeqLength x d_ff]
  FF_out: Matrix;      // Feedforward output [SeqLength x d_model]
  X_res2: Matrix;      // Final Layer representation [SeqLength x d_model]
  Logits: Matrix;      // Token prediction values [SeqLength x VocabSize]
}

// Custom scratch-built single-block causal GPT Transformer in TypeScript/NumPy equivalents
export class ScratchTransformer {
  // Hyperparameters — increased for better capacity
  public vocabSize: number;
  public dModel = 64;
  public dHead = 32;
  public dFf = 128;
  public maxSeqLen = 48;

  // Parameters
  public W_token: Matrix;  // Token embedding dictionary [VocabSize x dModel]
  public W_pos: Matrix;    // Positional embedding dictionary [maxSeqLen x dModel]

  // Layer Normalization parameters (Pre-LN architecture)
  public LN1_gamma: Matrix; // LayerNorm1 scale [1 x dModel]
  public LN1_beta: Matrix;  // LayerNorm1 shift [1 x dModel]
  public LN2_gamma: Matrix; // LayerNorm2 scale [1 x dModel]
  public LN2_beta: Matrix;  // LayerNorm2 shift [1 x dModel]

  // Self-Attention layer weights
  public W_q: Matrix;      // Query weights [dModel x dHead]
  public W_k: Matrix;      // Key weights [dModel x dHead]
  public W_v: Matrix;      // Value weights [dModel x dHead]
  public W_proj: Matrix;   // Output project weight [dHead x dModel]

  // Feedforward layer weights
  public W_ff1: Matrix;    // FF input weight [dModel x dFf]
  public B_ff1: Matrix;    // FF input bias [1 x dFf]
  public W_ff2: Matrix;    // FF output weight [dFf x dModel]
  public B_ff2: Matrix;    // FF output bias [1 x dModel]

  // Unembedding output projection
  public W_out: Matrix;    // Ultimate projection weights [dModel x VocabSize]
  public B_out: Matrix;    // Ultimate projection bias [1 x VocabSize]

  constructor(vocabSize: number) {
    this.vocabSize = vocabSize;

    // Xavier glorot randomized values for embedding layers
    this.W_token = new Matrix(vocabSize, this.dModel, undefined, true);
    
    // Sinusoidal positional encoding for better position awareness
    this.W_pos = new Matrix(this.maxSeqLen, this.dModel);
    for (let pos = 0; pos < this.maxSeqLen; pos++) {
      for (let i = 0; i < this.dModel; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / this.dModel);
        this.W_pos.data[pos][i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      }
    }

    // Layer Normalization parameters — initialized to identity transform
    this.LN1_gamma = new Matrix(1, this.dModel);
    this.LN1_beta = new Matrix(1, this.dModel);
    this.LN2_gamma = new Matrix(1, this.dModel);
    this.LN2_beta = new Matrix(1, this.dModel);
    for (let c = 0; c < this.dModel; c++) {
      this.LN1_gamma.data[0][c] = 1.0;
      this.LN2_gamma.data[0][c] = 1.0;
      // beta already initialized to 0
    }

    // Xavier init for attention projections
    this.W_q = new Matrix(this.dModel, this.dHead, undefined, true);
    this.W_k = new Matrix(this.dModel, this.dHead, undefined, true);
    this.W_v = new Matrix(this.dModel, this.dHead, undefined, true);
    this.W_proj = new Matrix(this.dHead, this.dModel, undefined, true);

    // He initialization for ReLU feedforward layers (better gradient flow)
    this.W_ff1 = Matrix.heInit(this.dModel, this.dFf);
    this.B_ff1 = new Matrix(1, this.dFf); // Zero init biases
    this.W_ff2 = Matrix.heInit(this.dFf, this.dModel);
    this.B_ff2 = new Matrix(1, this.dModel);

    // Xavier init for unembedding
    this.W_out = new Matrix(this.dModel, vocabSize, undefined, true);
    this.B_out = new Matrix(1, vocabSize);
  }

  // Count total trainable parameters
  public countParameters(): number {
    let total = 0;
    total += this.W_token.rows * this.W_token.cols;
    total += this.W_pos.rows * this.W_pos.cols;
    total += this.LN1_gamma.cols + this.LN1_beta.cols;
    total += this.LN2_gamma.cols + this.LN2_beta.cols;
    total += this.W_q.rows * this.W_q.cols;
    total += this.W_k.rows * this.W_k.cols;
    total += this.W_v.rows * this.W_v.cols;
    total += this.W_proj.rows * this.W_proj.cols;
    total += this.W_ff1.rows * this.W_ff1.cols;
    total += this.B_ff1.cols;
    total += this.W_ff2.rows * this.W_ff2.cols;
    total += this.B_ff2.cols;
    total += this.W_out.rows * this.W_out.cols;
    total += this.B_out.cols;
    return total;
  }

  // Complete forward pass taking prompt IDs & returning logits and caches
  public forward(tokenIds: number[]): { logits: Matrix; cache: ForwardCache } {
    const seqLen = tokenIds.length;
    if (seqLen > this.maxSeqLen) {
      throw new Error(`Input length of ${seqLen} exceeds supported max context of ${this.maxSeqLen}`);
    }

    // 1. Embedding layer
    const E_tok = new Matrix(seqLen, this.dModel);
    const E_pos = new Matrix(seqLen, this.dModel);
    for (let i = 0; i < seqLen; i++) {
      const tokId = tokenIds[i];
      // Gather token embedding
      E_tok.data[i] = [...this.W_token.data[tokId]];
      // Gather position embedding
      E_pos.data[i] = [...this.W_pos.data[i]];
    }

    const X_in = E_tok.add(E_pos); // Embedding activation

    // 2. Pre-Attention Layer Normalization (Pre-LN architecture — more stable than Post-LN)
    const ln1 = Matrix.layerNorm(X_in, this.LN1_gamma, this.LN1_beta);
    const X_ln1 = ln1.output;

    // 3. Self Attention projection: Q = X_ln1 * W_q, K = X_ln1 * W_k, V = X_ln1 * W_v
    const Q = Matrix.dot(X_ln1, this.W_q);
    const K = Matrix.dot(X_ln1, this.W_k);
    const V = Matrix.dot(X_ln1, this.W_v);

    // Attention probabilities computation: A_raw = (Q * K^T) / sqrt(dHead)
    const K_T = K.transpose();
    const Q_K_T = Matrix.dot(Q, K_T);
    const scale = 1 / Math.sqrt(this.dHead);
    const A_raw = Q_K_T.scale(scale);

    // Apply causal mask: for cell indexed i, j where j > i (cannot read future tokens), raw A score is extremely negative
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        if (j > i) {
          A_raw.data[i][j] = -1e15; // causal mask approximation
        }
      }
    }

    // Row-wise softmax to compute attention weights
    const A = A_raw.softmax();

    // Weighted context value output: O = A * V
    const O = Matrix.dot(A, V);

    // Project output back to dModel dimensions
    const O_proj = Matrix.dot(O, this.W_proj);

    // Visual skip-residual connection
    const X_res1 = X_in.add(O_proj);

    // 4. Pre-FFN Layer Normalization
    const ln2 = Matrix.layerNorm(X_res1, this.LN2_gamma, this.LN2_beta);
    const X_ln2 = ln2.output;

    // 5. Position-Wise Feedforward Block: FF = ReLU(X_ln2 * W_ff1 + B_ff1) * W_ff2 + B_ff2
    const FF_h1 = Matrix.dot(X_ln2, this.W_ff1).add(this.B_ff1);
    const FF_relu = FF_h1.relu();
    const FF_out = Matrix.dot(FF_relu, this.W_ff2).add(this.B_ff2);

    // Second residual connection
    const X_res2 = X_res1.add(FF_out);

    // 6. Decode to final tokens projection (Unembedding Logits)
    const Logits = Matrix.dot(X_res2, this.W_out).add(this.B_out);

    const cache: ForwardCache = {
      inputTokens: [...tokenIds],
      seqLength: seqLen,
      E_tok,
      E_pos,
      X_in,
      X_ln1,
      ln1_mean: ln1.mean,
      ln1_var: ln1.variance,
      Q,
      K,
      V,
      A_raw,
      A,
      O,
      O_proj,
      X_res1,
      X_ln2,
      ln2_mean: ln2.mean,
      ln2_var: ln2.variance,
      FF_h1,
      FF_relu,
      FF_out,
      X_res2,
      Logits
    };

    return { logits: Logits, cache };
  }

  // analytical Backpropagation to compute gradients and updates
  // targets a single sequence batch to preserve training logic readability
  public trainStep(tokenIds: number[], targets: number[], learningRate: number): { loss: number; cache: ForwardCache } {
    const { logits, cache } = this.forward(tokenIds);
    const seqLen = tokenIds.length;

    // First: Calculate Cross-Entropy Loss and logits gradient dLogits
    // loss_t = -log(softmax(logits_t)[target_t])
    let totalLoss = 0;
    const dLogits = new Matrix(seqLen, this.vocabSize);

    for (let t = 0; t < seqLen; t++) {
      const predLogits = logits.data[t];
      const maxL = Math.max(...predLogits);
      const exps = predLogits.map(v => Math.exp(v - maxL));
      const sumExps = exps.reduce((acc, curr) => acc + curr, 0) || 1e-9;
      const probs = exps.map(v => v / sumExps);

      const targetVal = targets[t];
      // compute loss safely
      totalLoss -= Math.log(Math.max(probs[targetVal], 1e-15));

      // Gradient dL/dLogits_t = probs_t - target_indicator_t
      for (let c = 0; c < this.vocabSize; c++) {
        dLogits.data[t][c] = probs[c];
      }
      dLogits.data[t][targetVal] -= 1;
      
      // Average the gradient over the sequence
      for (let c = 0; c < this.vocabSize; c++) {
        dLogits.data[t][c] /= seqLen;
      }
    }
    const meanLoss = totalLoss / seqLen;

    // Forward formula of Logits: Logits = X_res2 * W_out + B_out
    // dW_out = X_res2^T * dLogits
    const X_res2_T = cache.X_res2.transpose();
    const dW_out = Matrix.dot(X_res2_T, dLogits);
    
    // dB_out = sum of columns in dLogits
    const dB_out = new Matrix(1, this.vocabSize);
    for (let c = 0; c < this.vocabSize; c++) {
      let sum = 0;
      for (let t = 0; t < seqLen; t++) {
        sum += dLogits.data[t][c];
      }
      dB_out.data[0][c] = sum;
    }

    // dL/dX_res2 = dLogits * W_out^T
    const W_out_T = this.W_out.transpose();
    const dX_res2 = Matrix.dot(dLogits, W_out_T);

    // Deep trace Feedforward parameters backprop: X_res2 = X_res1 + FF_out
    // Hence dFF_out = dX_res2
    const dFF_out = dX_res2.copy();

    // FF_out = FF_relu * W_ff2 + B_ff2
    // dW_ff2 = FF_relu^T * dFF_out
    const FF_relu_T = cache.FF_relu.transpose();
    const dW_ff2 = Matrix.dot(FF_relu_T, dFF_out);

    // dB_ff2 = column sums of dFF_out
    const dB_ff2 = new Matrix(1, this.dModel);
    for (let c = 0; c < this.dModel; c++) {
      let sum = 0;
      for (let t = 0; t < seqLen; t++) {
        sum += dFF_out.data[t][c];
      }
      dB_ff2.data[0][c] = sum;
    }

    // dFF_relu = dFF_out * W_ff2^T
    const W_ff2_T = this.W_ff2.transpose();
    const dFF_relu = Matrix.dot(dFF_out, W_ff2_T);

    // Derivative through ReLU activation
    const dFF_h1 = cache.FF_h1.reluDerivative(dFF_relu);

    // FF_h1 = X_ln2 * W_ff1 + B_ff1
    // dW_ff1 = X_ln2^T * dFF_h1
    const X_ln2_T = cache.X_ln2.transpose();
    const dW_ff1 = Matrix.dot(X_ln2_T, dFF_h1);

    // dB_ff1 = column sums of dFF_h1
    const dB_ff1 = new Matrix(1, this.dFf);
    for (let c = 0; c < this.dFf; c++) {
      let sum = 0;
      for (let t = 0; t < seqLen; t++) {
        sum += dFF_h1.data[t][c];
      }
      dB_ff1.data[0][c] = sum;
    }

    // Backprop through LayerNorm2
    const W_ff1_T = this.W_ff1.transpose();
    const dX_ln2 = Matrix.dot(dFF_h1, W_ff1_T);
    const ln2Back = Matrix.layerNormBackward(
      dX_ln2, cache.X_res1, this.LN2_gamma, cache.ln2_mean, cache.ln2_var
    );

    // dX_res1 = dX_res2 + ln2Back.dX (from residual connection branch)
    const dX_res1 = dX_res2.add(ln2Back.dX);

    // Trace back through Visual Attention layers: X_res1 = X_in + O_proj
    // Therefore dO_proj = dX_res1
    const dO_proj = dX_res1.copy();

    // O_proj = O * W_proj
    // dW_proj = O^T * dO_proj
    const O_T = cache.O.transpose();
    const dW_proj = Matrix.dot(O_T, dO_proj);

    // dO = dO_proj * W_proj^T
    const W_proj_T = this.W_proj.transpose();
    const dO = Matrix.dot(dO_proj, W_proj_T);

    // Attention operation: O = A * V
    // dV = A^T * dO
    const A_T = cache.A.transpose();
    const dV = Matrix.dot(A_T, dO);

    // dA = dO * V^T
    const V_T = cache.V.transpose();
    const dA = Matrix.dot(dO, V_T);

    // Backprop Softmax: A = softmax(A_raw)
    // dRaw_t = A_t * (dA_t - sum(dA_t * A_t))
    const dA_raw = new Matrix(seqLen, seqLen);
    for (let i = 0; i < seqLen; i++) {
      let dotProd = 0;
      for (let j = 0; j < seqLen; j++) {
        dotProd += dA.data[i][j] * cache.A.data[i][j];
      }
      for (let j = 0; j < seqLen; j++) {
        dA_raw.data[i][j] = cache.A.data[i][j] * (dA.data[i][j] - dotProd);
      }
    }

    // Apply scaling back: A_raw = (Q * K^T) * scale
    const dScale = dA_raw.scale(1 / Math.sqrt(this.dHead));

    // dQ = dScale * K
    const dQ = Matrix.dot(dScale, cache.K);

    // dK = dScale^T * Q
    const dScale_T = dScale.transpose();
    const dK = Matrix.dot(dScale_T, cache.Q);

    // Projections of inputs back to residuals: Q = X_ln1 * W_q, K = X_ln1 * W_k, V = X_ln1 * W_v
    // dW_q = X_ln1^T * dQ
    // dW_k = X_ln1^T * dK
    // dW_v = X_ln1^T * dV
    const X_ln1_T = cache.X_ln1.transpose();
    const dW_q = Matrix.dot(X_ln1_T, dQ);
    const dW_k = Matrix.dot(X_ln1_T, dK);
    const dW_v = Matrix.dot(X_ln1_T, dV);

    // Backprop through LayerNorm1
    const W_q_T = this.W_q.transpose();
    const W_k_T = this.W_k.transpose();
    const W_v_T = this.W_v.transpose();
    const dX_ln1 = Matrix.dot(dQ, W_q_T)
      .add(Matrix.dot(dK, W_k_T))
      .add(Matrix.dot(dV, W_v_T));

    const ln1Back = Matrix.layerNormBackward(
      dX_ln1, cache.X_in, this.LN1_gamma, cache.ln1_mean, cache.ln1_var
    );

    // Upstream gradient for embeddings layer: dX_in = ln1Back.dX + dX_res1 (residual contribution)
    const dX_in = ln1Back.dX.add(dX_res1);

    // Clip all gradients for maximum execution stability (anti-explosion mechanism)
    const clipVal = 1.0;
    const clip = (M: Matrix) => M.clip(-clipVal, clipVal);

    // Final Stage updates with simple SGD
    this.W_out = this.W_out.subtract(clip(dW_out).scale(learningRate));
    this.B_out = this.B_out.subtract(clip(dB_out).scale(learningRate));
    this.W_ff2 = this.W_ff2.subtract(clip(dW_ff2).scale(learningRate));
    this.B_ff2 = this.B_ff2.subtract(clip(dB_ff2).scale(learningRate));
    this.W_ff1 = this.W_ff1.subtract(clip(dW_ff1).scale(learningRate));
    this.B_ff1 = this.B_ff1.subtract(clip(dB_ff1).scale(learningRate));
    this.W_proj = this.W_proj.subtract(clip(dW_proj).scale(learningRate));
    this.W_q = this.W_q.subtract(clip(dW_q).scale(learningRate));
    this.W_k = this.W_k.subtract(clip(dW_k).scale(learningRate));
    this.W_v = this.W_v.subtract(clip(dW_v).scale(learningRate));

    // Update LayerNorm parameters
    this.LN1_gamma = this.LN1_gamma.subtract(clip(ln1Back.dGamma).scale(learningRate));
    this.LN1_beta = this.LN1_beta.subtract(clip(ln1Back.dBeta).scale(learningRate));
    this.LN2_gamma = this.LN2_gamma.subtract(clip(ln2Back.dGamma).scale(learningRate));
    this.LN2_beta = this.LN2_beta.subtract(clip(ln2Back.dBeta).scale(learningRate));

    // Update Embeddings
    for (let i = 0; i < seqLen; i++) {
      const tokId = tokenIds[i];
      // SGD gradient update for W_token row
      for (let d = 0; d < this.dModel; d++) {
        const dW_tok_id = Math.max(-clipVal, Math.min(clipVal, dX_in.data[i][d]));
        this.W_token.data[tokId][d] -= learningRate * dW_tok_id;
      }
    }

    return { loss: meanLoss, cache };
  }

  // Text generator starting from prompt, running forward steps iteratively
  // Uses Top-K sampling with temperature for coherent and diverse generation
  public generate(promptIds: number[], maxLength = 50, temp = 0.8, topK = 12): number[] {
    const generated = [...promptIds];
    
    for (let step = 0; step < maxLength; step++) {
      // slice context if it exceeds max context length
      const seqStart = Math.max(0, generated.length - this.maxSeqLen);
      const context = generated.slice(seqStart);
      
      const { logits } = this.forward(context);
      
      // Look at the logits of the last sequence position
      const lastTokenLogits = logits.data[context.length - 1];
      
      // Temperature scales logits (clamp temp to avoid division by zero)
      const effectiveTemp = Math.max(temp, 0.05);
      const scaled = lastTokenLogits.map(v => v / effectiveTemp);
      
      // Top-K filtering: keep only the top K logits, set rest to -Infinity
      const indexed = scaled.map((val, idx) => ({ val, idx }));
      indexed.sort((a, b) => b.val - a.val);
      const topKIndices = new Set(indexed.slice(0, topK).map(item => item.idx));
      
      const filtered = scaled.map((val, idx) => topKIndices.has(idx) ? val : -Infinity);
      
      // Softmax over filtered logits
      const maxVal = Math.max(...filtered.filter(v => v > -Infinity));
      const exps = filtered.map(v => v === -Infinity ? 0 : Math.exp(v - maxVal));
      const sumExps = exps.reduce((acc, curr) => acc + curr, 0) || 1e-9;
      const probs = exps.map(v => v / sumExps);
      
      // Sample token matching probabilities
      let r = Math.random();
      let nextId = indexed[0].idx; // default to most probable
      let cumSum = 0;
      for (let i = 0; i < probs.length; i++) {
        cumSum += probs[i];
        if (r < cumSum) {
          nextId = i;
          break;
        }
      }
      
      generated.push(nextId);
    }

    return generated;
  }
}
