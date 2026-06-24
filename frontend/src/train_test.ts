import { ScratchTransformer } from "./model/transformer";
import { STARTUP_QUOTES } from "./model/dataset";

export class WordTokenizer {
  public vocabulary: string[];
  public charToIdx: Record<string, number> = {};
  public idxToChar: Record<number, string> = {};

  // Split by words, spaces, and punctuation
  private tokenizeRegex = /[\w']+|[^\w\s]+|\s+/g;

  constructor(texts: string[]) {
    const allTokens = new Set<string>();
    allTokens.add("<pad>");
    allTokens.add("<unk>");

    for (const text of texts) {
      const tokens = text.match(this.tokenizeRegex) || [];
      for (const t of tokens) {
        allTokens.add(t);
      }
    }
    this.vocabulary = Array.from(allTokens).sort();
    this.vocabulary.forEach((token, idx) => {
      this.charToIdx[token] = idx;
      this.idxToChar[idx] = token;
    });
  }

  get vocabSize(): number { return this.vocabulary.length; }

  public encode(text: string): number[] {
    const ids: number[] = [];
    const tokens = text.match(this.tokenizeRegex) || [];
    for (const token of tokens) {
      ids.push(this.charToIdx[token] ?? this.charToIdx["<unk>"]);
    }
    return ids;
  }

  public decode(ids: number[]): string {
    return ids
      .map(id => {
        const token = this.idxToChar[id];
        return token === "<pad>" || token === "<unk>" ? "" : token;
      })
      .join("");
  }
}

const tokenizer = new WordTokenizer(STARTUP_QUOTES);
console.log("Vocab size:", tokenizer.vocabSize);
const transformer = new ScratchTransformer(tokenizer.vocabSize);

let totalLoss = 0;
const steps = 1000;
console.log("Training for", steps, "steps...");
for (let step = 0; step < steps; step++) {
  const quote = STARTUP_QUOTES[step % STARTUP_QUOTES.length];
  const encoded = tokenizer.encode(quote);
  if (encoded.length < 3) continue;

  const sliceLen = Math.min(encoded.length - 1, transformer.maxSeqLen);
  const inputs = encoded.slice(0, sliceLen);
  const targets = encoded.slice(1, sliceLen + 1);

  let lr = 0.05;
  if (step < 50) lr = 0.01 + 0.04 * (step/50);
  
  const result = transformer.trainStep(inputs, targets, lr);
  totalLoss += result.loss;
}

console.log("Final loss:", totalLoss / steps);

const promptIds = tokenizer.encode("We ");
for(let i=0; i<3; i++) {
  const outputIds = transformer.generate(promptIds, 20, 0.5, 5);
  console.log("Output:", tokenizer.decode(outputIds));
}
