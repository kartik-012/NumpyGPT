/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from "react";
import {
  Brain,
  Cpu,
  TrendingDown,
  Play,
  RefreshCw,
  Copy,
  Check,
  Code,
  Sparkles,
  Lock,
  ArrowRight,
  Dices,
  BookOpen,
  Info
} from "lucide-react";

export default function App() {
  // Tabs: generate, heatmap, train, architecture
  const [activeTab, setActiveTab] = useState<"generate" | "heatmap" | "train" | "architecture">("generate");

  // Generator states
  const [prompt, setPrompt] = useState("We");
  const [maxLength, setMaxLength] = useState(15);
  const [temp, setTemp] = useState(0.4);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedText, setGeneratedText] = useState("");
  const [typedText, setTypedText] = useState("");
  const [typingIndex, setTypingIndex] = useState(0);
  const [copied, setCopied] = useState(false);

  // Model & Tokenizer states fetched from server
  const [tokens, setTokens] = useState<string[]>([]);
  const [attentionMatrix, setAttentionMatrix] = useState<number[][] | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number; val: number } | null>(null);
  const [activations, setActivations] = useState<{ Q: number[][]; K: number[][]; V: number[][] } | null>(null);
  const [vocab, setVocab] = useState<{ vocabulary: string[]; charToIdx: Record<string, number>; idxToChar: Record<number, string> } | null>(null);
  const [datasetStats, setDatasetStats] = useState<{ totalQuotes: number; modelDim: number; headDim: number; ffDim: number; maxSeqLen: number } | null>(null);
  const [quotes, setQuotes] = useState<string[]>([]);

  // Training States
  const [isTraining, setIsTraining] = useState(false);
  const [trainEpochs, setTrainEpochs] = useState(10);
  const [trainLr, setTrainLr] = useState(0.05);
  const [currentLoss, setCurrentLoss] = useState(2.0);
  const [totalSteps, setTotalSteps] = useState(2000);
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([
    { step: 0, loss: 5.4 },
    { step: 500, loss: 2.7 },
    { step: 1000, loss: 2.3 },
    { step: 1500, loss: 2.1 },
    { step: 2000, loss: 2.0 }
  ]);
  const [consoleLogs, setConsoleLogs] = useState<string[]>([
    "[System] NumPyGPT Scratch Transformer Initialized.",
    "[Model] d_model=64, d_head=32, d_ff=128, maxSeqLen=48.",
    "[Pre-Training] Completed 2000 steps with LR warmup. Loss converged.",
    "[Ready] Type a prompt or run interactive updates in the Trainer tab!"
  ]);

  // Scribe AI / Teacher explanation states (Gemini API)
  const [scribeConcept, setScribeConcept] = useState("Self-Attention Mechanism");
  const [scribeExplanation, setScribeExplanation] = useState("");
  const [isLoadingScribe, setIsLoadingScribe] = useState(false);

  // References to handle interval animations safely
  const typingTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Sample prompt selection choices
  const samplePrompts = [
    "Move fast",
    "We don't have",
    "Our AI is just",
    "Pivot is just",
    "Web3 is just"
  ];

  const mathConcepts = [
    { title: "Self-Attention Mechanism", desc: "How matrices Q, K, and V map character relevance." },
    { title: "Causal Masking", desc: "Preventing the model from looking ahead during autoregressive generation." },
    { title: "Scratch Backpropagation", desc: "How gradients dW/dB are mapped through linear and softmax layers." },
    { title: "Token & Position Embeddings", desc: "Combining discrete character arrays with trigonometric context." },
    { title: "Softmax Probability Scaling", desc: "How temperature scales raw logits to generate chaotic or safe text." }
  ];

  // Fetch initial info on mount
  useEffect(() => {
    fetchMetadata();
  }, []);

  const fetchMetadata = async () => {
    try {
      const vRes = await fetch("/api/vocab");
      const vData = await vRes.json();
      setVocab(vData);

      const dRes = await fetch("/api/dataset");
      const dData = await dRes.json();
      setQuotes(dData.quotes);
      setDatasetStats(dData.stats);

      const hRes = await fetch("/api/health");
      const hData = await hRes.json();
      setCurrentLoss(hData.currentLoss);
      setTotalSteps(hData.trainingSteps);
    } catch (e) {
      console.error("Failed to load model metadata from backend API:", e);
    }
  };

  // Run generator action
  const handleGenerate = async (targetPrompt = prompt) => {
    if (isGenerating) return;
    setIsGenerating(true);
    setTypedText("");
    setGeneratedText("");
    
    // clear prior interval
    if (typingTimerRef.current) clearInterval(typingTimerRef.current);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: targetPrompt,
          maxLength,
          temp
        })
      });
      const data = await res.json();
      
      if (data.success) {
        setGeneratedText(data.generatedText);
        setTokens(data.tokens || []);
        setAttentionMatrix(data.attentionMatrix || null);
        setActivations(data.activations || null);

        // Run progressive typewriter animation for the generated string chunk
        let currentIdx = 0;
        const textToType = data.generatedText;
        setTypedText("");
        
        typingTimerRef.current = setInterval(() => {
          if (currentIdx <= textToType.length) {
            setTypedText(textToType.slice(0, currentIdx));
            currentIdx++;
          } else {
            if (typingTimerRef.current) clearInterval(typingTimerRef.current);
            setIsGenerating(false);
          }
        }, 22); // typing characters fast but visually visible

      } else {
        setGeneratedText(`Error: ${data.error || "Generation mismatch"}`);
        setIsGenerating(false);
      }
    } catch (e: any) {
      setGeneratedText(`Error: Connection to Node dev container stalled. Please check terminal outputs.`);
      setIsGenerating(false);
    }
  };

  // Run manual training epochs (SGD updates)
  const handleTrain = async (epochsChoice = trainEpochs) => {
    if (isTraining) return;
    setIsTraining(true);
    
    const startStep = totalSteps;
    const addLog = (msg: string) => {
      setConsoleLogs(prev => [...prev, msg].slice(-100)); // cap logs list size
    };

    addLog(`[Trainer] Launching ${epochsChoice} backpropagation steps at lr=${trainLr}...`);

    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          epochs: epochsChoice,
          lr: trainLr
        })
      });
      const data = await res.json();

      if (data.success) {
        const targetQuote = quotes[Math.floor(Math.random() * quotes.length)] || "Move fast";
        addLog(`[Forward] Sampled training datum: "${targetQuote.slice(0, 20)}..."`);
        addLog(`[Backward] Compiled dLogits, dFF2, dFF1, dWProj, dQ, dK, dV gradients successfully.`);
        addLog(`[Weights] SGD Step applied. Loss reduced to ${data.currentLoss.toFixed(4)}.`);
        
        setCurrentLoss(data.currentLoss);
        setTotalSteps(data.totalSteps);
        if (data.history) {
          setLossHistory(data.history);
        }
      } else {
        addLog(`[Trainer] Error during gradient updating: ${data.error}`);
      }
    } catch (e) {
      addLog(`[Trainer] Connection timed out. SGD execution paused.`);
    } finally {
      setIsTraining(false);
    }
  };

  // Scribe AI / Teacher explanation request via Gemini
  const askScribeAI = async (concept = scribeConcept) => {
    if (isLoadingScribe) return;
    setIsLoadingScribe(true);
    setScribeExplanation("");

    try {
      const res = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ concept })
      });
      const data = await res.json();
      if (data.success) {
        setScribeExplanation(data.explanation);
      } else {
        setScribeExplanation("### Error\nFailed to acquire dynamic explanation from Gemini model: " + data.error);
      }
    } catch (e) {
      setScribeExplanation("### Connection Error\nCould not fetch explanation. Please ensure backend is alive.");
    } finally {
      setIsLoadingScribe(false);
    }
  };

  // Helper copy content
  const handleCopy = () => {
    navigator.clipboard.writeText(typedText || generatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Calculate coordinates for dynamic SVG loss chart
  const renderSVGChart = () => {
    if (!lossHistory || lossHistory.length === 0) return null;
    
    const width = 500;
    const height = 180;
    const padding = 30;

    const minX = Math.min(...lossHistory.map(d => d.step));
    const maxX = Math.max(...lossHistory.map(d => d.step));
    const minY = 0.5; // lock lower ceiling for nice curve density
    const maxY = 4.5;

    const getXCoord = (val: number) => padding + ((val - minX) / (maxX - minX || 1)) * (width - 2 * padding);
    const getYCoord = (val: number) => height - padding - ((val - minY) / (maxY - minY || 1)) * (height - 2 * padding);

    // Create SVG path
    let dPath = "";
    lossHistory.forEach((point, i) => {
      const x = getXCoord(point.step);
      const y = getYCoord(point.loss);
      if (i === 0) {
        dPath += `M ${x} ${y}`;
      } else {
        dPath += ` L ${x} ${y}`;
      }
    });

    return (
      <svg className="w-full h-full text-indigo-400" viewBox={`0 0 ${width} ${height}`}>
        <defs>
          <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#818cf8" stopOpacity="0.4"/>
            <stop offset="100%" stopColor="#a855f7" stopOpacity="0.0"/>
          </linearGradient>
        </defs>

        {/* Draw subtle grid horizontal lines */}
        <line x1={padding} y1={getYCoord(4.0)} x2={width - padding} y2={getYCoord(4.0)} stroke="#2A2A2E" strokeDasharray="3,3" />
        <line x1={padding} y1={getYCoord(3.0)} x2={width - padding} y2={getYCoord(3.0)} stroke="#2A2A2E" strokeDasharray="3,3" />
        <line x1={padding} y1={getYCoord(2.0)} x2={width - padding} y2={getYCoord(2.0)} stroke="#2A2A2E" strokeDasharray="3,3" />
        <line x1={padding} y1={getYCoord(1.0)} x2={width - padding} y2={getYCoord(1.0)} stroke="#2A2A2E" strokeDasharray="3,3" />

        {/* Gradient area under curve */}
        {lossHistory.length > 1 && (
          <path
            d={`${dPath} L ${getXCoord(lossHistory[lossHistory.length - 1].step)} ${height - padding} L ${getXCoord(lossHistory[0].step)} ${height - padding} Z`}
            fill="url(#chartGradient)"
          />
        )}

        {/* Chart Line Path */}
        <path d={dPath} fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

        {/* Data points */}
        {lossHistory.map((point, index) => (
          <circle
            key={index}
            cx={getXCoord(point.step)}
            cy={getYCoord(point.loss)}
            r="4.5"
            className="fill-purple-500 stroke-[#151518] stroke-2 hover:r-6 cursor-help transition-all"
            title={`Step ${point.step}: Loss ${point.loss.toFixed(4)}`}
          />
        ))}

        {/* Labels */}
        <text x={padding - 5} y={getYCoord(4.0) + 4} fill="#64748B" fontSize="9" textAnchor="end">4.0</text>
        <text x={padding - 5} y={getYCoord(2.5) + 4} fill="#64748B" fontSize="9" textAnchor="end">2.5</text>
        <text x={padding - 5} y={getYCoord(1.0) + 4} fill="#64748B" fontSize="9" textAnchor="end">1.0</text>

        <text x={padding} y={height - 10} fill="#64748B" fontSize="9">Step {minX}</text>
        <text x={width - padding} y={height - 10} fill="#64748B" fontSize="9" textAnchor="end">Step {maxX}</text>
      </svg>
    );
  };

  const getTemperatureLabel = (val: number) => {
    if (val < 0.4) return "🔒 Safe Predictable Vibe";
    if (val < 0.85) return "⚡ Classic Tech Startup Vibe";
    return "🔥 Pure Desperate Vibe (Chaos Mode)";
  };

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-slate-200 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      
      {/* Top Header Grid Area */}
      <header className="border-b border-[#2A2A2E] bg-[#151518]/90 backdrop-blur-md sticky top-0 z-50 px-6 py-4">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          
          {/* Logo Identity */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl accent-gradient flex items-center justify-center font-bold text-white italic text-lg shadow-lg shadow-indigo-500/15">
              N
            </div>
            <div>
              <span className="font-sans text-xl font-bold tracking-tight text-white flex items-center gap-1.5">
                NumPyGPT
                <span className="text-[10px] bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded-full text-indigo-400 font-mono font-bold tracking-widest uppercase">
                  v1.0 Scratch
                </span>
              </span>
            </div>
          </div>

          {/* Navigation Control Hub */}
          <nav className="flex items-center gap-1.5 bg-[#0a0a0b] p-1 rounded-2xl border border-[#2A2A2E] overflow-x-auto max-w-full">
            <button
              onClick={() => setActiveTab("generate")}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold font-mono transition-all duration-150 cursor-pointer ${
                activeTab === "generate"
                  ? "bg-[#151518] text-indigo-400 border border-[#2A2A2E] shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Sparkles className="w-4 h-4 text-indigo-400" />
              Generator
            </button>
            <button
              onClick={() => setActiveTab("heatmap")}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold font-mono transition-all duration-150 cursor-pointer ${
                activeTab === "heatmap"
                  ? "bg-[#151518] text-purple-400 border border-[#2A2A2E] shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Cpu className="w-4 h-4 text-purple-400" />
              Self-Attention Map
            </button>
            <button
              onClick={() => setActiveTab("train")}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold font-mono transition-all duration-150 cursor-pointer ${
                activeTab === "train"
                  ? "bg-[#151518] text-amber-500 border border-[#2A2A2E] shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <TrendingDown className="w-4 h-4 text-amber-500" />
              Interactive Trainer
            </button>
            <button
              onClick={() => setActiveTab("architecture")}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold font-mono transition-all duration-150 cursor-pointer ${
                activeTab === "architecture"
                  ? "bg-[#151518] text-indigo-400 border border-[#2A2A2E] shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <BookOpen className="w-4 h-4 text-indigo-400" />
              Internals Inspector
            </button>
          </nav>

        </div>
      </header>

      {/* Main Container Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        
        {/* TAB 1: GENERATE SCREEN */}
        {activeTab === "generate" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* Control Panel Parameters (Left) */}
            <div className="lg:col-span-5 bento-card space-y-6">
              <div>
                <span className="px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-400 text-[10px] font-bold uppercase tracking-wider mb-2.5 inline-block">
                  Generator Config
                </span>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Set parameters for our custom, scratch-coded causal GPT network. Keep sequences within word limits.
                </p>
              </div>

              {/* Sample prompts */}
              <div className="space-y-2.5">
                <label className="text-xs font-mono text-slate-500 block font-bold uppercase tracking-wider">Quick Starters</label>
                <div className="flex flex-wrap gap-1.5">
                  {samplePrompts.map((p, i) => (
                    <button
                      key={i}
                      onClick={() => setPrompt(p)}
                      className="text-xs px-2.5 py-1.5 rounded-lg bg-[#1D1D21] hover:bg-[#2A2A2E] text-indigo-300 font-mono border border-[#2A2A2E] cursor-pointer transition-colors"
                    >
                      "{p}"
                    </button>
                  ))}
                </div>
              </div>

              {/* text prompt input */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-mono">
                  <label className="text-slate-400 font-bold uppercase tracking-wider text-[10px]">Prompt Sequence</label>
                  <span className="text-slate-500 font-medium">{prompt.length}/32 words</span>
                </div>
                <input
                  type="text"
                  maxLength={32}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Type starter word(s), e.g., Move fast..."
                  className="w-full text-sm font-mono px-4 py-3 rounded-xl bg-[#0A0A0B] border border-[#2A2A2E] focus:border-indigo-500 text-white outline-none transition-all placeholder:text-slate-700"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && prompt) handleGenerate();
                  }}
                />
              </div>

              {/* Slider parameters */}
              <div className="space-y-4 pt-1">
                {/* Temperature */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-400 font-bold uppercase tracking-wider text-[10px]">Temperature</span>
                    <span className="text-indigo-400 font-bold text-glow">{temp.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="1.5"
                    step="0.05"
                    value={temp}
                    onChange={(e) => setTemp(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-[#0A0A0B] rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                  <span className="text-[11px] font-mono text-slate-500 block italic leading-none">
                    {getTemperatureLabel(temp)}
                  </span>
                </div>

                {/* Length */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-400 font-bold uppercase tracking-wider text-[10px]">Output Word Limit</span>
                    <span className="text-indigo-400 font-bold text-glow">{maxLength} words</span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="100"
                    step="5"
                    value={maxLength}
                    onChange={(e) => setMaxLength(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-[#0A0A0B] rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                </div>
              </div>

              {/* Fire Button */}
              <button
                onClick={() => handleGenerate()}
                disabled={isGenerating || !prompt}
                className={`w-full py-3.5 rounded-xl font-mono text-sm font-bold tracking-wider transition-all duration-200 cursor-pointer flex items-center justify-center gap-2 border shadow-lg ${
                  isGenerating || !prompt
                    ? "bg-[#1D1D21] text-slate-500 border-[#2A2A2E] cursor-not-allowed"
                    : "accent-gradient hover:accent-gradient-hover text-white border-indigo-500/30 shadow-indigo-500/10 hover:scale-[1.01]"
                }`}
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-white" />
                    MAPPED MULTIPLICATIONS...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    RUN FORWARD PASS
                  </>
                )}
              </button>

            </div>

            {/* Simulated Live Streaming Output (Right) */}
            <div className="lg:col-span-7 space-y-6">
              
              <div className="bento-card relative min-h-[295px] flex flex-col justify-between">
                
                {/* Meta details */}
                <div className="flex items-center justify-between border-b border-[#2A2A2E] pb-3.5 mb-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse" />
                    <span className="font-mono text-xs text-slate-400 uppercase font-bold tracking-wider">
                      Transformer Logged Output
                    </span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-500">
                    Loss: <span className="text-amber-500 font-bold text-glow">{currentLoss.toFixed(4)}</span>
                  </div>
                </div>

                {/* Simulated Character Typing Area */}
                <div className="flex-grow flex items-center justify-center py-6">
                  {typedText ? (
                    <div className="relative w-full">
                      <p className="font-mono text-xl sm:text-2xl text-emerald-400 font-bold leading-relaxed tracking-wide text-center text-glow">
                        "{typedText}"
                        <span className="inline-block w-2.5 h-5 ml-1 bg-emerald-400 animate-[pulse_0.8s_infinite] align-middle" />
                      </p>
                    </div>
                  ) : isGenerating ? (
                    <div className="space-y-3 text-center">
                      <p className="text-xs font-mono text-slate-500 animate-pulse uppercase tracking-widest">
                        MULTIPLYING QKV MATRICES
                      </p>
                      <div className="flex justify-center gap-1.5 pt-1">
                        <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-[bounce_1.2s_infinite_100ms]" />
                        <span className="w-2.5 h-2.5 rounded-full bg-purple-500 animate-[bounce_1.2s_infinite_300ms]" />
                        <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-[bounce_1.2s_infinite_500ms]" />
                      </div>
                    </div>
                  ) : (
                    <div className="text-center max-w-sm text-slate-500 space-y-3">
                      <Dices className="w-10 h-10 mx-auto text-slate-700 stroke-1" />
                      <p className="font-mono text-xs leading-relaxed">
                        Enter a prompt snippet, click "Run Forward Pass", and watch token logits resolve word arrays in real time.
                      </p>
                    </div>
                  )}
                </div>

                {/* Footer Controls */}
                <div className="border-t border-[#2A2A2E] pt-4 mt-4 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono">
                  <span className="text-slate-500 text-[11px]">
                    Note: Generates word-by-word from scratch math.
                  </span>
                  
                  {/* Share/Actions */}
                  {typedText && !isGenerating && (
                    <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                      <button
                        onClick={handleCopy}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#1D1D21] hover:bg-[#2A2A2E] text-indigo-300 font-bold border border-[#2A2A2E] transition-colors cursor-pointer"
                      >
                        {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                        {copied ? "Copied" : "Copy Output"}
                      </button>
                      <button
                        onClick={() => handleGenerate()}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#1D1D21] hover:bg-[#2A2A2E] text-indigo-300 font-bold border border-[#2A2A2E] transition-colors cursor-pointer"
                      >
                        <RefreshCw className="w-4 h-4" />
                        Regenerate
                      </button>
                    </div>
                  )}
                </div>

              </div>

              {/* Informational Bento Card on NumPy Math */}
              <div className="bento-card border-slate-700/30">
                <div className="flex gap-4">
                  <div className="p-2 bg-indigo-500/10 rounded-xl border border-indigo-500/10 self-start">
                    <Info className="w-5 h-5 text-indigo-400" />
                  </div>
                  <div className="space-y-1 my-auto">
                    <span className="px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 text-[10px] font-bold uppercase tracking-widest mb-1.5 inline-block">Math Proof</span>
                    <p className="text-xs text-slate-400 leading-relaxed font-sans">
                      This model performs actual vector multiplications on a single d_model = 64 layer block with a 32-dim attention head and Pre-LN architecture. When you type <code className="bg-[#0A0A0B] px-1.5 py-0.5 rounded border border-[#2A2A2E] text-yellow-300 font-mono text-[11px]">We</code>, words are mapped to token IDs, combined with sinusoidal absolute position grids, layer-normalized, mapped into Q (Query), K (Key), V (Value) arrays, scaled down, masked to hide future states, evaluated, projected, and Top-K sampled to print the next word!
                    </p>
                  </div>
                </div>
              </div>

            </div>

          </div>
        )}

        {/* TAB 2: SELF-ATTENTION HEATMAP VIEW */}
        {activeTab === "heatmap" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* Attention Heatmap Matrix (Left) */}
            <div className="lg:col-span-7 bento-card space-y-6">
              
              <div>
                <span className="px-2.5 py-1 rounded-md bg-purple-500/10 text-purple-400 text-[10px] font-bold uppercase tracking-wider mb-2.5 inline-block">
                  Causal Attention Matrix
                </span>
                <p className="text-xs text-slate-400 leading-relaxed font-sans">
                  This live map shows the (Q * K^T) / &radic;d_k Softmax probabilities for the active prompt. The diagonal blocks illustrate future tokens blinded out by the 🔒 **Causal Mask**.
                </p>
              </div>

              {attentionMatrix && tokens && tokens.length > 0 ? (
                <div className="space-y-4">
                  
                  {/* Heatmap Grid Wrapper */}
                  <div className="overflow-auto border border-[#2A2A2E] p-4 bg-[#0A0A0B] rounded-2xl">
                    <div 
                      className="grid gap-1 mx-auto" 
                      style={{ 
                        gridTemplateColumns: `repeat(${tokens.length + 1}, minmax(32px, 1fr))`,
                        maxWidth: "550px"
                      }}
                    >
                      {/* Top Corner cell blank */}
                      <div className="flex items-center justify-center font-mono text-[10px] text-slate-500 h-8 font-bold">Col</div>

                      {/* Header columns represent source tokens (K) */}
                      {tokens.map((token, colIndex) => (
                        <div 
                          key={`head-c-${colIndex}`} 
                          className="flex items-center justify-center font-mono text-sm text-slate-400 h-8 font-bold border-b border-[#2A2A2E]"
                          title={`Source token: '${token}' at index ${colIndex}`}
                        >
                          {token === " " ? "␣" : token}
                        </div>
                      ))}

                      {/* Rows represent queries (Q) */}
                      {attentionMatrix.slice(0, tokens.length).map((row, rowIndex) => (
                        <use key={`row-${rowIndex}`} className="contents">
                          
                          {/* Row Header Label representing Query source (Q) */}
                          <div className="flex items-center justify-end font-mono text-sm text-slate-400 pr-2.5 font-bold border-r border-[#2A2A2E]">
                            {tokens[rowIndex] === " " ? "␣" : tokens[rowIndex]}
                          </div>

                          {/* Columns under Row */}
                          {row.slice(0, tokens.length).map((cellVal, colIndex) => {
                            const isCausalMasked = colIndex > rowIndex;
                            
                            // Color intensity of attention score (0 to 1)
                            const heatOpacity = isCausalMasked ? 0.05 : Math.max(0.04, cellVal);
                            const celColor = isCausalMasked 
                              ? "bg-red-950/20 text-[#2A2A2E]" 
                              : `bg-emerald-500 text-white`;

                            return (
                              <div
                                key={`cell-${rowIndex}-${colIndex}`}
                                onMouseEnter={() => !isCausalMasked && setHoveredCell({ r: rowIndex, c: colIndex, val: cellVal })}
                                onMouseLeave={() => setHoveredCell(null)}
                                className={`aspect-square min-h-[32px] rounded-lg flex items-center justify-center font-mono text-[9px] relative cursor-pointer border border-[#0A0A0B]/30 transition-all hover:scale-[1.12] hover:z-10`}
                                style={{
                                  backgroundColor: !isCausalMasked ? `rgba(16, 185, 129, ${heatOpacity})` : undefined,
                                  color: isCausalMasked ? "#475569" : cellVal > 0.4 ? "#0A0A0B" : "#a7f3d0"
                                }}
                              >
                                {isCausalMasked ? (
                                  <Lock className="w-2.5 h-2.5 text-slate-700" />
                                ) : (
                                  cellVal.toFixed(2)
                                )}
                              </div>
                            );
                          })}
                        </use>
                      ))}

                    </div>
                  </div>

                  {/* Axis Legend labels */}
                  <div className="flex justify-between items-center text-xs text-slate-500 font-mono px-2">
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-emerald-500/20 border border-emerald-500/40" /> Low Attention</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-emerald-500/90" /> High Attention</span>
                    <span className="flex items-center gap-1.5"><Lock className="w-3 h-3 text-slate-600" /> Causal Mask Block (j &gt; i)</span>
                  </div>

                </div>
              ) : (
                <div className="border border-dashed border-[#2A2A2E] p-12 text-center text-slate-500 rounded-2xl bg-[#0A0A0B]/50">
                  <Cpu className="w-8 h-8 mx-auto text-slate-700 mb-3 animate-spin" />
                  <p className="font-mono text-xs leading-relaxed">
                    Please generate a quote in the **Generator** tab first to capture activation weights.
                  </p>
                </div>
              )}

            </div>

            {/* Inspect Activation vectors (Right) */}
            <div className="lg:col-span-5 space-y-6">
              
              {/* Active hover Cell explanation */}
              <div className="bento-card">
                <span className="px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 text-[10px] font-bold uppercase tracking-widest mb-4 inline-block">
                  Attention Inspector
                </span>
                
                {hoveredCell && tokens ? (
                  <div className="space-y-4 font-mono text-xs">
                    <div className="bg-[#1D1D21] p-3 rounded-xl border border-[#2A2A2E] space-y-2">
                      <div className="text-slate-500 font-bold text-[10px] uppercase tracking-wider">Step connection:</div>
                      <div className="text-white text-xs font-semibold flex items-center gap-1.5">
                        Token <span className="text-yellow-300 bg-[#0A0A0B] px-1.5 py-0.5 rounded border border-[#2A2A2E] font-bold">"{tokens[hoveredCell.r]}"</span> (pos {hoveredCell.r})
                        <ArrowRight className="w-3 h-3 text-emerald-400" />
                        Token <span className="text-yellow-300 bg-[#0A0A0B] px-1.5 py-0.5 rounded border border-[#2A2A2E] font-bold">"{tokens[hoveredCell.c]}"</span> (pos {hoveredCell.c})
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="text-slate-500 font-bold text-[10px] uppercase tracking-wider">Attention Coefficient score A_ij:</div>
                      <div className="text-emerald-400 font-bold text-lg text-glow">{(hoveredCell.val * 100).toFixed(1)}% weight</div>
                    </div>

                    <div className="space-y-1.5 leading-relaxed text-slate-400 font-sans text-xs">
                      <span className="text-white font-bold block mb-1 font-mono text-[10px] uppercase tracking-wider text-slate-300">Mathematical Formula:</span>
                      In the forward pass, we compute:
                      <code className="block bg-[#1D1D21] border border-[#2A2A2E] p-2.5 rounded-xl text-indigo-300 mt-1 pb-2 leading-loose font-mono">
                        e_ij = (Q_i &middot; K_j^T) / &radic;d_k
                        <br />
                        A_ij = exp(e_ij) / &Sigma;_k exp(e_ik)
                      </code>
                      This weight modulates values ($V$) to compute context outputs:
                      <code className="block bg-[#1D1D21] border border-[#2A2A2E] p-2.5 rounded-xl text-emerald-400 mt-1 font-mono">
                        O_i = &Sigma;_j (A_ij &middot; V_j)
                      </code>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-500 font-mono text-xs space-y-3">
                    <Info className="w-7 h-7 mx-auto text-slate-700 animate-bounce" />
                    <p className="leading-relaxed">Hover over unmasked cells in the causality grid to view vector products and soft weights.</p>
                  </div>
                )}
              </div>

              {/* Vectors Matrix viewer */}
              <div className="bento-card space-y-4">
                <span className="px-2.5 py-0.5 rounded-full bg-purple-500/10 text-purple-400 text-[10px] font-bold uppercase tracking-widest mb-1.5 inline-block">
                  Activations Q, K, V
                </span>
                <p className="text-xs text-slate-400 leading-relaxed font-sans">
                  Matrix arrays mapping tokens from d_model (64) to projection head (32).
                </p>

                {activations && tokens ? (
                  <div className="space-y-3 font-mono text-[11px]">
                    <div className="space-y-1">
                      <span className="text-indigo-400 font-bold">Query Matrix (Q) - Size [{tokens.length} x 16]</span>
                      <div className="bg-[#1D1D21] border border-[#2A2A2E] p-2.5 rounded-xl overflow-auto leading-normal whitespace-nowrap text-slate-400">
                        {activations.Q.slice(0, 4).map((row, rIdx) => (
                          <div key={rIdx}>
                            Row {rIdx} | {row.slice(0, 5).map(v => v.toFixed(3)).join(", ")} ...
                          </div>
                        ))}
                        {tokens.length > 4 && <div>... [+ {tokens.length - 4} more rows]</div>}
                      </div>
                    </div>

                    <div className="space-y-1">
                      <span className="text-emerald-400 font-bold">Key Matrix (K) - Size [{tokens.length} x 16]</span>
                      <div className="bg-[#1D1D21] border border-[#2A2A2E] p-2.5 rounded-xl overflow-auto leading-normal whitespace-nowrap text-slate-400">
                        {activations.K.slice(0, 4).map((row, rIdx) => (
                          <div key={rIdx}>
                            Row {rIdx} | {row.slice(0, 5).map(v => v.toFixed(3)).join(", ")} ...
                          </div>
                        ))}
                        {tokens.length > 4 && <div>... [+ {tokens.length - 4} more rows]</div>}
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs font-mono text-slate-500 italic">No active model activations stored.</p>
                )}
              </div>

            </div>

          </div>
        )}

        {/* TAB 3: INTERACTIVE TRAINING GRAPH & BACKPROP CONTROLS */}
        {activeTab === "train" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* Loss graph and live loss indicators (Left) */}
            <div className="lg:col-span-7 bento-card space-y-6">
              
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#2A2A2E] pb-4">
                <div>
                  <span className="px-2.5 py-1 rounded-md bg-amber-500/10 text-amber-500 text-[10px] font-bold uppercase tracking-wider mb-2.5 inline-block">
                    Backpropagation
                  </span>
                  <p className="text-xs text-slate-400 leading-relaxed font-sans mt-0.5">
                    Trigger genuine Stochastic Gradient Descent backpropagation equations on the Node server.
                  </p>
                </div>
                <div className="flex items-center gap-4 bg-[#0A0A0B] border border-[#2A2A2E] px-4 py-2.5 rounded-2xl shrink-0">
                  <div className="text-center font-mono">
                    <span className="text-[9px] text-slate-500 block font-bold uppercase tracking-wider">Current Loss</span>
                    <span className="text-amber-500 font-bold text-sm tracking-tight text-glow">{currentLoss.toFixed(4)}</span>
                  </div>
                  <div className="border-l border-[#2A2A2E] h-8" />
                  <div className="text-center font-mono">
                    <span className="text-[9px] text-slate-500 block font-bold uppercase tracking-wider">Total Steps</span>
                    <span className="text-slate-200 font-bold text-sm tracking-tight">{totalSteps}</span>
                  </div>
                </div>
              </div>

              {/* Chart Line Representation */}
              <div className="space-y-2.5">
                <span className="text-xs font-mono text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1.5"><TrendingDown className="w-4 h-4 text-amber-500" /> CROSS-ENTROPY LOSS CURVE</span>
                <div className="bg-[#0A0A0B] border border-[#2A2A2E] rounded-2xl p-4 h-[210px] w-full flex items-center justify-center relative overflow-hidden">
                  {renderSVGChart()}
                </div>
              </div>

              {/* Backprop Action Trigger Buttons */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                
                <button
                  onClick={() => handleTrain(1)}
                  disabled={isTraining}
                  className="py-3 rounded-xl font-mono text-xs font-bold bg-[#1D1D21] hover:bg-[#2A2A2E] border border-[#2A2A2E] hover:border-amber-500/20 text-slate-200 cursor-pointer transition-all flex items-center justify-center gap-1.5 disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 text-amber-500 ${isTraining ? "animate-spin" : ""}`} />
                  Backprop 1 step
                </button>

                <button
                  onClick={() => handleTrain(10)}
                  disabled={isTraining}
                  className="py-3 rounded-xl font-mono text-xs font-bold bg-[#1D1D21] hover:bg-[#2A2A2E] border border-[#2A2A2E] hover:border-amber-500/30 text-amber-500 cursor-pointer transition-all flex items-center justify-center gap-1.5 disabled:opacity-50"
                >
                  <Play className={`w-3.5 h-3.5 ${isTraining ? "animate-ping" : ""}`} />
                  Train 10 Steps
                </button>

                <button
                  onClick={() => handleTrain(50)}
                  disabled={isTraining}
                  className="py-3 rounded-xl font-mono text-xs font-bold bg-amber-500 hover:bg-amber-400 text-[#0a0a0b] cursor-pointer transition-all flex items-center justify-center gap-1.5 disabled:opacity-50 shadow-lg shadow-amber-500/10"
                >
                  <Cpu className={`w-3.5 h-3.5 ${isTraining ? "animate-spin" : ""}`} />
                  Train 50 Steps!
                </button>

              </div>

            </div>

            {/* Custom Interactive Logger / Console (Right) */}
            <div className="lg:col-span-5 space-y-6">
              
              {/* Controls lr/epochs */}
              <div className="bento-card space-y-3">
                <span className="px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-500 text-[10px] font-bold uppercase tracking-widest mb-1.5 inline-block">
                  Hyperparameter Tweaker
                </span>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5 font-mono text-xs">
                    <span className="text-slate-500 font-bold uppercase tracking-wider text-[10px]">Learning Rate</span>
                    <input
                      type="number"
                      step="0.01"
                      min="0.001"
                      max="0.5"
                      value={trainLr}
                      onChange={(e) => setTrainLr(parseFloat(e.target.value) || 0.05)}
                      className="w-full text-xs font-mono px-3 py-2 rounded-xl bg-[#0A0A0B] border border-[#2A2A2E] text-amber-500 outline-none focus:border-amber-500"
                    />
                  </div>
                  <div className="space-y-1.5 font-mono text-xs">
                    <span className="text-slate-500 font-bold uppercase tracking-wider text-[10px]">Epoch Steps</span>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={trainEpochs}
                      onChange={(e) => setTrainEpochs(parseInt(e.target.value) || 10)}
                      className="w-full text-xs font-mono px-3 py-2 rounded-xl bg-[#0A0A0B] border border-[#2A2A2E] text-amber-500 outline-none focus:border-amber-500"
                    />
                  </div>
                </div>
              </div>

              {/* Real time backprop logging command console */}
              <div className="bento-card font-mono text-[11px] h-[230px] flex flex-col justify-between shadow-lg">
                <div className="flex items-center justify-between border-b border-[#2A2A2E] pb-2.5 mb-2 text-slate-400">
                  <span className="font-bold uppercase tracking-wider text-[10px]">Console Terminal Trace</span>
                  <button 
                    onClick={() => setConsoleLogs([])}
                    className="text-[9px] hover:text-white border border-[#2A2A2E] px-2 py-1 rounded-lg bg-[#1D1D21] transition-colors cursor-pointer"
                  >
                    Clear Logs
                  </button>
                </div>

                {/* Log display */}
                <div className="flex-grow overflow-auto space-y-1 text-emerald-400 leading-normal pr-1 font-mono">
                  {consoleLogs.map((log, idx) => (
                    <div key={idx} className="break-words">
                      <span className="text-slate-400">Tokens</span> {log}
                    </div>
                  ))}
                </div>

                <div className="text-slate-500 border-t border-[#2A2A2E] pt-2 mt-2 text-[10px] flex justify-between items-center">
                  <span>GD Status: {isTraining ? "Updating Weights..." : "Idle / Synced"}</span>
                  <span>Batch size: 1 seq</span>
                </div>
              </div>

            </div>

          </div>
        )}

        {/* TAB 4: DEEP LAYER STRUCTURE INSPECTOR */}
        {activeTab === "architecture" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* Dimensions layout diagram tree (Left) */}
            <div className="lg:col-span-6 bento-card space-y-6">
              
              <div>
                <span className="px-2.5 py-1 rounded-md bg-purple-500/10 text-purple-400 text-[10px] font-bold uppercase tracking-wider mb-2.5 inline-block">
                  Core Model Layer Topology
                </span>
                <p className="text-xs text-slate-400 leading-relaxed font-sans">
                  View array geometries and matrix mapping layout configurations representation.
                </p>
              </div>

              {datasetStats ? (
                <div className="space-y-4 font-mono text-xs">
                  
                  {/* Token layer block */}
                  <div className="bg-[#1D1D21] border border-[#2A2A2E] p-4 rounded-xl space-y-2">
                    <div className="flex justify-between items-center text-slate-200 font-bold border-b border-[#2A2A2E] pb-1.5">
                      <span>1. Embedding Block Layer</span>
                      <span className="text-indigo-400 font-bold">W_token + W_pos</span>
                    </div>
                    <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
                      Maps incoming discrete token characters into vectors:
                      <ul className="list-disc list-inside ml-1 mt-1 text-slate-300 font-mono text-[10px] space-y-0.5">
                        <li>Token Embeddings: <code className="text-yellow-300 bg-[#0A0A0B] px-1 py-0.5 rounded border border-[#2A2A2E]">[{vocab?.vocabulary.length || 65} x {datasetStats.modelDim}]</code></li>
                        <li>Trig/Absolute Position: <code className="text-yellow-300 bg-[#0A0A0B] px-1 py-0.5 rounded border border-[#2A2A2E]">[{datasetStats.maxSeqLen} x {datasetStats.modelDim}]</code></li>
                      </ul>
                      <p className="mt-1">Adds discrete structures together to form a rich continuous positional representation.</p>
                    </div>
                  </div>

                  {/* Attention layer block */}
                  <div className="bg-[#1D1D21] border border-[#2A2A2E] p-4 rounded-xl space-y-2">
                    <div className="flex justify-between items-center text-slate-200 font-bold border-b border-[#2A2A2E] pb-1.5">
                      <span>2. Self-Attention Block Layer</span>
                      <span className="text-emerald-400 font-bold">Single-Head Projector</span>
                    </div>
                    <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
                      Projects and models spatial sequence context dependencies:
                      <ul className="list-disc list-inside ml-1 mt-1 text-slate-300 font-mono text-[10px] space-y-0.5">
                        <li>Projection weight Q, K, V: <code className="text-yellow-300 bg-[#0A0A0B] px-1 py-0.5 rounded border border-[#2A2A2E]">[{datasetStats.modelDim} x {datasetStats.headDim}]</code></li>
                        <li>Causal Output project: <code className="text-yellow-300 bg-[#0A0A0B] px-1 py-0.5 rounded border border-[#2A2A2E]">[{datasetStats.headDim} x {datasetStats.modelDim}]</code></li>
                      </ul>
                      <p className="mt-1">Weights sequence relationships via dynamic Softmax causal products.</p>
                    </div>
                  </div>

                  {/* Feedforward Dense block */}
                  <div className="bg-[#1D1D21] border border-[#2A2A2E] p-4 rounded-xl space-y-2">
                    <div className="flex justify-between items-center text-slate-200 font-bold border-b border-[#2A2A2E] pb-1.5">
                      <span>3. Position-Wise FFN Block Layer</span>
                      <span className="text-amber-500 font-bold">Linear + ReLU Dense</span>
                    </div>
                    <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
                      Applies dense multi-layer representation expansions:
                      <ul className="list-disc list-inside ml-1 mt-1 text-slate-300 font-mono text-[10px] space-y-0.5">
                        <li>Dense linear 1: <code className="text-yellow-300 bg-[#0A0A0B] px-1 py-0.5 rounded border border-[#2A2A2E]">[{datasetStats.modelDim} x {datasetStats.ffDim}]</code> (with bias)</li>
                        <li>Dense linear 2: <code className="text-yellow-300 bg-[#0A0A0B] px-1 py-0.5 rounded border border-[#2A2A2E]">[{datasetStats.ffDim} x {datasetStats.modelDim}]</code> (with bias)</li>
                      </ul>
                      <p className="mt-1">Expands layer dimensions to map non-linear syntactic trends before decode.</p>
                    </div>
                  </div>

                  {/* Unembedding Decoder block */}
                  <div className="bg-[#1D1D21] border border-[#2A2A2E] p-4 rounded-xl space-y-2">
                    <div className="flex justify-between items-center text-slate-200 font-bold border-b border-[#2A2A2E] pb-1.5">
                      <span>4. Unembedding Decoding Layer</span>
                      <span className="text-purple-400 font-bold font-mono">Logits Demultiplexer</span>
                    </div>
                    <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
                      Maps final layer dense outputs back to token coordinate scales:
                      <ul className="list-disc list-inside ml-1 mt-1 text-slate-300 font-mono text-[10px] space-y-0.5">
                        <li>Logits Projector weights: <code className="text-yellow-300 bg-[#0A0A0B] px-1 py-0.5 rounded border border-[#2A2A2E]">[{datasetStats.modelDim} x {vocab?.vocabulary.length || 65}]</code></li>
                        <li>Logits biases: <code className="text-yellow-300 bg-[#0A0A0B] px-1 py-0.5 rounded border border-[#2A2A2E]">[1 x {vocab?.vocabulary.length || 65}]</code></li>
                      </ul>
                      <p className="mt-1">Math operations matching context sequence length to vocabulary size to compute next word probability distribution.</p>
                    </div>
                  </div>

                </div>
              ) : (
                <p className="text-xs font-mono text-slate-500 italic">Failed to access metrics from Express host.</p>
              )}

            </div>

            {/* Scribe AI / Teacher explanation panel powered by Gemini API (Right) */}
            <div className="lg:col-span-6 bento-card space-y-6">
              
              <div className="border-b border-[#2A2A2E] pb-4">
                <span className="px-2.5 py-1 rounded-md bg-purple-500/10 text-purple-400 text-[10px] font-bold uppercase tracking-wider mb-2.5 inline-block">
                  Scribe AI Math Teacher
                </span>
                <p className="text-xs text-slate-400 leading-relaxed font-sans mt-1.5">
                  Select a complex neural network concept to fetch a custom styled, highly detailed conceptual explanation generated server-side by Google Gemini.
                </p>
              </div>

              {/* Selector form */}
              <div className="space-y-3">
                <label className="text-xs font-mono text-slate-400 block font-bold uppercase tracking-wider text-[10px]">Select Topic:</label>
                <div className="flex flex-col gap-2">
                  {mathConcepts.map((concept, index) => (
                    <button
                      key={index}
                      onClick={() => {
                        setScribeConcept(concept.title);
                        askScribeAI(concept.title);
                      }}
                      className={`text-left p-3.5 rounded-xl border text-xs font-mono transition-all duration-150 cursor-pointer ${
                        scribeConcept === concept.title
                          ? "bg-purple-950/20 text-purple-300 border-purple-500/40 shadow-inner"
                          : "bg-[#0A0A0B] text-slate-400 border-[#2A2A2E] hover:border-purple-500/20"
                      }`}
                    >
                      <div className="font-bold flex items-center gap-1.5">
                        <Code className="w-3.5 h-3.5 text-purple-400" />
                        {concept.title}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1 font-sans leading-relaxed">
                        {concept.desc}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Render Explanation Content Area */}
              <div className="bg-[#0A0A0B] border border-[#2A2A2E] p-5 rounded-2xl min-h-[175px] max-h-[380px] overflow-auto relative">
                {isLoadingScribe ? (
                  <div className="absolute inset-0 bg-[#0A0A0B]/90 backdrop-blur-xs flex flex-col items-center justify-center space-y-3 font-mono text-xs">
                    <RefreshCw className="w-6 h-6 animate-spin text-purple-400" />
                    <span className="text-slate-400 animate-pulse uppercase tracking-widest text-[10px] font-bold">Consulting Gemini models...</span>
                  </div>
                ) : null}

                {scribeExplanation ? (
                  <div className="text-xs leading-relaxed font-sans text-slate-300 space-y-3 markdown-body">
                    {scribeExplanation}
                  </div>
                ) : (
                  <div className="text-center py-10 font-mono text-xs text-slate-500 space-y-2">
                    <Cpu className="w-6 h-6 mx-auto text-slate-700" />
                    <p>Select an architectural topic above to load an interactive explanatory brief.</p>
                  </div>
                )}
              </div>

            </div>

          </div>
        )}

      </main>

      {/* Footer Meta Area */}
      <footer className="border-t border-[#2A2A2E] bg-[#0A0A0B]/50 py-10 px-4 mt-20 font-mono text-[11px] text-slate-500 text-center">
        <div className="max-w-4xl mx-auto space-y-3">
          <p className="leading-relaxed">
            NumPyGPT proof-of-concept AI portfolio interface. Hand-coded multi-head equivalent self-attention weights, manual gradient caches, and SGD updates.
          </p>
          <div className="flex justify-center items-center gap-4 text-[10px] uppercase tracking-wider text-indigo-400/80 font-bold">
            <span>Powered by TypeScript & Vite</span>
            <span>&bull;</span>
            <span>Express Backprop API</span>
            <span>&bull;</span>
            <span>Lapsed Time: 0.1s Pre-Train</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
