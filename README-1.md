# Mechanistic Interpretability Framework for LLM Sleeper Agent Detection

An open-source implementation of a lightweight, active-defense architecture designed to identify and intercept latent backdoor triggers within Large Language Models (LLMs) before output decoding begins. 

This repository provides the engineering modules required to map vector divergence across internal transformer residual stream hooks, optimize logistic linear classification probes, and execute runtime token containment routines.

---

## 🚀 Core Features
* **Zero-Latency Latent Interception:** Intercepts forward pass activations directly at layer $\ell=2$ to avoid generation-stage decoding overhead.
* **Non-Destructive Containment:** Neutralizes verified adversarial state transitions by zeroing out targeted residual tensor dimensions without modifying or fine-tuning underlying network weights.
* **Transparent Architecture Inspection:** Built on top of `TransformerLens` for clean extraction and caching of multi-dimensional internal representation matrices across all processing layers.

---

## 🛠️ System Architecture

Our active defense network operates across three concurrent pipelines:
1. **Parallel Activation Caching:** Evaluates contextual strings to capture baseline ($\mathcal{C}_{\text{clean}}$) and perturbed ($\mathcal{C}_{\text{triggered}}$) residual tensor profiles via the `run_with_cache` interface.
2. **Linear Probe Boundary Optimization:** Maximizes linear separability over high-dimensional internal arrays using custom single-layer logistic classifiers.
3. **Runtime Interception Hook:** Continuously computes an empirical anomaly score ($\alpha$). When $\alpha \geq \tau$ (operational safety threshold, default $\tau = 0.85$), an inline intervention zero-fills active tensor channels, enforcing a safe null state default before generation initializes.

---

## 📦 Directory Structure

```text
llm-sleeper-agent-detector/
│
├── README.md               # Project documentation and reproduction guide
├── paper/                  # LaTeX artifacts and academic drafts
│   └── sleeper_agent_paper.tex
│
├── notebooks/              # Sandbox prototyping and exploratory analysis
│   └── phase1_setup.ipynb
│
├── src/                    # Modular, production-grade source code
│   └── probe.py            # Primary extraction, training, and hooking script
│
├── figures/                # High-resolution architectural and performance plots
│   └── (all your plots go here)
│
└── data/                   # Balanced prompt distribution environment
    └── prompts/
        ├── clean_prompts.txt
        └── triggered_prompts.txt
🔧 Installation & DependenciesEnsure your execution environment features an active NVIDIA GPU with a minimum VRAM envelope of 15GB (e.g., commodity NVIDIA T4 instances). Clone the repository and install the dependencies:Bashgit clone [https://github.com/yourusername/llm-sleeper-agent-detector.git](https://github.com/yourusername/llm-sleeper-agent-detector.git)
cd llm-sleeper-agent-detector
pip install torch numpy transformerlens
⚡ Execution & VerificationTo train the linear probe, extract the residual layer features from the dataset, and simulate real-time runtime interventions, execute the primary source module:Bashpython src/probe.py
Expected Output LogsUpon a successful execution pass, your terminal will trace the extraction pipeline and demonstrate absolute linear separation capabilities:Plaintext[*] Initializing model: pythia-410m via TransformerLens...
[+] Loaded 10 clean and 10 triggered prompts.
[*] Extracting internal representations from layer 2...
[*] Optimizing linear classification probe boundary...
[+] Probe boundary training complete.

--- Testing Control Sequence ---
[*] Evaluating sequence processing...
[Inference Hook] Live Anomaly Score (alpha): 0.0012
[+] Stream cleared. Proceeding to standard generation.

--- Testing Malicious Adversarial Payload Sequence ---
[*] Evaluating sequence processing...
[Inference Hook] Live Anomaly Score (alpha): 0.9948
[🚨 INTERCEPTION] Breach confirmed (alpha >= 0.85). Executing containment sequence...
📊 Empirical Metrics SummaryTarget Family: EleutherAI Pythia-410m (Frozen parameters)Optimization Base: Layer $\ell=2$ Post-Residual Stream BlockReceiver Operating Characteristic (ROC AUC): 1.0000True Positive Rate (TPR): 100.00%False Positive Rate (FPR): 0.00%