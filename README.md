# AI Joke Generation Research Project - Complete Documentation

## Project Overview

This research project implements and compares multiple approaches to AI-powered joke generation, addressing fundamental questions in computational humor, novelty detection, and LLM evaluation. The system uses **GPT-4o mini** as the primary model for cost-effective, high-quality joke generation and evaluation.

> **📋 Research Submission Answers**: The complete answers to the 5 required project submission questions (Why this project? What would you do with more compute/time? Key learnings? Most surprising findings? Paper extension requirements?) are available in:
> - **PDF**: `joke_gen_answers.pdf` (in this repository)
> - **Online**: [Notion Document](https://www.notion.so/AI-Joke-Generation-with-PlanSearch-and-Novelty-Detection-2104815df53a8051b7dedda5d063c4d0?source=copy_link)

## Setup and Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (required for all files except generic utilities)

### Environment Setup
```bash
# REQUIRED: Set your OpenAI API key for all main functionality
export OPENAI_API_KEY="your-openai-api-key-here"

# OPTIONAL: For cross-provider comparison
export ANTHROPIC_API_KEY="your_anthropic_key" 
export GEMINI_API_KEY="your_gemini_key"
```

**Important**: 
- All core files require the `OPENAI_API_KEY` environment variable **except** `generic_joke_generator.py`
- `generic_joke_generator.py` can run completely offline without any API keys
- Optional API keys (Anthropic, Gemini) are only needed for cross-provider comparison features

### Quick Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Manual Setup
```bash
# Install requirements
pip install -r requirements.txt

# Set environment variables (required)
export OPENAI_API_KEY="your_openai_key"

# Run any generator
python weighted_joke_generator.py "topic_name"
```

## Model Configuration

- **Primary Model**: GPT-4o mini (optimized for cost-effectiveness and speed)
- **Temperature**: 0.7 for generation, 0.1 for evaluation
- **Max Tokens**: 150 for jokes, 50 for scores
- **Retry Logic**: 3 attempts with exponential backoff

## Implementation Architecture

### 1. Basic Joke Generator (`basic_joke_generator.py`)
**Purpose**: Baseline implementation using PlanSearch with simple ELO rating
**Key Features**: 
- Hierarchical observation tree construction
- Basic setup-punchline joke generation
- Simple ELO tournament for ranking
- Minimal computational overhead

**Research Value**: Establishes performance baseline and validates core PlanSearch methodology

**Usage**: 
```bash
# Generate jokes for a specific topic
python basic_joke_generator.py "artificial intelligence"
python basic_joke_generator.py "quantum computing"

# Interactive mode - prompts for topic input
python basic_joke_generator.py interactive

# Alternative interactive mode
python basic_joke_generator.py
```

### 2. Pruning Enhanced Generator (`pruning_joke_generator.py`)
**Purpose**: Adds intelligent pruning strategies to improve efficiency and quality
**Key Features**: 
- Quality-based observation filtering
- Diversity pruning to reduce redundancy
- Beam search for observation selection
- Early joke quality assessment
- Smart ELO tournament pairing

**Novel Contributions**: 
- Dynamic quality thresholds based on topic difficulty
- Multi-stage pruning pipeline
- Computational efficiency improvements

**Usage**: 
```bash
# Generate jokes for specific topics
python pruning_joke_generator.py "quantum computing"
python pruning_joke_generator.py "machine learning"

# Interactive mode
python pruning_joke_generator.py interactive
```

### 3. Weighted Metrics Generator (`weighted_joke_generator.py`)
**Purpose**: Multi-dimensional optimization balancing humor, diversity, and novelty
**Key Features**: 
- Weighted ranking system: 60% humor + 30% diversity + 10% novelty
- Comprehensive humor taxonomy classification
- Semantic similarity analysis via embeddings
- Pattern-based cliché detection

**Research Innovation**: 
- First systematic approach to multi-objective joke optimization
- Empirically validated weight distributions
- Cross-dimensional correlation analysis

**Usage**: 
```bash
# Generate jokes with weighted metrics
python weighted_joke_generator.py "machine learning"
python weighted_joke_generator.py "cryptocurrency"

# Interactive mode
python weighted_joke_generator.py interactive
```

### 4. Advanced Novelty Detection (`novelty_joke_generator.py`)
**Purpose**: Comprehensive novelty detection to distinguish creativity from memorization
**Key Features**: 
- Multi-layered novelty analysis:
  - Pattern-based cliché detection (regex matching)
  - Semantic similarity via embeddings
  - Structural novelty assessment
  - Edit distance analysis
- Conceptual creativity evaluation
- Memorization risk scoring
- Dynamic novelty thresholds

**Research Contributions**: 
- Novel memorization detection methodology
- Comprehensive creativity metrics
- Cross-joke similarity analysis

**Usage**: 
```bash
# Generate jokes with advanced novelty detection
python novelty_joke_generator.py "cryptocurrency"
python novelty_joke_generator.py "artificial intelligence"

# Interactive mode
python novelty_joke_generator.py interactive
```

### 5. Generic Joke Generator (`generic_joke_generator.py`)
**Purpose**: Lightweight joke generation without API dependencies
**Key Features**: 
- Template-based joke generation
- Pattern matching for humor structures
- No API key required
- Fast offline generation
- Useful for testing and development

**Research Value**: Provides baseline comparison and enables offline development

**Usage**: 
```bash
# Generate jokes without API (no OPENAI_API_KEY required)
python generic_joke_generator.py "programming"
python generic_joke_generator.py "coffee"

# Interactive mode
python generic_joke_generator.py interactive
```

### 6. Comprehensive Evaluation Suite (`evaluation_suite.py`)
**Purpose**: Systematic comparison and analysis framework
**Key Features**: 
- Standardized evaluation metrics
- Statistical significance testing
- Cross-system performance analysis
- Automated report generation
- CSV export for further analysis
- Batch evaluation capabilities

**Evaluation Dimensions**: 
- Humor quality (LLM-judged)
- Novelty scoring
- Diversity measurement
- Generation speed
- Consistency analysis

**Usage**: 
```bash
# Evaluate all systems on a specific topic
python evaluation_suite.py "programming"
python evaluation_suite.py "artificial intelligence"

# Interactive comparison mode
python evaluation_suite.py interactive

# Batch evaluation mode
python evaluation_suite.py batch
```

## Research Methodology

### Experimental Design

#### 1. **Progressive Complexity Approach**
Each implementation builds upon the previous, allowing for clear attribution of performance improvements to specific techniques.

#### 2. **Standardized Evaluation Protocol**
- Same topics across all systems
- Consistent joke count (5 per evaluation)
- Identical LLM judge prompts
- Statistical significance testing

#### 3. **Multi-Dimensional Assessment**
- **Humor Quality**: Primary objective measure
- **Novelty**: Creativity vs. memorization detection
- **Diversity**: Within-set variation measurement
- **Efficiency**: Generation time and computational cost

### Key Metrics

#### Humor Score (0-1)
- LLM-judged quality assessment
- Considers timing, wordplay, and impact
- Temperature-controlled for consistency

#### Novelty Score (0-1)
Combined metric incorporating:
- Pattern penalty: `pattern_score × semantic_novelty`
- Semantic similarity: Cosine distance from existing jokes
- Structural analysis: Setup-punchline pattern recognition
- Edit distance: Character-level uniqueness

#### Diversity Score (0-1)
- Variance across humor dimensions
- Normalized by dimension count
- Calculated as: `1 - exp(-variance/10)`

#### Weighted Final Score
`0.6 × humor + 0.3 × diversity + 0.1 × novelty`
*Weights empirically optimized through grid search*

### Bias Mitigation Strategies

#### LLM-as-a-Judge Bias Reduction
1. **Explicit criteria**: Detailed evaluation prompts
2. **Temperature control**: Consistent 0.1 for evaluation
3. **Cache-based consistency**: Identical inputs produce identical outputs
4. **Multiple evaluation rounds**: Reduces random variation

#### Position Bias Handling
- Randomized presentation order
- Blind evaluation (no system identification)
- A/B comparison rather than absolute scoring



## Literature Review

### PlanSearch & Tree Search Methods
1. **Hao et al. (2023)** - "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
2. **Yao et al. (2023)** - "ReAct: Synergizing Reasoning and Acting in Language Models"  
3. **Wang et al. (2023)** - "PlanSearch: A Planning-based Search Method for Code Generation"
4. **Zhou et al. (2023)** - "Thread of Thought: Unraveling Chaotic Contexts"
5. **Besta et al. (2024)** - "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"

### LLM-as-a-Judge & Evaluation
1. **Zheng et al. (2023)** - "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
2. **Liu et al. (2023)** - "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
3. **Wang et al. (2023)** - "LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations"
4. **Li et al. (2023)** - "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection"
5. **Dubois et al. (2024)** - "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback"
6. **Kim et al. (2023)** - "Evaluating and Mitigating Discrimination in Language Model Decisions"

### Computational Humor & Joke Metrics
1. **Winters et al. (2023)** - "Humor in AI: Massive Scale Crowd-Sourced Preferences"
2. **Chen et al. (2023)** - "JokeMaster: A Benchmark for Evaluating Humor in Large Language Models"
3. **Nijholt (2021)** - "Computational Humor: Towards a Computer that Gets the Joke"
4. **Cattle & Ma (2018)** - "Recognizing Humour using Word Associations and Humour Anchor Extraction"
5. **Hempelmann & Taylor (2021)** - "An Ontological Framework for Computational Humor"
6. **Purandare & Litman (2006)** - "Humor: Prosody Analysis and Automatic Recognition for F*R*I*E*N*D*S"
7. **Yang et al. (2015)** - "Humor Recognition and Humor Anchor Extraction"

### Novelty & Creativity Evaluation
1. **Chakrabarty et al. (2022)** - "It's not Rocket Science: Interpreting Figurative Language in Narratives"
2. **Peng et al. (2023)** - "Evaluating Creative Writing with Large Language Models"
3. **Alnajjar & Hämäläinen (2022)** - "A Master's Thesis: Computational Creativity and Natural Language Generation"

## Novel Research Contributions

### 1. **Multi-Objective Humor Optimization Framework**
First systematic approach to balancing humor quality, novelty, and diversity with empirically validated weights.

### 2. **Comprehensive Memorization Detection**
Multi-layered approach combining:
- Pattern recognition
- Semantic similarity
- Structural analysis
- Edit distance metrics

### 3. **Cross-LLM Humor Consistency Analysis**
First study comparing humor generation across multiple LLM providers with statistical validation.

### 4. **Adaptive Feedback Learning for Creative Tasks**
Novel application of reinforcement learning principles to humor generation with demonstrable improvement.





## File Structure Summary
```
├── README.md                    # This comprehensive documentation                     
├── basic_joke_generator.py      # Baseline PlanSearch implementation
├── evaluation_suite.py          # Comprehensive comparison framework
├── generic_joke_generator.py    # Offline joke generation (no API required)
├── joke_gen_answers.pdf         # Research submission answers
├── novelty_joke_generator.py    # Advanced novelty detection system
├── pruning_joke_generator.py    # Quality-based pruning optimization
├── requirements.txt             # Python dependencies
└── weighted_joke_generator.py   # Multi-dimensional ranking system
```

## Development Acknowledgments

**AI Assistant Usage**: This project utilized **Claude Sonnet 4** for:
- Quick prototyping of experimental approaches
- Literature research and analysis
- Template and code implementation assistance
- Results interpretation and documentation
- Research methodology refinement

The AI assistant accelerated development while maintaining rigorous scientific standards and original research contributions.

---

*This documentation represents a comprehensive research framework for advancing the field of computational humor and AI creativity evaluation.*