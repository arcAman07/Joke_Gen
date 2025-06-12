#!/usr/bin/env python3
"""
Joke Generator with Weighted Metrics - Version 3
Adds multi-dimensional weighted ranking system (Humor + Diversity + Novelty)
"""

import os
import sys
import random
import itertools
import hashlib
import time
import re
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict, Counter

try:
    import openai
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

# Configuration
TEMPERATURE = 0.9
BRANCH_FACTOR = 5
MAX_DEPTH = 2
ELO_K_FACTOR = 32
INITIAL_ELO = 1200

# Weighted metrics configuration
HUMOR_WEIGHT = 0.6
DIVERSITY_WEIGHT = 0.3
NOVELTY_WEIGHT = 0.1

# Pruning configuration
MIN_OBSERVATION_QUALITY = 0.3
MIN_JOKE_QUALITY = 0.2
BEAM_WIDTH = 8

@dataclass
class WeightedMetrics:
    humor_score: float = 0.0
    diversity_contribution: float = 0.0
    novelty_score: float = 0.0
    taxonomy: Dict[str, float] = field(default_factory=dict)
    structural_pattern: str = ""
    weighted_score: float = 0.0
    quality_score: float = 0.0
    creativity_score: float = 0.0
    relevance_score: float = 0.0

@dataclass
class WeightedJoke:
    text: str
    path: List[str]
    elo_rating: float = INITIAL_ELO
    id: str = ""
    metrics: WeightedMetrics = field(default_factory=WeightedMetrics)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:8]

class OpenAIClient:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def complete(self, prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=n,
                max_tokens=300
            )
            return [choice.message.content.strip() for choice in response.choices]
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ["Error generating response"]
    
    def get_embedding(self, text: str) -> List[float]:
        """Get text embedding for similarity analysis"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except:
            # Fallback: simple hash-based embedding
            text_hash = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(48):
                byte_group = text_hash[i*4:(i+1)*4] if i*4 < len(text_hash) else text_hash[-4:]
                value = int.from_bytes(byte_group, 'big') / (2**32)
                embedding.append(value)
            return embedding

# Global client instance
llm_client = None

def llm_complete(prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
    global llm_client
    if llm_client is None:
        llm_client = OpenAIClient()
    return llm_client.complete(prompt, n, temperature)

def get_embedding(text: str) -> List[float]:
    global llm_client
    if llm_client is None:
        llm_client = OpenAIClient()
    return llm_client.get_embedding(text)

class NoveltyAnalyzer:
    """Analyzes novelty by detecting common patterns and semantic similarity"""
    
    def __init__(self):
        # Common joke patterns that should be penalized
        self.common_patterns = {
            "cross_road": r"why did .* cross the road",
            "walks_into_bar": r"walks into a bar",
            "lightbulb": r"how many .* to change a lightbulb",
            "knock_knock": r"knock knock",
            "difference_between": r"what's the difference between",
            "call_a": r"what do you call a",
            "says_to": r"(doctor|lawyer|priest) says to",
            "yo_mama": r"yo mama",
            "chuck_norris": r"chuck norris",
        }
        
        self.pattern_penalties = {
            "cross_road": 0.7,
            "walks_into_bar": 0.7,
            "lightbulb": 0.6,
            "knock_knock": 0.8,
            "difference_between": 0.5,
            "call_a": 0.5,
            "says_to": 0.6,
            "yo_mama": 0.4,
            "chuck_norris": 0.4,
        }
    
    def check_common_patterns(self, joke_text: str) -> Tuple[float, str]:
        """Check for common joke patterns and return penalty"""
        joke_lower = joke_text.lower()
        
        for pattern_name, pattern in self.common_patterns.items():
            if re.search(pattern, joke_lower):
                penalty = self.pattern_penalties[pattern_name]
                return penalty, pattern_name
        
        return 1.0, "original"
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_novelty_score(self, joke: WeightedJoke, all_jokes: List[WeightedJoke]) -> float:
        """Calculate overall novelty score combining pattern and semantic analysis"""
        # Pattern-based novelty
        pattern_score, pattern_type = self.check_common_patterns(joke.text)
        joke.metrics.structural_pattern = pattern_type
        
        # Semantic novelty (similarity to other jokes)
        embedding = get_embedding(joke.text)
        semantic_novelty = 1.0
        
        if len(all_jokes) > 1:
            similarities = []
            for other in all_jokes:
                if other.id != joke.id:
                    other_embedding = get_embedding(other.text)
                    sim = self.cosine_similarity(embedding, other_embedding)
                    similarities.append(sim)
            
            if similarities:
                max_similarity = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)
                # Use both max and average similarity
                semantic_novelty = 1.0 - (0.7 * avg_similarity + 0.3 * max_similarity)
        
        # Combine pattern and semantic novelty
        final_novelty = pattern_score * semantic_novelty
        joke.metrics.novelty_score = final_novelty
        
        return final_novelty

class DiversityAnalyzer:
    """Analyzes diversity across multiple humor dimensions"""
    
    def __init__(self):
        self.humor_dimensions = [
            "wordplay",      # Puns, double meanings
            "absurdity",     # Surreal, nonsensical
            "observational", # Everyday life humor
            "dark",          # Morbid or taboo
            "intellectual",  # Requires knowledge
            "slapstick",     # Physical comedy
            "meta",          # Self-referential
            "irony",         # Opposite of expected
            "callback",      # References earlier content
            "misdirection"   # Unexpected punchline
        ]
    
    def analyze_joke_taxonomy(self, joke: WeightedJoke) -> Dict[str, float]:
        """Classify joke across humor dimensions"""
        prompt = f"""Analyze this joke across humor dimensions and rate each from 0-10:

Joke: {joke.text}

Rate each dimension:
- wordplay: puns, double meanings, language play
- absurdity: surreal, nonsensical, illogical elements
- observational: everyday life situations and behaviors
- dark: morbid, taboo, or edgy content
- intellectual: requires knowledge or references
- slapstick: physical comedy or visual humor
- meta: self-referential or breaking fourth wall
- irony: saying opposite of what's meant
- callback: references to earlier content
- misdirection: unexpected punchline or twist

Format: dimension:score (one per line, e.g., wordplay:7)"""
        
        response = llm_complete(prompt, temperature=0.1)[0]
        
        taxonomy = {}
        for line in response.split('\n'):
            if ':' in line:
                try:
                    dim, score = line.split(':', 1)
                    dim = dim.strip().lower()
                    score = float(score.strip())
                    if dim in self.humor_dimensions:
                        taxonomy[dim] = min(10, max(0, score))
                except:
                    continue
        
        # Fill in missing dimensions with default values
        for dim in self.humor_dimensions:
            if dim not in taxonomy:
                taxonomy[dim] = 0.0
        
        joke.metrics.taxonomy = taxonomy
        return taxonomy
    
    def calculate_diversity_score(self, jokes: List[WeightedJoke]) -> float:
        """Calculate overall diversity across all jokes"""
        if len(jokes) < 2:
            return 0.0
        
        dimension_values = defaultdict(list)
        
        # Collect values for each dimension across all jokes
        for joke in jokes:
            if not joke.metrics.taxonomy:
                self.analyze_joke_taxonomy(joke)
            
            for dim, score in joke.metrics.taxonomy.items():
                dimension_values[dim].append(score)
        
        # Calculate variance for each dimension
        total_variance = 0.0
        for dim, values in dimension_values.items():
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                total_variance += variance
        
        # Average variance across dimensions
        avg_variance = total_variance / len(self.humor_dimensions)
        
        # Convert to diversity score (higher variance = more diverse)
        diversity_score = 1 - math.exp(-avg_variance / 10)
        
        return diversity_score
    
    def calculate_joke_diversity_contribution(self, joke: WeightedJoke, other_jokes: List[WeightedJoke]) -> float:
        """Calculate how much this joke contributes to overall diversity"""
        if not other_jokes:
            return 1.0
        
        if not joke.metrics.taxonomy:
            self.analyze_joke_taxonomy(joke)
        
        # Count distribution of dominant humor types in other jokes
        dimension_counts = defaultdict(int)
        for other in other_jokes:
            if other.id != joke.id and other.metrics.taxonomy:
                # Find dominant dimension for this joke
                if other.metrics.taxonomy:
                    dominant_dim = max(other.metrics.taxonomy.items(), key=lambda x: x[1])[0]
                    dimension_counts[dominant_dim] += 1
        
        # Find this joke's dominant dimension
        if joke.metrics.taxonomy:
            joke_dominant = max(joke.metrics.taxonomy.items(), key=lambda x: x[1])[0]
            
            # Rarity score: less common dimensions get higher scores
            total_others = len(other_jokes)
            if total_others > 0:
                rarity_score = 1.0 - (dimension_counts[joke_dominant] / total_others)
            else:
                rarity_score = 1.0
        else:
            rarity_score = 0.5
        
        joke.metrics.diversity_contribution = rarity_score
        return rarity_score

class QualityEvaluator:
    """Enhanced quality evaluation with caching"""
    
    def __init__(self):
        self.evaluation_cache = {}
    
    def evaluate_joke_quality(self, joke: str) -> Tuple[float, float, float]:
        """Evaluate joke on multiple dimensions"""
        cache_key = f"joke_{hash(joke)}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        prompt = f"""Evaluate this joke on three dimensions (0-10 each):

Joke: {joke}

Rate:
1. Quality: How funny/well-crafted is it? Consider timing, setup, punchline
2. Creativity: How original/unexpected? Avoid clichés and common patterns
3. Relevance: How well does it relate to its topic and make sense?

Format: quality:X creativity:Y relevance:Z"""
        
        try:
            response = llm_complete(prompt, temperature=0.1)[0]
            
            quality = creativity = relevance = 5.0
            for line in response.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    try:
                        val = float(value.strip())
                        if 'quality' in key:
                            quality = val
                        elif 'creativity' in key:
                            creativity = val
                        elif 'relevance' in key:
                            relevance = val
                    except:
                        continue
            
            # Normalize to [0,1]
            quality = max(0.0, min(1.0, quality / 10.0))
            creativity = max(0.0, min(1.0, creativity / 10.0))
            relevance = max(0.0, min(1.0, relevance / 10.0))
            
        except:
            quality = creativity = relevance = 0.5
        
        result = (quality, creativity, relevance)
        self.evaluation_cache[cache_key] = result
        return result

class WeightedRankingSystem:
    """Multi-dimensional weighted ranking system"""
    
    def __init__(self, humor_weight: float = HUMOR_WEIGHT, 
                 diversity_weight: float = DIVERSITY_WEIGHT,
                 novelty_weight: float = NOVELTY_WEIGHT):
        self.humor_weight = humor_weight
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        
        self.novelty_analyzer = NoveltyAnalyzer()
        self.diversity_analyzer = DiversityAnalyzer()
        self.quality_evaluator = QualityEvaluator()
        
        print(f"Weighted ranking: {humor_weight:.1f} humor + {diversity_weight:.1f} diversity + {novelty_weight:.1f} novelty")
    
    def calculate_weighted_score(self, joke: WeightedJoke, all_jokes: List[WeightedJoke]) -> float:
        """Calculate final weighted score combining all dimensions"""
        
        # 1. Humor score (normalized ELO rating)
        humor_score = joke.elo_rating / 1500.0  # Normalize around 1.0
        joke.metrics.humor_score = humor_score
        
        # 2. Diversity contribution
        diversity_contribution = self.diversity_analyzer.calculate_joke_diversity_contribution(joke, all_jokes)
        
        # 3. Novelty score
        novelty_score = self.novelty_analyzer.calculate_novelty_score(joke, all_jokes)
        
        # 4. Basic quality metrics (for additional context)
        quality, creativity, relevance = self.quality_evaluator.evaluate_joke_quality(joke.text)
        joke.metrics.quality_score = quality
        joke.metrics.creativity_score = creativity
        joke.metrics.relevance_score = relevance
        
        # 5. Calculate weighted combination
        weighted_score = (
            self.humor_weight * humor_score +
            self.diversity_weight * diversity_contribution +
            self.novelty_weight * novelty_score
        )
        
        joke.metrics.weighted_score = weighted_score
        
        return weighted_score
    
    def rank_jokes(self, jokes: List[WeightedJoke]) -> List[WeightedJoke]:
        """Rank jokes using weighted scoring system"""
        print(f"Calculating weighted scores for {len(jokes)} jokes...")
        
        # Calculate weighted scores
        for joke in jokes:
            self.calculate_weighted_score(joke, jokes)
        
        # Sort by weighted score
        jokes.sort(key=lambda j: j.metrics.weighted_score, reverse=True)
        
        # Calculate overall diversity for reporting
        overall_diversity = self.diversity_analyzer.calculate_diversity_score(jokes)
        print(f"Overall diversity score: {overall_diversity:.2f}")
        
        return jokes

class WeightedPlanSearchGenerator:
    """PlanSearch generator optimized for weighted ranking"""
    
    def __init__(self, topic: str, branch_factor: int = BRANCH_FACTOR, max_depth: int = MAX_DEPTH):
        self.topic = topic
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.observation_tree = {}
        self.jokes = []
        self.quality_evaluator = QualityEvaluator()
    
    def generate_observations(self, context: Optional[List[str]] = None) -> List[str]:
        """Generate diverse, high-quality observations"""
        if context:
            prompt = f"""Given the topic '{self.topic}' and these observations:
{chr(10).join(f'- {obs}' for obs in context)}

Generate {self.branch_factor} NEW observations that combine these ideas creatively.
Focus on:
- Unexpected connections and contradictions
- Wordplay and pun opportunities  
- Cultural references and stereotypes
- Technical or specific details
- Absurd or surreal angles

Make them specific, creative, and rich with comedy potential."""
        else:
            prompt = f"""Generate {self.branch_factor} diverse, creative observations about '{self.topic}'.

Include a mix of:
- Common stereotypes and assumptions
- Technical or insider knowledge
- Cultural references and wordplay opportunities
- Absurd or unexpected angles
- Contradictions and subversions

Make each observation specific and comedy-rich."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        observations = []
        
        for response in responses:
            lines = response.split('\n')
            for line in lines:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith(('-', '•')):
                    # Remove numbering
                    if cleaned.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        cleaned = cleaned[2:].strip()
                    if cleaned:
                        observations.append(cleaned)
        
        return observations[:self.branch_factor]
    
    def build_observation_tree(self):
        """Build observation tree focused on diversity"""
        print(f"Building weighted observation tree for '{self.topic}'...")
        
        # Level 1: Basic diverse observations
        level1_obs = self.generate_observations()
        self.observation_tree[1] = level1_obs
        print(f"Level 1: {len(level1_obs)} diverse observations")
        
        if self.max_depth < 2:
            return
        
        # Level 2: Strategic combinations for maximum diversity
        level2_obs = []
        
        # Try all combinations but limit to best ones
        for obs_pair in itertools.combinations(level1_obs, 2):
            derived = self.generate_observations(context=list(obs_pair))
            # Take only the first derived observation to maintain focus
            level2_obs.extend([(obs_pair, d) for d in derived[:1]])
        
        self.observation_tree[2] = level2_obs
        print(f"Level 2: {len(level2_obs)} combined observations")
    
    def generate_joke_from_path(self, observation_path: List[str]) -> str:
        """Generate high-quality joke optimized for weighted metrics"""
        prompt = f"""Topic: {self.topic}

Observations:
{chr(10).join(f'- {obs}' for obs in observation_path)}

Create a single, excellent joke that cleverly uses these observations.

Requirements:
- Make it genuinely funny with good timing
- Avoid clichéd formats (crossing roads, bar jokes, etc.)
- Use wordplay, misdirection, or unexpected connections
- Keep it concise and punchy
- Ensure it's original and creative

Format: Just the joke text, no explanations."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        return responses[0].strip()
    
    def generate_all_jokes(self) -> List[WeightedJoke]:
        """Generate jokes optimized for weighted ranking"""
        print("Generating jokes optimized for weighted metrics...")
        jokes = []
        
        # Jokes from level 1 observations
        for obs in self.observation_tree.get(1, []):
            joke_text = self.generate_joke_from_path([obs])
            jokes.append(WeightedJoke(text=joke_text, path=[obs]))
        
        # Jokes from level 2 observation combinations
        for (obs_pair, derived_obs) in self.observation_tree.get(2, []):
            path = list(obs_pair) + [derived_obs]
            joke_text = self.generate_joke_from_path(path)
            jokes.append(WeightedJoke(text=joke_text, path=path))
        
        # Strategic contradiction jokes for diversity
        for obs in self.observation_tree.get(1, [])[:4]:  # Limit to avoid repetition
            prompt = f"""Topic: {self.topic}
Observation: {obs}

Generate a clever, original joke that CONTRADICTS or subverts this observation.
Avoid clichés and make it genuinely witty and unexpected.
Just the joke, no explanation."""
            responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
            jokes.append(WeightedJoke(text=responses[0].strip(), path=[obs, "[contradicted]"]))
        
        print(f"Generated {len(jokes)} jokes for weighted evaluation")
        self.jokes = jokes
        return jokes

class WeightedEloSystem:
    """ELO system optimized for weighted ranking"""
    
    def __init__(self, k_factor: float = ELO_K_FACTOR):
        self.k_factor = k_factor
        self.comparison_count = 0
        self.comparison_cache = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, joke_a: WeightedJoke, joke_b: WeightedJoke, winner: str) -> Tuple[float, float]:
        expected_a = self.expected_score(joke_a.elo_rating, joke_b.elo_rating)
        expected_b = 1 - expected_a
        
        if winner == "A":
            score_a, score_b = 1, 0
        elif winner == "B":
            score_a, score_b = 0, 1
        else:
            score_a, score_b = 0.5, 0.5
        
        new_rating_a = joke_a.elo_rating + self.k_factor * (score_a - expected_a)
        new_rating_b = joke_b.elo_rating + self.k_factor * (score_b - expected_b)
        
        return new_rating_a, new_rating_b
    
    def judge_pair(self, joke_a: WeightedJoke, joke_b: WeightedJoke) -> str:
        """Enhanced judging for humor quality"""
        cache_key = (joke_a.id, joke_b.id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        prompt = f"""Compare these jokes and determine which is funnier overall:

Joke A: {joke_a.text}

Joke B: {joke_b.text}

Consider:
- Humor quality and timing
- Originality and creativity
- Cleverness of wordplay or concept
- Overall comedic impact

Reply with only 'A' or 'B' for the funnier joke."""
        
        responses = llm_complete(prompt, n=1, temperature=0.1)
        winner = responses[0].strip().upper()
        
        if winner not in ["A", "B"]:
            winner = random.choice(["A", "B"])
        
        self.comparison_cache[cache_key] = winner
        return winner
    
    def run_tournament(self, jokes: List[WeightedJoke], rounds: int = 3) -> List[WeightedJoke]:
        """Run ELO tournament for humor assessment"""
        print(f"Running weighted ELO tournament with {len(jokes)} jokes...")
        
        n_jokes = len(jokes)
        if n_jokes < 2:
            return jokes
        
        total_comparisons = rounds * n_jokes
        
        for i in range(total_comparisons):
            if i % 5 == 0:
                print(f"  Comparison {i}/{total_comparisons}...")
            
            idx_a, idx_b = random.sample(range(n_jokes), 2)
            joke_a, joke_b = jokes[idx_a], jokes[idx_b]
            
            cache_key = (joke_a.id, joke_b.id)
            if cache_key in self.comparison_cache:
                continue
            
            winner = self.judge_pair(joke_a, joke_b)
            new_rating_a, new_rating_b = self.update_ratings(joke_a, joke_b, winner)
            
            jokes[idx_a].elo_rating = new_rating_a
            jokes[idx_b].elo_rating = new_rating_b
            
            self.comparison_count += 1
        
        jokes.sort(key=lambda j: j.elo_rating, reverse=True)
        print(f"ELO tournament complete. {self.comparison_count} comparisons made.")
        return jokes

def generate_weighted_jokes(topic: str, top_n: int = 5, 
                          humor_weight: float = HUMOR_WEIGHT,
                          diversity_weight: float = DIVERSITY_WEIGHT,
                          novelty_weight: float = NOVELTY_WEIGHT) -> List[WeightedJoke]:
    """Main function to generate jokes with weighted ranking"""
    
    # Generate jokes
    generator = WeightedPlanSearchGenerator(topic)
    generator.build_observation_tree()
    jokes = generator.generate_all_jokes()
    
    # ELO ranking for humor assessment
    elo_system = WeightedEloSystem()
    elo_ranked = elo_system.run_tournament(jokes, rounds=3)
    
    # Weighted ranking combining humor, diversity, and novelty
    ranking_system = WeightedRankingSystem(humor_weight, diversity_weight, novelty_weight)
    final_ranked = ranking_system.rank_jokes(elo_ranked)
    
    return final_ranked[:top_n]

def interactive_mode():
    """Interactive mode with weighted metrics display"""
    print("\n🎭 Weighted Metrics AI Joke Generator (Version 3)")
    print("=" * 60)
    print(f"Features: Multi-Dimensional Ranking (Humor + Diversity + Novelty)")
    print(f"Weights: {HUMOR_WEIGHT:.1f} humor + {DIVERSITY_WEIGHT:.1f} diversity + {NOVELTY_WEIGHT:.1f} novelty")
    
    while True:
        topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using the weighted metrics joke generator! 😄")
            break
        
        if not topic:
            print("Please enter a valid topic.")
            continue
        
        # Optional: Allow custom weights
        use_custom = input("Use custom weights? (y/n): ").strip().lower()
        if use_custom == 'y':
            try:
                h = float(input(f"Humor weight (default {HUMOR_WEIGHT}): ") or HUMOR_WEIGHT)
                d = float(input(f"Diversity weight (default {DIVERSITY_WEIGHT}): ") or DIVERSITY_WEIGHT)
                n = float(input(f"Novelty weight (default {NOVELTY_WEIGHT}): ") or NOVELTY_WEIGHT)
                
                # Normalize weights
                total = h + d + n
                h, d, n = h/total, d/total, n/total
                print(f"Using normalized weights: {h:.2f} humor + {d:.2f} diversity + {n:.2f} novelty")
            except:
                h, d, n = HUMOR_WEIGHT, DIVERSITY_WEIGHT, NOVELTY_WEIGHT
                print("Using default weights")
        else:
            h, d, n = HUMOR_WEIGHT, DIVERSITY_WEIGHT, NOVELTY_WEIGHT
        
        print(f"\nGenerating jokes about '{topic}' with weighted ranking...")
        print("This may take 2-3 minutes...\n")
        
        try:
            start_time = time.time()
            top_jokes = generate_weighted_jokes(topic, top_n=5, 
                                              humor_weight=h, diversity_weight=d, novelty_weight=n)
            elapsed_time = time.time() - start_time
            
            print(f"\n🌟 Top 5 Weighted-Ranked Jokes about '{topic}':")
            print("=" * 60)
            
            for i, joke in enumerate(top_jokes, 1):
                print(f"\n{i}. {joke.text}")
                print(f"   📊 Weighted Score: {joke.metrics.weighted_score:.3f}")
                print(f"   Components:")
                print(f"     • Humor (ELO): {joke.metrics.humor_score:.2f} (rating: {joke.elo_rating:.0f})")
                print(f"     • Diversity: {joke.metrics.diversity_contribution:.2f}")
                print(f"     • Novelty: {joke.metrics.novelty_score:.2f} ({joke.metrics.structural_pattern})")
                
                if joke.metrics.taxonomy:
                    top_dims = sorted(joke.metrics.taxonomy.items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                    dims_str = ', '.join(f'{d[0]}({d[1]:.1f})' for d in top_dims)
                    print(f"     • Style: {dims_str}")
            
            print(f"\n⏱️  Generated in {elapsed_time:.1f} seconds")
            
        except Exception as e:
            print(f"\nError generating jokes: {e}")
            print("Please try again with a different topic.")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] != "interactive":
        topic = sys.argv[1]
        print(f"\n🎭 Weighted Metrics Joke Generation for: {topic}")
        print("=" * 60)
        
        top_jokes = generate_weighted_jokes(topic, top_n=5)
        
        print("\n🌟 Top 5 Jokes with Weighted Analysis:")
        print("=" * 60)
        
        for i, joke in enumerate(top_jokes, 1):
            print(f"\n{i}. {joke.text}")
            print(f"   Weighted Score: {joke.metrics.weighted_score:.3f}")
            print(f"   Breakdown: Humor={joke.metrics.humor_score:.2f}, "
                  f"Diversity={joke.metrics.diversity_contribution:.2f}, "
                  f"Novelty={joke.metrics.novelty_score:.2f}")
            print(f"   Pattern: {joke.metrics.structural_pattern}")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()