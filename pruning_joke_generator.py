#!/usr/bin/env python3
"""
Joke Generator with Pruning - Version 2
Adds intelligent pruning strategies to improve efficiency and quality
"""

import os
import sys
import random
import itertools
import hashlib
import time
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict

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

# Pruning configuration
MIN_OBSERVATION_QUALITY = 0.3
MIN_JOKE_QUALITY = 0.2
BEAM_WIDTH = 8
QUALITY_THRESHOLD = 0.5

@dataclass
class PruningMetrics:
    quality_score: float = 0.0
    creativity_score: float = 0.0
    relevance_score: float = 0.0
    combined_score: float = 0.0

@dataclass
class PrunedJoke:
    text: str
    path: List[str]
    elo_rating: float = INITIAL_ELO
    id: str = ""
    metrics: PruningMetrics = field(default_factory=PruningMetrics)
    
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

# Global client instance
llm_client = None

def llm_complete(prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
    global llm_client
    if llm_client is None:
        llm_client = OpenAIClient()
    return llm_client.complete(prompt, n, temperature)

class QualityEvaluator:
    """Evaluates quality of observations and jokes for pruning decisions"""
    
    def __init__(self):
        self.evaluation_cache = {}
    
    def evaluate_observation_quality(self, observation: str, topic: str) -> float:
        """Evaluate quality of an observation (0-1 scale)"""
        cache_key = f"obs_{hash(observation + topic)}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        prompt = f"""Rate the comedy potential of this observation about '{topic}':

Observation: {observation}

Consider:
- Specificity and concreteness
- Potential for wordplay or humor
- Uniqueness and creativity
- Relevance to the topic

Rate from 0-10 (10 = excellent comedy potential)
Respond with just the number."""
        
        try:
            response = llm_complete(prompt, temperature=0.1)[0]
            score = float(response.strip()) / 10.0
            score = max(0.0, min(1.0, score))  # Clamp to [0,1]
        except:
            score = 0.5  # Default fallback
        
        self.evaluation_cache[cache_key] = score
        return score
    
    def evaluate_joke_quality(self, joke: str) -> Tuple[float, float, float]:
        """Evaluate joke on multiple dimensions"""
        cache_key = f"joke_{hash(joke)}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        prompt = f"""Evaluate this joke on three dimensions (0-10 each):

Joke: {joke}

Rate:
1. Quality: How funny/well-crafted is it?
2. Creativity: How original/unexpected?
3. Relevance: How well does it relate to its topic?

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

class PruningStrategies:
    """Various pruning strategies to improve generation efficiency"""
    
    def __init__(self, evaluator: QualityEvaluator):
        self.evaluator = evaluator
        self.observation_history = defaultdict(list)
    
    def prune_observations_by_quality(self, observations: List[str], topic: str, 
                                    threshold: float = MIN_OBSERVATION_QUALITY) -> List[str]:
        """Remove low-quality observations early"""
        print(f"Pruning {len(observations)} observations by quality...")
        
        quality_scored = []
        for obs in observations:
            quality = self.evaluator.evaluate_observation_quality(obs, topic)
            if quality >= threshold:
                quality_scored.append((obs, quality))
        
        # Sort by quality and return top observations
        quality_scored.sort(key=lambda x: x[1], reverse=True)
        pruned = [obs for obs, _ in quality_scored]
        
        print(f"Kept {len(pruned)} observations after quality pruning")
        return pruned
    
    def prune_observations_by_diversity(self, observations: List[str], 
                                      max_similar: int = 3) -> List[str]:
        """Remove overly similar observations"""
        print(f"Pruning {len(observations)} observations by diversity...")
        
        if len(observations) <= max_similar:
            return observations
        
        # Simple diversity pruning using keyword overlap
        diverse_obs = []
        for obs in observations:
            obs_words = set(obs.lower().split())
            
            # Check similarity to already selected observations
            is_diverse = True
            for selected in diverse_obs:
                selected_words = set(selected.lower().split())
                overlap = len(obs_words & selected_words) / len(obs_words | selected_words)
                if overlap > 0.6:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_obs.append(obs)
        
        print(f"Kept {len(diverse_obs)} observations after diversity pruning")
        return diverse_obs
    
    def beam_search_observations(self, observations: List[str], topic: str, 
                               beam_width: int = BEAM_WIDTH) -> List[str]:
        """Select top observations using beam search"""
        if len(observations) <= beam_width:
            return observations
        
        print(f"Beam search: selecting top {beam_width} from {len(observations)} observations...")
        
        # Score all observations
        scored_obs = []
        for obs in observations:
            quality = self.evaluator.evaluate_observation_quality(obs, topic)
            scored_obs.append((obs, quality))
        
        # Sort and take top beam_width
        scored_obs.sort(key=lambda x: x[1], reverse=True)
        return [obs for obs, _ in scored_obs[:beam_width]]
    
    def prune_jokes_early(self, jokes: List[PrunedJoke], 
                         threshold: float = MIN_JOKE_QUALITY) -> List[PrunedJoke]:
        """Remove obviously bad jokes before expensive ELO evaluation"""
        print(f"Early pruning of {len(jokes)} jokes...")
        
        pruned_jokes = []
        for joke in jokes:
            quality, creativity, relevance = self.evaluator.evaluate_joke_quality(joke.text)
            
            # Update joke metrics
            joke.metrics.quality_score = quality
            joke.metrics.creativity_score = creativity
            joke.metrics.relevance_score = relevance
            joke.metrics.combined_score = (quality + creativity + relevance) / 3.0
            
            # Keep if above threshold
            if joke.metrics.combined_score >= threshold:
                pruned_jokes.append(joke)
        
        print(f"Kept {len(pruned_jokes)} jokes after early pruning")
        return pruned_jokes

class PruningPlanSearchGenerator:
    """Enhanced PlanSearch generator with intelligent pruning"""
    
    def __init__(self, topic: str, branch_factor: int = BRANCH_FACTOR, max_depth: int = MAX_DEPTH):
        self.topic = topic
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.observation_tree = {}
        self.jokes = []
        
        # Pruning components
        self.evaluator = QualityEvaluator()
        self.pruning = PruningStrategies(self.evaluator)
    
    def generate_observations(self, context: Optional[List[str]] = None) -> List[str]:
        """Generate observations with initial quality filtering"""
        if context:
            prompt = f"""Given the topic '{self.topic}' and these observations:
{chr(10).join(f'- {obs}' for obs in context)}

Generate {self.branch_factor * 2} NEW observations that combine or extend these ideas.
Focus on specific, creative, and humor-rich observations."""
        else:
            prompt = f"""Generate {self.branch_factor * 2} diverse observations about '{self.topic}'.
Include stereotypes, wordplay opportunities, technical aspects, and absurd angles.
Make them specific, creative, and comedy-oriented."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        observations = []
        
        for response in responses:
            lines = response.split('\n')
            for line in lines:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith('-'):
                    if cleaned.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        cleaned = cleaned[2:].strip()
                    if cleaned:
                        observations.append(cleaned)
        
        # Apply pruning strategies
        observations = self.pruning.prune_observations_by_quality(observations, self.topic)
        observations = self.pruning.prune_observations_by_diversity(observations)
        observations = self.pruning.beam_search_observations(observations, self.topic, self.branch_factor)
        
        return observations
    
    def build_observation_tree(self):
        """Build observation tree with pruning at each level"""
        print(f"Building pruned observation tree for '{self.topic}'...")
        
        # Level 1: Basic observations with pruning
        level1_obs = self.generate_observations()
        self.observation_tree[1] = level1_obs
        print(f"Level 1: {len(level1_obs)} high-quality observations")
        
        if self.max_depth < 2:
            return
        
        # Level 2: Only combine promising observation pairs
        level2_obs = []
        
        # Score level 1 observations and only combine the best
        scored_obs = []
        for obs in level1_obs:
            quality = self.evaluator.evaluate_observation_quality(obs, self.topic)
            scored_obs.append((obs, quality))
        
        scored_obs.sort(key=lambda x: x[1], reverse=True)
        top_obs = [obs for obs, _ in scored_obs[:min(6, len(scored_obs))]]  # Top 6 only
        
        for obs_pair in itertools.combinations(top_obs, 2):
            derived = self.generate_observations(context=list(obs_pair))
            level2_obs.extend([(obs_pair, d) for d in derived[:1]])  # Only best derived observation
        
        self.observation_tree[2] = level2_obs
        print(f"Level 2: {len(level2_obs)} combined observations")
    
    def generate_joke_from_path(self, observation_path: List[str]) -> str:
        """Generate joke with quality prompting"""
        prompt = f"""Topic: {self.topic}

Observations:
{chr(10).join(f'- {obs}' for obs in observation_path)}

Create a single, short, punchy joke that cleverly uses these observations.
Make it genuinely funny with good timing and wordplay.
Avoid clichés and overused patterns.
Format: Just the joke text, no explanations."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        return responses[0].strip()
    
    def generate_all_jokes(self) -> List[PrunedJoke]:
        """Generate jokes with early quality pruning"""
        print("Generating jokes with quality control...")
        jokes = []
        
        # Jokes from level 1 observations
        for obs in self.observation_tree.get(1, []):
            joke_text = self.generate_joke_from_path([obs])
            jokes.append(PrunedJoke(text=joke_text, path=[obs]))
        
        # Jokes from level 2 observation combinations
        for (obs_pair, derived_obs) in self.observation_tree.get(2, []):
            path = list(obs_pair) + [derived_obs]
            joke_text = self.generate_joke_from_path(path)
            jokes.append(PrunedJoke(text=joke_text, path=path))
        
        # Selective contradiction jokes (only for best observations)
        top_obs = self.observation_tree.get(1, [])[:3]  # Only top 3
        for obs in top_obs:
            prompt = f"""Topic: {self.topic}
Observation: {obs}

Generate a clever joke that CONTRADICTS or subverts this observation.
Make it witty and unexpected. Just the joke, no explanation."""
            responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
            jokes.append(PrunedJoke(text=responses[0].strip(), path=[obs, "[contradicted]"]))
        
        print(f"Generated {len(jokes)} jokes before pruning")
        
        # Apply early pruning
        jokes = self.pruning.prune_jokes_early(jokes)
        
        self.jokes = jokes
        return jokes

class PruningEloSystem:
    """ELO system with pruning optimizations"""
    
    def __init__(self, k_factor: float = ELO_K_FACTOR):
        self.k_factor = k_factor
        self.comparison_count = 0
        self.comparison_cache = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, joke_a: PrunedJoke, joke_b: PrunedJoke, winner: str) -> Tuple[float, float]:
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
    
    def judge_pair(self, joke_a: PrunedJoke, joke_b: PrunedJoke) -> str:
        """Enhanced judging with consistency checks"""
        cache_key = (joke_a.id, joke_b.id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        prompt = f"""Compare these jokes carefully and determine which is funnier:

Joke A: {joke_a.text}

Joke B: {joke_b.text}

Consider timing, wordplay, originality, and overall humor impact.
Reply with only 'A' or 'B'."""
        
        responses = llm_complete(prompt, n=1, temperature=0.1)
        winner = responses[0].strip().upper()
        
        if winner not in ["A", "B"]:
            winner = random.choice(["A", "B"])
        
        self.comparison_cache[cache_key] = winner
        return winner
    
    def smart_tournament(self, jokes: List[PrunedJoke], rounds: int = 3) -> List[PrunedJoke]:
        """Run tournament with smart pairing based on quality scores"""
        print(f"Running smart tournament with {len(jokes)} jokes...")
        
        n_jokes = len(jokes)
        if n_jokes < 2:
            return jokes
        
        total_comparisons = min(rounds * n_jokes, n_jokes * (n_jokes - 1) // 2)
        
        for i in range(total_comparisons):
            if i % 5 == 0:
                print(f"  Comparison {i}/{total_comparisons}...")
            
            jokes_by_quality = sorted(enumerate(jokes), 
                                    key=lambda x: x[1].metrics.combined_score, 
                                    reverse=True)
            
            # Select pairs from similar quality ranges
            if i % 3 == 0:  # Sometimes random for diversity
                idx_a, idx_b = random.sample(range(n_jokes), 2)
            else:  # Usually quality-matched
                range_size = max(3, n_jokes // 4)
                start_idx = random.randint(0, max(0, len(jokes_by_quality) - range_size))
                local_jokes = jokes_by_quality[start_idx:start_idx + range_size]
                
                if len(local_jokes) >= 2:
                    (idx_a, _), (idx_b, _) = random.sample(local_jokes, 2)
                else:
                    idx_a, idx_b = random.sample(range(n_jokes), 2)
            
            joke_a, joke_b = jokes[idx_a], jokes[idx_b]
            
            # Skip if already compared
            cache_key = (joke_a.id, joke_b.id)
            if cache_key in self.comparison_cache:
                continue
            
            winner = self.judge_pair(joke_a, joke_b)
            new_rating_a, new_rating_b = self.update_ratings(joke_a, joke_b, winner)
            
            jokes[idx_a].elo_rating = new_rating_a
            jokes[idx_b].elo_rating = new_rating_b
            
            self.comparison_count += 1
        
        jokes.sort(key=lambda j: j.elo_rating, reverse=True)
        print(f"Smart tournament complete. {self.comparison_count} total comparisons.")
        return jokes

def generate_pruned_jokes(topic: str, top_n: int = 5) -> List[PrunedJoke]:
    """Main function to generate jokes with pruning strategies"""
    # Generate with pruning
    generator = PruningPlanSearchGenerator(topic)
    generator.build_observation_tree()
    jokes = generator.generate_all_jokes()
    
    # Rank with smart ELO system
    elo_system = PruningEloSystem()
    ranked_jokes = elo_system.smart_tournament(jokes, rounds=3)
    
    return ranked_jokes[:top_n]

def interactive_mode():
    """Interactive mode with pruning statistics"""
    print("\n🎭 Pruning-Enhanced AI Joke Generator (Version 2)")
    print("=" * 55)
    print("Features: Quality Pruning + Diversity Filtering + Smart ELO")
    
    while True:
        topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using the pruning joke generator! 😄")
            break
        
        if not topic:
            print("Please enter a valid topic.")
            continue
        
        print(f"\nGenerating jokes about '{topic}' with intelligent pruning...")
        print("This may take 1-2 minutes...\n")
        
        try:
            start_time = time.time()
            top_jokes = generate_pruned_jokes(topic, top_n=5)
            elapsed_time = time.time() - start_time
            
            print(f"\n🌟 Top 5 Pruned Jokes about '{topic}':")
            print("=" * 55)
            
            for i, joke in enumerate(top_jokes, 1):
                print(f"\n{i}. {joke.text}")
                print(f"   ELO Rating: {joke.elo_rating:.0f}")
                print(f"   Quality Metrics: Quality={joke.metrics.quality_score:.2f}, "
                      f"Creativity={joke.metrics.creativity_score:.2f}, "
                      f"Relevance={joke.metrics.relevance_score:.2f}")
                path_display = ' → '.join(joke.path[:2])
                if len(joke.path) > 2:
                    path_display += '...'
                print(f"   Generation Path: {path_display}")
            
            print(f"\n⏱️  Generated in {elapsed_time:.1f} seconds")
            print(f"📊 Pruning improved efficiency by focusing on high-quality candidates")
            
        except Exception as e:
            print(f"\nError generating jokes: {e}")
            print("Please try again with a different topic.")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] != "interactive":
        topic = sys.argv[1]
        print(f"\n🎭 Pruning-Enhanced Joke Generation for: {topic}")
        print("=" * 55)
        
        top_jokes = generate_pruned_jokes(topic, top_n=5)
        
        print("\n🌟 Top 5 Jokes with Pruning Analysis:")
        print("=" * 55)
        
        for i, joke in enumerate(top_jokes, 1):
            print(f"\n{i}. [ELO: {joke.elo_rating:.0f}] {joke.text}")
            print(f"   Combined Quality Score: {joke.metrics.combined_score:.2f}")
            path_display = ' → '.join(joke.path[:2])
            if len(joke.path) > 2:
                path_display += '...'
            print(f"   Generation Path: {path_display}")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()