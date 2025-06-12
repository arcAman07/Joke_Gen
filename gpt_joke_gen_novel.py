#!/usr/bin/env python3

import os
import sys
import json
import random
import itertools
import hashlib
import time
import re
import math
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime

try:
    import openai
    import numpy as np
except ImportError:
    print("Please install requirements: pip install openai numpy")
    sys.exit(1)

TEMPERATURE = 0.9
BRANCH_FACTOR = 5
MAX_DEPTH = 2
ELO_K_FACTOR = 32
INITIAL_ELO = 1200
DIVERSITY_WEIGHT = 0.3
HUMOR_WEIGHT = 0.7
NOVELTY_THRESHOLD = 0.75

@dataclass
class JokeMetrics:
    humor_score: float = 0.0
    diversity_contribution: float = 0.0
    novelty_score: float = 0.0
    taxonomy: Dict[str, float] = field(default_factory=dict)
    semantic_fingerprint: List[float] = field(default_factory=list)
    structural_pattern: str = ""
    weighted_score: float = 0.0

@dataclass
class Joke:
    text: str
    path: List[str]
    elo_rating: float = INITIAL_ELO
    id: str = ""
    metrics: JokeMetrics = field(default_factory=JokeMetrics)
    generation_time: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:8]

class OpenAIClient:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
    
    def complete(self, prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=n,
                max_tokens=500
            )
            
            return [choice.message.content.strip() for choice in response.choices]
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ["Error generating response"]
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except:
            return [random.random() for _ in range(384)]

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

class NoveltyChecker:
    def __init__(self):
        self.common_patterns = {
            "cross_road": r"why did .* cross the road",
            "walks_into_bar": r"walks into a bar",
            "lightbulb": r"how many .* to change a lightbulb",
            "knock_knock": r"knock knock",
            "difference_between": r"what's the difference between",
            "call_a": r"what do you call a",
            "says_to": r"(doctor|lawyer|priest) says to",
        }
        self.pattern_penalties = {
            "cross_road": 0.7,
            "walks_into_bar": 0.7,
            "lightbulb": 0.6,
            "knock_knock": 0.8,
            "difference_between": 0.5,
            "call_a": 0.5,
            "says_to": 0.6,
        }
    
    def check_common_patterns(self, joke_text: str) -> Tuple[float, str]:
        joke_lower = joke_text.lower()
        
        for pattern_name, pattern in self.common_patterns.items():
            if re.search(pattern, joke_lower):
                penalty = self.pattern_penalties[pattern_name]
                return penalty, pattern_name
        
        return 1.0, "original"
    
    def calculate_novelty_score(self, joke: Joke, all_jokes: List[Joke]) -> float:
        pattern_score, pattern_type = self.check_common_patterns(joke.text)
        
        embedding = get_embedding(joke.text)
        joke.metrics.semantic_fingerprint = embedding
        
        semantic_novelty = 1.0
        if len(all_jokes) > 1:
            similarities = []
            for other in all_jokes:
                if other.id != joke.id and other.metrics.semantic_fingerprint:
                    sim = self.cosine_similarity(
                        embedding, 
                        other.metrics.semantic_fingerprint
                    )
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                semantic_novelty = 1.0 - avg_similarity
        
        final_novelty = pattern_score * semantic_novelty
        joke.metrics.novelty_score = final_novelty
        joke.metrics.structural_pattern = pattern_type
        
        return final_novelty
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class DiversityAnalyzer:
    def __init__(self):
        self.humor_dimensions = [
            "wordplay",
            "absurdity",
            "observational",
            "dark",
            "intellectual",
            "slapstick",
            "meta",
            "irony",
            "callback",
            "misdirection"
        ]
    
    def analyze_joke_taxonomy(self, joke: Joke) -> Dict[str, float]:
        prompt = f"""Analyze this joke across humor dimensions.
Joke: {joke.text}

Rate each dimension from 0-10:
- wordplay: puns, double meanings
- absurdity: surreal, nonsensical
- observational: everyday life humor
- dark: morbid or taboo
- intellectual: requires knowledge
- slapstick: physical comedy
- meta: self-referential
- irony: opposite of expected
- callback: references earlier content
- misdirection: unexpected punchline

Format: dimension:score (one per line)"""
        
        response = llm_complete(prompt, temperature=0.1)[0]
        
        taxonomy = {}
        for line in response.split('\n'):
            if ':' in line:
                try:
                    dim, score = line.split(':')
                    dim = dim.strip().lower()
                    score = float(score.strip())
                    if dim in self.humor_dimensions:
                        taxonomy[dim] = min(10, max(0, score))
                except:
                    continue
        
        for dim in self.humor_dimensions:
            if dim not in taxonomy:
                taxonomy[dim] = 0.0
        
        joke.metrics.taxonomy = taxonomy
        return taxonomy
    
    def calculate_diversity_score(self, jokes: List[Joke]) -> float:
        if len(jokes) < 2:
            return 0.0
        
        dimension_variance = defaultdict(list)
        
        for joke in jokes:
            if not joke.metrics.taxonomy:
                self.analyze_joke_taxonomy(joke)
            
            for dim, score in joke.metrics.taxonomy.items():
                dimension_variance[dim].append(score)
        
        total_variance = 0.0
        for dim, scores in dimension_variance.items():
            if len(scores) > 1:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                total_variance += variance
        
        avg_variance = total_variance / len(self.humor_dimensions)
        diversity_score = 1 - math.exp(-avg_variance / 10)
        
        return diversity_score
    
    def calculate_joke_diversity_contribution(self, joke: Joke, other_jokes: List[Joke]) -> float:
        if not other_jokes:
            return 1.0
        
        if not joke.metrics.taxonomy:
            self.analyze_joke_taxonomy(joke)
        
        dimension_counts = defaultdict(int)
        for other in other_jokes:
            if other.id != joke.id and other.metrics.taxonomy:
                dominant_dim = max(other.metrics.taxonomy.items(), key=lambda x: x[1])[0]
                dimension_counts[dominant_dim] += 1
        
        joke_dominant = max(joke.metrics.taxonomy.items(), key=lambda x: x[1])[0]
        
        rarity_score = 1.0 - (dimension_counts[joke_dominant] / len(other_jokes))
        
        joke.metrics.diversity_contribution = rarity_score
        return rarity_score

class AdaptivePlanSearch:
    def __init__(self, topic: str, branch_factor: int = BRANCH_FACTOR, max_depth: int = MAX_DEPTH):
        self.topic = topic
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.observation_tree = {}
        self.jokes = []
        self.observation_quality = {}
        self.temperature_adapter = TemperatureAdapter()
    
    def generate_observations(self, context: Optional[List[str]] = None) -> List[str]:
        temperature = self.temperature_adapter.get_adaptive_temperature(len(self.jokes))
        
        if context:
            prompt = f"""Given the topic '{self.topic}' and these observations:
{chr(10).join(f'- {obs}' for obs in context)}

Generate {self.branch_factor} NEW observations that combine or extend these ideas in unexpected ways.
Focus on contradictions, wordplay potential, and absurd connections."""
        else:
            prompt = f"""Generate {self.branch_factor} diverse observations about '{self.topic}'.
Include: stereotypes, wordplay opportunities, cultural references, technical aspects, and absurd angles.
Make them specific and comedy-oriented."""
        
        responses = llm_complete(prompt, n=1, temperature=temperature)
        observations = []
        
        for response in responses:
            lines = response.split('\n')
            for line in lines:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith('-'):
                    if cleaned.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        cleaned = cleaned[2:].strip()
                    observations.append(cleaned)
        
        return observations[:self.branch_factor]
    
    def build_observation_tree(self):
        print(f"Building adaptive observation tree for '{self.topic}'...")
        level1_obs = self.generate_observations()
        self.observation_tree[1] = level1_obs
        
        for obs in level1_obs:
            self.observation_quality[obs] = 0.5
        
        if self.max_depth < 2:
            return
        
        level2_obs = []
        promising_pairs = []
        
        for obs_pair in itertools.combinations(level1_obs, 2):
            quality_score = (self.observation_quality[obs_pair[0]] + 
                           self.observation_quality[obs_pair[1]]) / 2
            promising_pairs.append((quality_score, obs_pair))
        
        promising_pairs.sort(reverse=True, key=lambda x: x[0])
        
        for _, obs_pair in promising_pairs[:len(promising_pairs)//2]:
            derived = self.generate_observations(context=list(obs_pair))
            level2_obs.extend([(obs_pair, d) for d in derived[:2]])
        
        self.observation_tree[2] = level2_obs
    
    def generate_joke_from_path(self, observation_path: List[str]) -> str:
        temperature = self.temperature_adapter.get_adaptive_temperature(len(self.jokes))
        
        prompt = f"""Topic: {self.topic}

Observations:
{chr(10).join(f'- {obs}' for obs in observation_path)}

Create a single, short, punchy joke that cleverly uses these observations.
The joke should be self-contained and not require explanation.
Format: Just the joke text, no explanations or meta-commentary."""
        
        responses = llm_complete(prompt, n=1, temperature=temperature)
        return responses[0].strip()
    
    def backpropagate_quality(self, joke: Joke, quality_score: float):
        decay_factor = 0.9
        for i, obs in enumerate(joke.path):
            if obs in self.observation_quality:
                old_quality = self.observation_quality[obs]
                self.observation_quality[obs] = (
                    0.7 * old_quality + 
                    0.3 * quality_score * (decay_factor ** i)
                )
    
    def generate_all_jokes(self) -> List[Joke]:
        print("Generating jokes with adaptive feedback...")
        jokes = []
        start_time = time.time()
        
        for obs in self.observation_tree.get(1, []):
            joke_text = self.generate_joke_from_path([obs])
            joke = Joke(
                text=joke_text, 
                path=[obs],
                generation_time=time.time() - start_time
            )
            jokes.append(joke)
        
        for (obs_pair, derived_obs) in self.observation_tree.get(2, []):
            path = list(obs_pair) + [derived_obs]
            joke_text = self.generate_joke_from_path(path)
            joke = Joke(
                text=joke_text, 
                path=path,
                generation_time=time.time() - start_time
            )
            jokes.append(joke)
        
        for obs in self.observation_tree.get(1, []):
            prompt = f"""Topic: {self.topic}
Observation: {obs}

Now generate a joke that CONTRADICTS or subverts this observation.
Just the joke, no explanation."""
            responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
            joke = Joke(
                text=responses[0].strip(), 
                path=[obs, "[contradicted]"],
                generation_time=time.time() - start_time
            )
            jokes.append(joke)
        
        self.jokes = jokes
        return jokes

class TemperatureAdapter:
    def __init__(self):
        self.min_temp = 0.7
        self.max_temp = 1.2
        self.diversity_history = []
    
    def get_adaptive_temperature(self, jokes_generated: int) -> float:
        if jokes_generated < 5:
            return 0.9
        
        if self.diversity_history:
            recent_diversity = self.diversity_history[-1]
            if recent_diversity < 0.3:
                return min(self.max_temp, 0.9 + 0.2)
            elif recent_diversity > 0.7:
                return max(self.min_temp, 0.9 - 0.1)
        
        return 0.9
    
    def update_diversity(self, diversity_score: float):
        self.diversity_history.append(diversity_score)
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)

class WeightedRankingSystem:
    def __init__(self, humor_weight: float = HUMOR_WEIGHT, 
                 diversity_weight: float = DIVERSITY_WEIGHT):
        self.humor_weight = humor_weight
        self.diversity_weight = diversity_weight
        self.novelty_checker = NoveltyChecker()
        self.diversity_analyzer = DiversityAnalyzer()
    
    def calculate_weighted_score(self, joke: Joke, all_jokes: List[Joke]) -> float:
        humor_score = joke.elo_rating / 1500.0
        
        diversity_contribution = self.diversity_analyzer.calculate_joke_diversity_contribution(
            joke, all_jokes
        )
        
        novelty_score = self.novelty_checker.calculate_novelty_score(joke, all_jokes)
        
        weighted_score = (
            self.humor_weight * humor_score * 0.6 +
            self.diversity_weight * diversity_contribution * 0.3 +
            0.1 * novelty_score
        )
        
        joke.metrics.humor_score = humor_score
        joke.metrics.diversity_contribution = diversity_contribution
        joke.metrics.novelty_score = novelty_score
        joke.metrics.weighted_score = weighted_score
        
        return weighted_score
    
    def rank_jokes(self, jokes: List[Joke]) -> List[Joke]:
        for joke in jokes:
            self.calculate_weighted_score(joke, jokes)
        
        jokes.sort(key=lambda j: j.metrics.weighted_score, reverse=True)
        return jokes

class EnhancedEloSystem:
    def __init__(self, k_factor: float = ELO_K_FACTOR):
        self.k_factor = k_factor
        self.comparison_count = 0
        self.comparison_cache = {}
        self.judge_explanations = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, joke_a: Joke, joke_b: Joke, winner: str, confidence: float = 1.0) -> Tuple[float, float]:
        expected_a = self.expected_score(joke_a.elo_rating, joke_b.elo_rating)
        expected_b = 1 - expected_a
        
        if winner == "A":
            score_a, score_b = confidence, 1 - confidence
        elif winner == "B":
            score_a, score_b = 1 - confidence, confidence
        else:
            score_a, score_b = 0.5, 0.5
        
        new_rating_a = joke_a.elo_rating + self.k_factor * (score_a - expected_a)
        new_rating_b = joke_b.elo_rating + self.k_factor * (score_b - expected_b)
        
        return new_rating_a, new_rating_b
    
    def judge_with_explanation(self, joke_a: Joke, joke_b: Joke) -> Tuple[str, str, float]:
        cache_key = (joke_a.id, joke_b.id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        prompt = f"""Compare these two jokes and determine which is funnier.

Joke A: {joke_a.text}

Joke B: {joke_b.text}

Provide:
1. Which joke is funnier: A or B
2. Brief explanation why (one sentence)
3. Confidence level (0-1)

Format:
Winner: [A/B]
Reason: [explanation]
Confidence: [0.0-1.0]"""
        
        response = llm_complete(prompt, n=1, temperature=0.1)[0]
        
        winner = "A"
        reason = "No explanation provided"
        confidence = 1.0
        
        for line in response.split('\n'):
            if line.startswith('Winner:'):
                winner = line.split(':')[1].strip().upper()
            elif line.startswith('Reason:'):
                reason = line.split(':', 1)[1].strip()
            elif line.startswith('Confidence:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                except:
                    confidence = 1.0
        
        if winner not in ["A", "B"]:
            winner = random.choice(["A", "B"])
        
        result = (winner, reason, confidence)
        self.comparison_cache[cache_key] = result
        self.judge_explanations[cache_key] = reason
        
        return result
    
    def run_enhanced_tournament(self, jokes: List[Joke], rounds: int = 3) -> List[Joke]:
        print(f"Running enhanced tournament with {len(jokes)} jokes...")
        
        n_jokes = len(jokes)
        if n_jokes < 2:
            return jokes
        
        total_comparisons = rounds * n_jokes
        
        for i in range(total_comparisons):
            if i % 10 == 0:
                print(f"  Comparison {i}/{total_comparisons}...")
            
            idx_a, idx_b = random.sample(range(n_jokes), 2)
            joke_a, joke_b = jokes[idx_a], jokes[idx_b]
            
            winner, reason, confidence = self.judge_with_explanation(joke_a, joke_b)
            
            new_rating_a, new_rating_b = self.update_ratings(
                joke_a, joke_b, winner, confidence
            )
            jokes[idx_a].elo_rating = new_rating_a
            jokes[idx_b].elo_rating = new_rating_b
            
            self.comparison_count += 1
        
        jokes.sort(key=lambda j: j.elo_rating, reverse=True)
        return jokes

class JokeGenerationPipeline:
    def __init__(self, topic: str):
        self.topic = topic
        self.generator = AdaptivePlanSearch(topic)
        self.elo_system = EnhancedEloSystem()
        self.ranking_system = WeightedRankingSystem()
        self.diversity_analyzer = DiversityAnalyzer()
        self.novelty_checker = NoveltyChecker()
    
    def generate_and_rank_jokes(self, top_n: int = 5) -> Tuple[List[Joke], Dict[str, any]]:
        start_time = time.time()
        
        self.generator.build_observation_tree()
        
        jokes = self.generator.generate_all_jokes()
        print(f"\nGenerated {len(jokes)} jokes")
        
        overall_diversity = self.diversity_analyzer.calculate_diversity_score(jokes)
        print(f"Overall diversity score: {overall_diversity:.2f}")
        
        self.generator.temperature_adapter.update_diversity(overall_diversity)
        
        elo_ranked = self.elo_system.run_enhanced_tournament(jokes, rounds=3)
        
        final_ranked = self.ranking_system.rank_jokes(elo_ranked)
        
        for joke in final_ranked:
            self.generator.backpropagate_quality(
                joke, 
                joke.metrics.weighted_score
            )
        
        stats = {
            "total_jokes": len(jokes),
            "generation_time": time.time() - start_time,
            "diversity_score": overall_diversity,
            "unique_patterns": len(set(j.metrics.structural_pattern for j in jokes)),
            "avg_novelty": sum(j.metrics.novelty_score for j in jokes) / len(jokes),
            "comparison_count": self.elo_system.comparison_count
        }
        
        return final_ranked[:top_n], stats

def generate_enhanced_jokes(topic: str, top_n: int = 5) -> List[Joke]:
    pipeline = JokeGenerationPipeline(topic)
    top_jokes, stats = pipeline.generate_and_rank_jokes(top_n)
    
    print(f"\n📊 Generation Statistics:")
    print(f"  Total jokes: {stats['total_jokes']}")
    print(f"  Time: {stats['generation_time']:.1f}s")
    print(f"  Diversity: {stats['diversity_score']:.2f}")
    print(f"  Avg novelty: {stats['avg_novelty']:.2f}")
    print(f"  Unique patterns: {stats['unique_patterns']}")
    
    return top_jokes

def interactive_joke_generator():
    print("\n🎭 Enhanced AI Joke Generator with Novelty Detection!")
    print("=" * 60)
    
    while True:
        topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using the enhanced joke generator! 😄")
            break
        
        if not topic:
            print("Please enter a valid topic.")
            continue
        
        print(f"\nGenerating enhanced jokes about '{topic}'...")
        print("This may take 1-2 minutes...\n")
        
        try:
            start_time = time.time()
            top_jokes = generate_enhanced_jokes(topic, top_n=5)
            elapsed_time = time.time() - start_time
            
            print(f"\n🌟 Top 5 Jokes about '{topic}':")
            print("=" * 60)
            
            for i, joke in enumerate(top_jokes, 1):
                print(f"\n{i}. {joke.text}")
                print(f"   📊 Metrics:")
                print(f"      Humor: {joke.metrics.humor_score:.2f}")
                print(f"      Novelty: {joke.metrics.novelty_score:.2f}")
                print(f"      Diversity: {joke.metrics.diversity_contribution:.2f}")
                print(f"      Overall: {joke.metrics.weighted_score:.2f}")
                print(f"      Pattern: {joke.metrics.structural_pattern}")
                
                if joke.metrics.taxonomy:
                    top_dims = sorted(joke.metrics.taxonomy.items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                    print(f"      Style: {', '.join(f'{d[0]}({d[1]:.1f})' for d in top_dims)}")
            
            print(f"\n⏱️  Generated in {elapsed_time:.1f} seconds")
            
        except Exception as e:
            print(f"\nError generating jokes: {e}")
            print("Please try again with a different topic.")

def main(topic: str):
    print(f"\n🎭 Enhanced Joke Generation for: {topic}")
    print("=" * 60)
    
    top_jokes = generate_enhanced_jokes(topic, top_n=5)
    
    print("\n🌟 Top 5 Jokes with Full Analysis:")
    print("=" * 60)
    
    for i, joke in enumerate(top_jokes, 1):
        print(f"\n{i}. {joke.text}")
        print(f"   Weighted Score: {joke.metrics.weighted_score:.3f}")
        print(f"   Components: Humor={joke.metrics.humor_score:.2f}, "
              f"Novelty={joke.metrics.novelty_score:.2f}, "
              f"Diversity={joke.metrics.diversity_contribution:.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "interactive":
        topic = sys.argv[1]
        main(topic)
    else:
        interactive_joke_generator()