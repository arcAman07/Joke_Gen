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
from abc import ABC, abstractmethod

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

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

class AnthropicProvider(LLMProvider):
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Please install anthropic: pip install anthropic")
            sys.exit(1)
    
    def complete(self, prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
        try:
            responses = []
            for _ in range(n):
                message = self.client.messages.create(
                    model="claude-3-sonnet-20241022",
                    max_tokens=500,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                responses.append(message.content[0].text.strip())
            return responses
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return ["Error generating response"]
    
    def get_embedding(self, text: str) -> List[float]:
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(48):
            byte_group = text_hash[i*4:(i+1)*4] if i*4 < len(text_hash) else text_hash[-4:]
            value = int.from_bytes(byte_group, 'big') / (2**32)
            embedding.append(value)
        return embedding

class GeminiProvider(LLMProvider):
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = input("Enter your Gemini API key (or press Enter to use free tier): ").strip()
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
        
        try:
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.genai = genai
        except ImportError:
            print("Please install google-generativeai: pip install google-generativeai")
            sys.exit(1)
    
    def complete(self, prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
        try:
            responses = []
            generation_config = self.genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=500,
            )
            
            for _ in range(n):
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                responses.append(response.text.strip())
            
            return responses
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ["Error generating response"]
    
    def get_embedding(self, text: str) -> List[float]:
        words = text.lower().split()
        embedding = []
        
        for i in range(384):
            if i < len(words):
                word_hash = hash(words[i])
                value = (word_hash % 1000) / 1000.0
            else:
                value = random.random()
            embedding.append(value)
        
        return embedding

class FallbackProvider(LLMProvider):
    def __init__(self):
        print("Warning: No API keys found. Using fallback mode with limited functionality.")
        self.templates = {
            "observations": [
                "People often {verb} when dealing with {topic}",
                "{topic} has an unexpected relationship with {random_object}",
                "The {adjective} nature of {topic} is often overlooked",
                "{topic} secretly {verb} at night",
                "Scientists discovered {topic} can {verb}"
            ],
            "jokes": [
                "Why did the {topic} {verb}? Because it wanted to {goal}!",
                "A {topic} walks into a bar. The bartender says, '{reaction}'",
                "What do you call a {topic} that {trait}? A {punchline}!",
                "How many {topic}s does it take to {task}? {number}, but {twist}",
                "I told my {topic} a joke about {subject}. It {response}."
            ]
        }
        self.verbs = ["dance", "sing", "complain", "celebrate", "hide", "jump", "whisper"]
        self.adjectives = ["mysterious", "hilarious", "quantum", "invisible", "explosive"]
        self.objects = ["banana", "unicorn", "smartphone", "time machine", "rubber duck"]
    
    def complete(self, prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
        responses = []
        
        if "observations" in prompt.lower():
            topic = self._extract_topic(prompt)
            for _ in range(n):
                obs = []
                for _ in range(5):
                    template = random.choice(self.templates["observations"])
                    obs.append(template.format(
                        topic=topic,
                        verb=random.choice(self.verbs),
                        adjective=random.choice(self.adjectives),
                        random_object=random.choice(self.objects)
                    ))
                responses.append("\n".join(obs))
        
        elif "joke" in prompt.lower():
            topic = self._extract_topic(prompt)
            for _ in range(n):
                template = random.choice(self.templates["jokes"])
                joke = template.format(
                    topic=topic,
                    verb=random.choice(self.verbs),
                    trait="can't stop " + random.choice(self.verbs) + "ing",
                    punchline=f"{random.choice(self.adjectives)} {topic}",
                    task=random.choice(["change a lightbulb", "write code", "make coffee"]),
                    number=random.randint(1, 10),
                    twist="they're all too busy " + random.choice(self.verbs) + "ing",
                    subject=random.choice(self.objects),
                    response="didn't get it",
                    reaction=f"Is this some kind of {random.choice(self.adjectives)} joke?",
                    goal=f"become more {random.choice(self.adjectives)}"
                )
                responses.append(joke)
        
        elif "which joke is funnier" in prompt.lower():
            responses = [random.choice(["A", "B"])]
        
        else:
            responses = [f"Generated response for: {prompt[:50]}..."]
        
        return responses[:n]
    
    def _extract_topic(self, prompt: str) -> str:
        if "topic '" in prompt:
            start = prompt.find("topic '") + 7
            end = prompt.find("'", start)
            return prompt[start:end] if end > start else "something"
        return "something"
    
    def get_embedding(self, text: str) -> List[float]:
        return [random.random() for _ in range(384)]

def get_llm_provider() -> LLMProvider:
    provider_name = os.environ.get("LLM_PROVIDER", "").upper()
    
    if provider_name == "ANTHROPIC" or os.environ.get("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude provider...")
        return AnthropicProvider()
    elif provider_name == "GEMINI" or os.environ.get("GEMINI_API_KEY"):
        print("Using Google Gemini provider...")
        return GeminiProvider()
    elif provider_name == "FALLBACK":
        return FallbackProvider()
    else:
        print("\nNo LLM provider specified. Choose one:")
        print("1. Anthropic Claude (requires API key)")
        print("2. Google Gemini (free tier available)")
        print("3. Fallback mode (no API required)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            api_key = input("Enter your Anthropic API key: ").strip()
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                return AnthropicProvider()
        elif choice == "2":
            return GeminiProvider()
        
        return FallbackProvider()

llm_provider = None

def llm_complete(prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
    global llm_provider
    if llm_provider is None:
        llm_provider = get_llm_provider()
    
    return llm_provider.complete(prompt, n, temperature)

def get_embedding(text: str) -> List[float]:
    global llm_provider
    if llm_provider is None:
        llm_provider = get_llm_provider()
    
    return llm_provider.get_embedding(text)

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
            "irony"
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
                taxonomy[dim] = random.uniform(0, 5)
        
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

class PlanSearchJokeGenerator:
    def __init__(self, topic: str, branch_factor: int = BRANCH_FACTOR, max_depth: int = MAX_DEPTH):
        self.topic = topic
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.observation_tree = {}
        self.jokes = []
    
    def generate_observations(self, context: Optional[List[str]] = None) -> List[str]:
        if context:
            prompt = f"""Given the topic '{self.topic}' and these observations:
{chr(10).join(f'- {obs}' for obs in context)}

Generate {self.branch_factor} NEW observations that combine or extend these ideas in unexpected ways.
Focus on contradictions, wordplay potential, and absurd connections."""
        else:
            prompt = f"""Generate {self.branch_factor} diverse observations about '{self.topic}'.
Include: stereotypes, wordplay opportunities, cultural references, technical aspects, and absurd angles.
Make them specific and comedy-oriented."""
        
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
        
        return observations[:self.branch_factor]
    
    def build_observation_tree(self):
        print(f"Building observation tree for '{self.topic}'...")
        level1_obs = self.generate_observations()
        self.observation_tree[1] = level1_obs
        
        if self.max_depth < 2:
            return
        
        level2_obs = []
        for obs_pair in itertools.combinations(level1_obs, 2):
            derived = self.generate_observations(context=list(obs_pair))
            level2_obs.extend([(obs_pair, d) for d in derived[:2]])
        
        self.observation_tree[2] = level2_obs
    
    def generate_joke_from_path(self, observation_path: List[str]) -> str:
        prompt = f"""Topic: {self.topic}

Observations:
{chr(10).join(f'- {obs}' for obs in observation_path)}

Create a single, short, punchy joke that cleverly uses these observations.
The joke should be self-contained and not require explanation.
Format: Just the joke text, no explanations or meta-commentary."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        return responses[0].strip()
    
    def generate_all_jokes(self) -> List[Joke]:
        print("Generating jokes from observation paths...")
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

class EloRatingSystem:
    def __init__(self, k_factor: float = ELO_K_FACTOR):
        self.k_factor = k_factor
        self.comparison_count = 0
        self.comparison_cache = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, joke_a: Joke, joke_b: Joke, winner: str) -> Tuple[float, float]:
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
    
    def judge_pair(self, joke_a: Joke, joke_b: Joke) -> str:
        cache_key = (joke_a.id, joke_b.id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        prompt = f"""Which joke is funnier?

Joke A: {joke_a.text}

Joke B: {joke_b.text}

Reply with only 'A' or 'B'."""
        
        responses = llm_complete(prompt, n=1, temperature=0.1)
        winner = responses[0].strip().upper()
        
        if winner not in ["A", "B"]:
            winner = random.choice(["A", "B"])
        
        self.comparison_cache[cache_key] = winner
        return winner
    
    def run_tournament(self, jokes: List[Joke], rounds: int = 3) -> List[Joke]:
        print(f"Running tournament with {len(jokes)} jokes...")
        
        n_jokes = len(jokes)
        if n_jokes < 2:
            return jokes
        
        total_comparisons = min(rounds * n_jokes, n_jokes * (n_jokes - 1) // 2)
        
        comparisons_made = 0
        for _ in range(total_comparisons):
            if comparisons_made % 10 == 0:
                print(f"  Comparison {comparisons_made}/{total_comparisons}...")
            
            idx_a, idx_b = random.sample(range(n_jokes), 2)
            joke_a, joke_b = jokes[idx_a], jokes[idx_b]
            
            if (joke_a.id, joke_b.id) in self.comparison_cache:
                continue
            
            winner = self.judge_pair(joke_a, joke_b)
            
            new_rating_a, new_rating_b = self.update_ratings(joke_a, joke_b, winner)
            jokes[idx_a].elo_rating = new_rating_a
            jokes[idx_b].elo_rating = new_rating_b
            
            comparisons_made += 1
            self.comparison_count += 1
        
        jokes.sort(key=lambda j: j.elo_rating, reverse=True)
        return jokes

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

def generate_top_jokes(topic: str, top_n: int = 5) -> List[Joke]:
    generator = PlanSearchJokeGenerator(topic)
    generator.build_observation_tree()
    
    jokes = generator.generate_all_jokes()
    print(f"\nGenerated {len(jokes)} jokes total")
    
    elo_system = EloRatingSystem()
    elo_ranked = elo_system.run_tournament(jokes, rounds=3)
    
    ranking_system = WeightedRankingSystem()
    final_ranked = ranking_system.rank_jokes(elo_ranked)
    
    diversity_analyzer = DiversityAnalyzer()
    overall_diversity = diversity_analyzer.calculate_diversity_score(jokes)
    print(f"Overall diversity score: {overall_diversity:.2f}")
    
    return final_ranked[:top_n]

def interactive_joke_generator():
    print("\n🎭 AI Joke Generator with Multiple LLM Support!")
    print("=" * 60)
    
    while True:
        topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using the joke generator! 😄")
            break
        
        if not topic:
            print("Please enter a valid topic.")
            continue
        
        print(f"\nGenerating jokes about '{topic}'...")
        print("This may take a minute...\n")
        
        try:
            start_time = time.time()
            top_jokes = generate_top_jokes(topic, top_n=5)
            elapsed_time = time.time() - start_time
            
            print(f"\n🌟 Top 5 Jokes about '{topic}':")
            print("=" * 60)
            
            for i, joke in enumerate(top_jokes, 1):
                print(f"\n{i}. {joke.text}")
                print(f"   📊 Scores: Humor={joke.metrics.humor_score:.2f}, "
                      f"Novelty={joke.metrics.novelty_score:.2f}, "
                      f"Diversity={joke.metrics.diversity_contribution:.2f}")
                print(f"   Pattern: {joke.metrics.structural_pattern}")
            
            print(f"\n⏱️  Generated in {elapsed_time:.1f} seconds")
            
        except Exception as e:
            print(f"\nError generating jokes: {e}")
            print("Please try again with a different topic.")

def main():
    if len(sys.argv) > 1:
        topic = sys.argv[1]
        print(f"\n🎭 Generating jokes about: {topic}")
        print("=" * 60)
        
        top_jokes = generate_top_jokes(topic, top_n=5)
        
        print("\n🌟 Top 5 Jokes:")
        print("=" * 60)
        
        for i, joke in enumerate(top_jokes, 1):
            print(f"\n{i}. {joke.text}")
            print(f"   Overall Score: {joke.metrics.weighted_score:.3f}")
    else:
        interactive_joke_generator()

if __name__ == "__main__":
    main()