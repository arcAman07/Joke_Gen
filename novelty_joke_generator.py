#!/usr/bin/env python3
"""
Joke Generator with Advanced Novelty Detection - Version 4
Comprehensive novelty detection to distinguish creativity from memorization
"""

import os
import sys
import random
import itertools
import hashlib
import time
import re
import math
import json
from typing import List, Tuple, Optional, Dict, Set
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
HUMOR_WEIGHT = 0.55
DIVERSITY_WEIGHT = 0.25
NOVELTY_WEIGHT = 0.20  # Increased importance

# Novelty thresholds
SIMILARITY_THRESHOLD = 0.75
PATTERN_PENALTY_THRESHOLD = 0.8
MIN_EDIT_DISTANCE = 20

@dataclass
class NoveltyMetrics:
    """Comprehensive novelty analysis"""
    pattern_score: float = 1.0
    semantic_novelty: float = 1.0
    structural_novelty: float = 1.0
    edit_distance_novelty: float = 1.0
    conceptual_novelty: float = 1.0
    combined_novelty: float = 1.0
    detected_patterns: List[str] = field(default_factory=list)
    similar_jokes: List[str] = field(default_factory=list)
    memorization_risk: float = 0.0

@dataclass
class AdvancedMetrics:
    humor_score: float = 0.0
    diversity_contribution: float = 0.0
    novelty_metrics: NoveltyMetrics = field(default_factory=NoveltyMetrics)
    taxonomy: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    quality_score: float = 0.0
    creativity_score: float = 0.0
    relevance_score: float = 0.0

@dataclass
class NoveltyJoke:
    text: str
    path: List[str]
    elo_rating: float = INITIAL_ELO
    id: str = ""
    metrics: AdvancedMetrics = field(default_factory=AdvancedMetrics)
    
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
                max_tokens=400
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
            # Fallback hash-based embedding
            return self._hash_embedding(text)
    
    def _hash_embedding(self, text: str) -> List[float]:
        """Fallback embedding using hash function"""
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(384):  # Match embedding dimension
            byte_idx = i % len(text_hash)
            value = text_hash[byte_idx] / 255.0
            embedding.append(value)
        return embedding

# Global client
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

class AdvancedNoveltyDetector:
    """Comprehensive novelty detection system"""
    
    def __init__(self):
        # Extended pattern database
        self.common_patterns = {
            # Classic patterns
            "cross_road": r"why did .* cross the road",
            "walks_into_bar": r"walks into a bar",
            "lightbulb": r"how many .* to change a lightbulb",
            "knock_knock": r"knock knock",
            "difference_between": r"what's the difference between",
            "call_a": r"what do you call a",
            
            # Dialog patterns
            "says_to": r"(doctor|lawyer|priest|teacher) says to",
            "patient_says": r"patient says",
            "wife_says": r"(wife|husband) says",
            
            # Setup-punchline patterns
            "one_day": r"one day",
            "so_bad": r"so bad that",
            "three_guys": r"three (guys|men|people)",
            "walked_into": r"walked into",
            
            # Internet-era patterns
            "yo_mama": r"yo mama",
            "chuck_norris": r"chuck norris",
            "your_mom": r"your mom",
            "that_moment": r"that moment when",
            
            # Programming humor patterns
            "infinite_loop": r"infinite loop",
            "null_pointer": r"null pointer",
            "stack_overflow": r"stack overflow",
            "compiler_error": r"compiler error",
        }
        
        self.pattern_penalties = {
            "cross_road": 0.3,     # Heavy penalty for overused patterns
            "walks_into_bar": 0.4,
            "lightbulb": 0.5,
            "knock_knock": 0.6,
            "difference_between": 0.4,
            "call_a": 0.5,
            "says_to": 0.6,
            "patient_says": 0.7,
            "wife_says": 0.7,
            "one_day": 0.8,
            "so_bad": 0.8,
            "three_guys": 0.6,
            "walked_into": 0.7,
            "yo_mama": 0.2,        # Very heavy penalty
            "chuck_norris": 0.2,
            "your_mom": 0.3,
            "that_moment": 0.8,
            "infinite_loop": 0.7,
            "null_pointer": 0.8,
            "stack_overflow": 0.8,
            "compiler_error": 0.8,
        }
        
        # Cache for expensive operations
        self.embedding_cache = {}
        self.pattern_cache = {}
        
    def detect_common_patterns(self, joke_text: str) -> Tuple[float, List[str]]:
        """Detect multiple common patterns and return combined penalty"""
        cache_key = joke_text.lower()
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        joke_lower = joke_text.lower()
        detected_patterns = []
        penalties = []
        
        for pattern_name, pattern in self.common_patterns.items():
            if re.search(pattern, joke_lower):
                detected_patterns.append(pattern_name)
                penalties.append(self.pattern_penalties[pattern_name])
        
        if not penalties:
            combined_penalty = 1.0
        else:
            # Use minimum penalty (most restrictive)
            combined_penalty = min(penalties)
        
        result = (combined_penalty, detected_patterns)
        self.pattern_cache[cache_key] = result
        return result
    
    def calculate_edit_distance(self, text1: str, text2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(text1) < len(text2):
            text1, text2 = text2, text1
        
        if len(text2) == 0:
            return len(text1)
        
        previous_row = list(range(len(text2) + 1))
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def analyze_structural_novelty(self, joke: NoveltyJoke) -> float:
        """Analyze structural patterns beyond regex matching"""
        text = joke.text.lower()
        
        # Count structural elements
        question_words = len(re.findall(r'\b(what|who|where|when|why|how)\b', text))
        setup_indicators = len(re.findall(r'\b(so|then|but|however|meanwhile)\b', text))
        punchline_indicators = len(re.findall(r'\b(because|since|turns out|actually)\b', text))
        
        # Analyze sentence structure
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Structural novelty based on complexity and variation
        if sentence_count == 1 and question_words == 0:
            structural_score = 0.6  # Simple one-liner
        elif question_words > 0 and punchline_indicators > 0:
            structural_score = 0.7  # Q&A format
        elif setup_indicators > 0 and sentence_count > 1:
            structural_score = 0.8  # Multi-part setup
        else:
            structural_score = 0.9  # Novel structure
        
        return structural_score
    
    def analyze_conceptual_novelty(self, joke: NoveltyJoke, topic: str) -> float:
        """Analyze conceptual creativity using LLM"""
        prompt = f"""Analyze the conceptual novelty of this joke about '{topic}':

Joke: {joke.text}

Rate the conceptual creativity from 0-10 considering:
- Unexpected connections or associations
- Original perspective on the topic
- Creative use of concepts or ideas
- Departure from obvious or predictable humor

Respond with just a number 0-10."""
        
        try:
            response = llm_complete(prompt, temperature=0.1)[0]
            score = float(response.strip()) / 10.0
            return max(0.0, min(1.0, score))
        except:
            return 0.7  # Default moderate novelty
    
    def detect_memorization_risk(self, joke: NoveltyJoke, all_jokes: List[NoveltyJoke]) -> float:
        """Assess risk that joke is memorized rather than generated"""
        text = joke.text
        
        # Check for exact or near-exact matches
        for other in all_jokes:
            if other.id != joke.id:
                edit_dist = self.calculate_edit_distance(text, other.text)
                if edit_dist < MIN_EDIT_DISTANCE:
                    return 0.9  # High memorization risk
        
        # Check for overly generic patterns
        generic_indicators = [
            r'\b(funny|hilarious|joke|humor)\b',
            r'\b(haha|lol|rofl)\b',
            r'\b(classic|old|ancient)\b'
        ]
        
        generic_count = sum(1 for pattern in generic_indicators 
                          if re.search(pattern, text.lower()))
        
        memorization_risk = min(0.8, generic_count * 0.3)
        return memorization_risk
    
    def comprehensive_novelty_analysis(self, joke: NoveltyJoke, all_jokes: List[NoveltyJoke], topic: str) -> NoveltyMetrics:
        """Perform comprehensive novelty analysis"""
        metrics = NoveltyMetrics()
        
        # 1. Pattern-based analysis
        pattern_score, detected_patterns = self.detect_common_patterns(joke.text)
        metrics.pattern_score = pattern_score
        metrics.detected_patterns = detected_patterns
        
        # 2. Semantic similarity analysis
        embedding = get_embedding(joke.text)
        similarities = []
        similar_jokes = []
        
        for other in all_jokes:
            if other.id != joke.id:
                other_embedding = get_embedding(other.text)
                sim = self.cosine_similarity(embedding, other_embedding)
                similarities.append(sim)
                
                if sim > SIMILARITY_THRESHOLD:
                    similar_jokes.append(other.text)
        
        if similarities:
            max_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            semantic_novelty = 1.0 - (0.7 * avg_sim + 0.3 * max_sim)
        else:
            semantic_novelty = 1.0
        
        metrics.semantic_novelty = max(0.0, semantic_novelty)
        metrics.similar_jokes = similar_jokes
        
        # 3. Structural novelty
        metrics.structural_novelty = self.analyze_structural_novelty(joke)
        
        # 4. Edit distance novelty
        min_edit_dist = float('inf')
        for other in all_jokes:
            if other.id != joke.id:
                dist = self.calculate_edit_distance(joke.text, other.text)
                min_edit_dist = min(min_edit_dist, dist)
        
        if min_edit_dist == float('inf'):
            edit_novelty = 1.0
        else:
            edit_novelty = min(1.0, min_edit_dist / 50.0)  # Normalize
        
        metrics.edit_distance_novelty = edit_novelty
        
        # 5. Conceptual novelty
        metrics.conceptual_novelty = self.analyze_conceptual_novelty(joke, topic)
        
        # 6. Memorization risk
        metrics.memorization_risk = self.detect_memorization_risk(joke, all_jokes)
        
        # 7. Combined novelty score
        combined = (
            0.25 * metrics.pattern_score +
            0.25 * metrics.semantic_novelty +
            0.20 * metrics.structural_novelty +
            0.15 * metrics.edit_distance_novelty +
            0.15 * metrics.conceptual_novelty
        )
        
        # Apply memorization penalty
        combined *= (1.0 - metrics.memorization_risk)
        
        metrics.combined_novelty = max(0.0, combined)
        
        return metrics

class DiversityAnalyzer:
    """Enhanced diversity analysis"""
    
    def __init__(self):
        self.humor_dimensions = [
            "wordplay", "absurdity", "observational", "dark", "intellectual",
            "slapstick", "meta", "irony", "callback", "misdirection",
            "self_deprecating", "surreal", "topical", "physical"
        ]
    
    def analyze_joke_taxonomy(self, joke: NoveltyJoke) -> Dict[str, float]:
        """Analyze joke across humor dimensions"""
        prompt = f"""Analyze this joke across humor dimensions (0-10 each):

Joke: {joke.text}

Rate each dimension:
- wordplay: puns, language play
- absurdity: surreal, illogical
- observational: everyday situations
- dark: morbid, edgy content
- intellectual: requires knowledge
- slapstick: physical comedy
- meta: self-referential
- irony: opposite meaning
- callback: references earlier content
- misdirection: unexpected twist
- self_deprecating: self-mockery
- surreal: dreamlike, bizarre
- topical: current events
- physical: body-based humor

Format: dimension:score (e.g., wordplay:7)"""
        
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
        
        # Fill missing dimensions
        for dim in self.humor_dimensions:
            if dim not in taxonomy:
                taxonomy[dim] = 0.0
        
        joke.metrics.taxonomy = taxonomy
        return taxonomy
    
    def calculate_joke_diversity_contribution(self, joke: NoveltyJoke, other_jokes: List[NoveltyJoke]) -> float:
        """Calculate diversity contribution with enhanced analysis"""
        if not other_jokes:
            return 1.0
        
        if not joke.metrics.taxonomy:
            self.analyze_joke_taxonomy(joke)
        
        # Analyze distribution across all dimensions
        dimension_usage = defaultdict(float)
        total_weight = 0
        
        for other in other_jokes:
            if other.id != joke.id and other.metrics.taxonomy:
                for dim, score in other.metrics.taxonomy.items():
                    dimension_usage[dim] += score
                    total_weight += score
        
        # Calculate rarity across all dimensions
        rarity_scores = []
        for dim, score in joke.metrics.taxonomy.items():
            if total_weight > 0:
                dim_usage_ratio = dimension_usage[dim] / total_weight
                # Higher score in less-used dimensions = higher diversity
                rarity = (1.0 - dim_usage_ratio) * (score / 10.0)
                rarity_scores.append(rarity)
        
        diversity_contribution = sum(rarity_scores) / len(rarity_scores) if rarity_scores else 0.5
        joke.metrics.diversity_contribution = diversity_contribution
        
        return diversity_contribution

class NoveltyRankingSystem:
    """Advanced ranking system with comprehensive novelty detection"""
    
    def __init__(self, humor_weight: float = HUMOR_WEIGHT,
                 diversity_weight: float = DIVERSITY_WEIGHT,
                 novelty_weight: float = NOVELTY_WEIGHT):
        self.humor_weight = humor_weight
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        
        self.novelty_detector = AdvancedNoveltyDetector()
        self.diversity_analyzer = DiversityAnalyzer()
        
        print(f"Advanced ranking: {humor_weight:.2f} humor + {diversity_weight:.2f} diversity + {novelty_weight:.2f} novelty")
    
    def calculate_weighted_score(self, joke: NoveltyJoke, all_jokes: List[NoveltyJoke], topic: str) -> float:
        """Calculate comprehensive weighted score"""
        
        # 1. Humor score (normalized ELO)
        humor_score = joke.elo_rating / 1500.0
        joke.metrics.humor_score = humor_score
        
        # 2. Diversity contribution
        diversity_contribution = self.diversity_analyzer.calculate_joke_diversity_contribution(joke, all_jokes)
        
        # 3. Comprehensive novelty analysis
        novelty_metrics = self.novelty_detector.comprehensive_novelty_analysis(joke, all_jokes, topic)
        joke.metrics.novelty_metrics = novelty_metrics
        
        # 4. Calculate final weighted score
        weighted_score = (
            self.humor_weight * humor_score +
            self.diversity_weight * diversity_contribution +
            self.novelty_weight * novelty_metrics.combined_novelty
        )
        
        joke.metrics.weighted_score = weighted_score
        
        return weighted_score
    
    def rank_jokes(self, jokes: List[NoveltyJoke], topic: str) -> List[NoveltyJoke]:
        """Rank jokes with advanced novelty detection"""
        print(f"Performing advanced ranking with novelty detection for {len(jokes)} jokes...")
        
        # Calculate all metrics
        for joke in jokes:
            self.calculate_weighted_score(joke, jokes, topic)
        
        # Sort by weighted score
        jokes.sort(key=lambda j: j.metrics.weighted_score, reverse=True)
        
        # Report novelty statistics
        avg_novelty = sum(j.metrics.novelty_metrics.combined_novelty for j in jokes) / len(jokes)
        pattern_detected = sum(1 for j in jokes if j.metrics.novelty_metrics.detected_patterns)
        high_memorization_risk = sum(1 for j in jokes if j.metrics.novelty_metrics.memorization_risk > 0.5)
        
        print(f"Novelty analysis: avg={avg_novelty:.2f}, patterns={pattern_detected}, memorization_risk={high_memorization_risk}")
        
        return jokes

class NoveltyPlanSearchGenerator:
    """PlanSearch generator optimized for novelty detection"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.branch_factor = BRANCH_FACTOR
        self.max_depth = MAX_DEPTH
        self.observation_tree = {}
        self.jokes = []
    
    def generate_observations(self, context: Optional[List[str]] = None) -> List[str]:
        """Generate creative, novel observations"""
        if context:
            prompt = f"""Topic: '{self.topic}' | Context: {context}

Generate {self.branch_factor} highly creative observations that:
- Combine the context ideas in unexpected ways
- Avoid clichés and obvious connections
- Find unique angles and perspectives
- Create opportunities for original wordplay
- Suggest contradictions or subversions

Focus on genuine creativity and novelty."""
        else:
            prompt = f"""Topic: '{self.topic}'

Generate {self.branch_factor} highly creative, original observations that:
- Reveal unexpected aspects or contradictions
- Suggest unique wordplay or conceptual connections
- Avoid obvious stereotypes and clichés
- Find fresh, surprising angles
- Enable novel joke construction

Prioritize creativity and originality over predictability."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        observations = []
        
        for response in responses:
            lines = response.split('\n')
            for line in lines:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith(('-', '•', '*')):
                    if cleaned.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        cleaned = cleaned[2:].strip()
                    if cleaned:
                        observations.append(cleaned)
        
        return observations[:self.branch_factor]
    
    def build_observation_tree(self):
        """Build observation tree optimized for novelty"""
        print(f"Building novelty-optimized observation tree for '{self.topic}'...")
        
        # Level 1: Creative base observations
        level1_obs = self.generate_observations()
        self.observation_tree[1] = level1_obs
        print(f"Level 1: {len(level1_obs)} creative observations")
        
        if self.max_depth < 2:
            return
        
        # Level 2: Novel combinations
        level2_obs = []
        for obs_pair in itertools.combinations(level1_obs, 2):
            derived = self.generate_observations(context=list(obs_pair))
            level2_obs.extend([(obs_pair, d) for d in derived[:1]])
        
        self.observation_tree[2] = level2_obs
        print(f"Level 2: {len(level2_obs)} novel combinations")
    
    def generate_joke_from_path(self, observation_path: List[str]) -> str:
        """Generate highly original joke"""
        prompt = f"""Topic: {self.topic}
Observations: {' | '.join(observation_path)}

Create an exceptionally original, creative joke that:
- Uses these observations in a novel way
- Avoids all common joke patterns and clichés
- Employs unexpected wordplay or conceptual connections
- Has perfect timing and structure
- Is genuinely funny through creativity, not familiarity

Generate only the joke text - no explanations."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        return responses[0].strip()
    
    def generate_all_jokes(self) -> List[NoveltyJoke]:
        """Generate jokes optimized for novelty"""
        print("Generating jokes with novelty optimization...")
        jokes = []
        
        # Direct observation jokes
        for obs in self.observation_tree.get(1, []):
            joke_text = self.generate_joke_from_path([obs])
            jokes.append(NoveltyJoke(text=joke_text, path=[obs]))
        
        # Combined observation jokes
        for (obs_pair, derived_obs) in self.observation_tree.get(2, []):
            path = list(obs_pair) + [derived_obs]
            joke_text = self.generate_joke_from_path(path)
            jokes.append(NoveltyJoke(text=joke_text, path=path))
        
        # Contradiction/subversion jokes
        for obs in self.observation_tree.get(1, [])[:3]:
            prompt = f"""Topic: {self.topic} | Observation: {obs}

Create an exceptionally clever joke that completely subverts or contradicts this observation.
Avoid all clichéd contradiction patterns. Be genuinely creative and original.
Just the joke text."""
            responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
            jokes.append(NoveltyJoke(text=responses[0].strip(), path=[obs, "[subverted]"]))
        
        print(f"Generated {len(jokes)} jokes for novelty analysis")
        self.jokes = jokes
        return jokes

class NoveltyEloSystem:
    """ELO system with enhanced judging criteria"""
    
    def __init__(self, k_factor: float = ELO_K_FACTOR):
        self.k_factor = k_factor
        self.comparison_count = 0
        self.comparison_cache = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, joke_a: NoveltyJoke, joke_b: NoveltyJoke, winner: str) -> Tuple[float, float]:
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
    
    def judge_pair(self, joke_a: NoveltyJoke, joke_b: NoveltyJoke) -> str:
        """Judge with emphasis on humor quality and originality"""
        cache_key = (joke_a.id, joke_b.id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        prompt = f"""Compare these jokes for overall quality and humor:

Joke A: {joke_a.text}

Joke B: {joke_b.text}

Consider:
- Humor quality and comedic timing
- Originality and creativity (avoid clichés)
- Cleverness of concept or wordplay
- Overall comedic impact and memorability

Which joke is better overall? Reply 'A' or 'B'."""
        
        responses = llm_complete(prompt, n=1, temperature=0.1)
        winner = responses[0].strip().upper()
        
        if winner not in ["A", "B"]:
            winner = random.choice(["A", "B"])
        
        self.comparison_cache[cache_key] = winner
        return winner
    
    def run_tournament(self, jokes: List[NoveltyJoke], rounds: int = 3) -> List[NoveltyJoke]:
        """Run ELO tournament with novelty awareness"""
        print(f"Running novelty-aware ELO tournament with {len(jokes)} jokes...")
        
        n_jokes = len(jokes)
        if n_jokes < 2:
            return jokes
        
        total_comparisons = rounds * n_jokes
        
        for i in range(total_comparisons):
            if i % 8 == 0:
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
        print(f"Novelty-aware tournament complete. {self.comparison_count} comparisons made.")
        return jokes

def generate_novelty_jokes(topic: str, top_n: int = 5) -> List[NoveltyJoke]:
    """Generate jokes with comprehensive novelty detection"""
    
    # Generate jokes
    generator = NoveltyPlanSearchGenerator(topic)
    generator.build_observation_tree()
    jokes = generator.generate_all_jokes()
    
    # ELO ranking
    elo_system = NoveltyEloSystem()
    elo_ranked = elo_system.run_tournament(jokes, rounds=3)
    
    # Advanced ranking with novelty detection
    ranking_system = NoveltyRankingSystem()
    final_ranked = ranking_system.rank_jokes(elo_ranked, topic)
    
    return final_ranked[:top_n]

def interactive_mode():
    """Interactive mode with comprehensive novelty analysis"""
    print("\n🎭 Advanced Novelty Detection AI Joke Generator (Version 4)")
    print("=" * 65)
    print("Features: Comprehensive Novelty Detection + Pattern Recognition + Memorization Detection")
    
    while True:
        topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using the novelty detection joke generator! 😄")
            break
        
        if not topic:
            print("Please enter a valid topic.")
            continue
        
        print(f"\nGenerating jokes about '{topic}' with advanced novelty detection...")
        print("This may take 2-4 minutes for comprehensive analysis...\n")
        
        try:
            start_time = time.time()
            top_jokes = generate_novelty_jokes(topic, top_n=5)
            elapsed_time = time.time() - start_time
            
            print(f"\n🌟 Top 5 Jokes with Novelty Analysis for '{topic}':")
            print("=" * 65)
            
            for i, joke in enumerate(top_jokes, 1):
                print(f"\n{i}. {joke.text}")
                print(f"   📊 Overall Score: {joke.metrics.weighted_score:.3f}")
                
                # Novelty breakdown
                nm = joke.metrics.novelty_metrics
                print(f"   🔍 Novelty Analysis:")
                print(f"     • Combined Novelty: {nm.combined_novelty:.2f}")
                print(f"     • Pattern Score: {nm.pattern_score:.2f} " + 
                      (f"(detected: {', '.join(nm.detected_patterns)})" if nm.detected_patterns else "(original)"))
                print(f"     • Semantic Novelty: {nm.semantic_novelty:.2f}")
                print(f"     • Structural Novelty: {nm.structural_novelty:.2f}")
                print(f"     • Conceptual Novelty: {nm.conceptual_novelty:.2f}")
                
                if nm.memorization_risk > 0.3:
                    print(f"     ⚠️  Memorization Risk: {nm.memorization_risk:.2f}")
                
                if nm.similar_jokes:
                    print(f"     ⚠️  Similar Jokes Found: {len(nm.similar_jokes)}")
                
                # Other metrics
                print(f"   📈 Components: Humor={joke.metrics.humor_score:.2f}, "
                      f"Diversity={joke.metrics.diversity_contribution:.2f}")
                
                if joke.metrics.taxonomy:
                    top_dims = sorted(joke.metrics.taxonomy.items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                    dims_str = ', '.join(f'{d[0]}({d[1]:.1f})' for d in top_dims)
                    print(f"   🎭 Style: {dims_str}")
            
            print(f"\n⏱️  Generated in {elapsed_time:.1f} seconds")
            print(f"🔬 Advanced novelty detection completed successfully")
            
        except Exception as e:
            print(f"\nError generating jokes: {e}")
            print("Please try again with a different topic.")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] != "interactive":
        topic = sys.argv[1]
        print(f"\n🎭 Advanced Novelty Detection for: {topic}")
        print("=" * 65)
        
        top_jokes = generate_novelty_jokes(topic, top_n=5)
        
        print("\n🌟 Top 5 Jokes with Novelty Analysis:")
        print("=" * 65)
        
        for i, joke in enumerate(top_jokes, 1):
            print(f"\n{i}. {joke.text}")
            print(f"   Overall Score: {joke.metrics.weighted_score:.3f}")
            print(f"   Novelty: {joke.metrics.novelty_metrics.combined_novelty:.2f} " +
                  f"(Pattern: {joke.metrics.novelty_metrics.pattern_score:.2f})")
            if joke.metrics.novelty_metrics.detected_patterns:
                print(f"   Detected Patterns: {', '.join(joke.metrics.novelty_metrics.detected_patterns)}")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()