#!/usr/bin/env python3
"""
Basic Joke Generator - Version 1
Plain PlanSearch with simple ELO rating system
"""

import os
import sys
import random
import itertools
import hashlib
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

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

@dataclass
class BasicJoke:
    text: str
    path: List[str]
    elo_rating: float = INITIAL_ELO
    id: str = ""
    
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
                model="gpt-4o-mini",  # Using mini for cost efficiency
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

class BasicPlanSearchGenerator:
    """Basic PlanSearch joke generator without any optimizations"""
    
    def __init__(self, topic: str, branch_factor: int = BRANCH_FACTOR, max_depth: int = MAX_DEPTH):
        self.topic = topic
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.observation_tree = {}
        self.jokes = []
    
    def generate_observations(self, context: Optional[List[str]] = None) -> List[str]:
        """Generate observations about the topic"""
        if context:
            prompt = f"""Given the topic '{self.topic}' and these observations:
{chr(10).join(f'- {obs}' for obs in context)}

Generate {self.branch_factor} new observations that combine or extend these ideas.
Make them comedy-oriented, funny and specific."""
        else:
            prompt = f"""Generate {self.branch_factor} diverse observations about '{self.topic}'.
Include stereotypes, wordplay opportunities, and absurd angles.
Make them specific and comedy-oriented."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        observations = []
        
        for response in responses:
            lines = response.split('\n')
            for line in lines:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith('-'):
                    # Remove numbering if present
                    if cleaned.startswith(('1.', '2.', '3.', '4.', '5.')):
                        cleaned = cleaned[2:].strip()
                    if cleaned:
                        observations.append(cleaned)
        
        return observations[:self.branch_factor]
    
    def build_observation_tree(self):
        """Build hierarchical observation tree"""
        print(f"Building observation tree for '{self.topic}'...")
        
        # Level 1: Basic observations
        level1_obs = self.generate_observations()
        self.observation_tree[1] = level1_obs
        print(f"Generated {len(level1_obs)} level-1 observations")
        
        if self.max_depth < 2:
            return
        
        # Level 2: Combinations of level 1 observations
        level2_obs = []
        for obs_pair in itertools.combinations(level1_obs, 2):
            derived = self.generate_observations(context=list(obs_pair))
            level2_obs.extend([(obs_pair, d) for d in derived[:2]])
        
        self.observation_tree[2] = level2_obs
        print(f"Generated {len(level2_obs)} level-2 observations")
    
    def generate_joke_from_path(self, observation_path: List[str]) -> str:
        """Generate a joke from a path of observations"""
        prompt = f"""Topic: {self.topic}

Observations:
{chr(10).join(f'- {obs}' for obs in observation_path)}

Create a single, short, punchy joke that uses these observations.
The joke should be self-contained and funny.
Format: Just the joke text, no explanations."""
        
        responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
        return responses[0].strip()
    
    def generate_all_jokes(self) -> List[BasicJoke]:
        """Generate jokes from all observation paths"""
        print("Generating jokes from observation paths...")
        jokes = []
        
        # Jokes from level 1 observations
        for obs in self.observation_tree.get(1, []):
            joke_text = self.generate_joke_from_path([obs])
            jokes.append(BasicJoke(text=joke_text, path=[obs]))
        
        # Jokes from level 2 observation combinations
        for (obs_pair, derived_obs) in self.observation_tree.get(2, []):
            path = list(obs_pair) + [derived_obs]
            joke_text = self.generate_joke_from_path(path)
            jokes.append(BasicJoke(text=joke_text, path=path))
        
        # Contradiction jokes (subvert level 1 observations)
        for obs in self.observation_tree.get(1, []):
            prompt = f"""Topic: {self.topic}
Observation: {obs}

Generate a joke that CONTRADICTS or subverts this observation.
Just the joke, no explanation."""
            responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
            jokes.append(BasicJoke(text=responses[0].strip(), path=[obs, "[contradicted]"]))
        
        self.jokes = jokes
        print(f"Generated {len(jokes)} total jokes")
        return jokes

class BasicEloSystem:
    """Simple ELO rating system for joke evaluation"""
    
    def __init__(self, k_factor: float = ELO_K_FACTOR):
        self.k_factor = k_factor
        self.comparison_count = 0
        self.comparison_cache = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for joke A vs joke B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, joke_a: BasicJoke, joke_b: BasicJoke, winner: str) -> Tuple[float, float]:
        """Update ELO ratings based on comparison result"""
        expected_a = self.expected_score(joke_a.elo_rating, joke_b.elo_rating)
        expected_b = 1 - expected_a
        
        if winner == "A":
            score_a, score_b = 1, 0
        elif winner == "B":
            score_a, score_b = 0, 1
        else:  # Tie
            score_a, score_b = 0.5, 0.5
        
        new_rating_a = joke_a.elo_rating + self.k_factor * (score_a - expected_a)
        new_rating_b = joke_b.elo_rating + self.k_factor * (score_b - expected_b)
        
        return new_rating_a, new_rating_b
    
    def judge_pair(self, joke_a: BasicJoke, joke_b: BasicJoke) -> str:
        """Use LLM to judge which joke is funnier"""
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
            winner = random.choice(["A", "B"])  # Fallback for unclear responses
        
        self.comparison_cache[cache_key] = winner
        return winner
    
    def run_tournament(self, jokes: List[BasicJoke], rounds: int = 3) -> List[BasicJoke]:
        """Run ELO tournament to rank jokes"""
        print(f"Running ELO tournament with {len(jokes)} jokes...")
        
        n_jokes = len(jokes)
        if n_jokes < 2:
            return jokes
        
        total_comparisons = rounds * n_jokes
        
        for i in range(total_comparisons):
            if i % 10 == 0:
                print(f"  Comparison {i}/{total_comparisons}...")
            
            # Select two random jokes to compare
            idx_a, idx_b = random.sample(range(n_jokes), 2)
            joke_a, joke_b = jokes[idx_a], jokes[idx_b]
            
            # Skip if already compared
            cache_key = (joke_a.id, joke_b.id)
            if cache_key in self.comparison_cache:
                continue
            
            # Judge and update ratings
            winner = self.judge_pair(joke_a, joke_b)
            new_rating_a, new_rating_b = self.update_ratings(joke_a, joke_b, winner)
            
            jokes[idx_a].elo_rating = new_rating_a
            jokes[idx_b].elo_rating = new_rating_b
            
            self.comparison_count += 1
        
        # Sort by ELO rating
        jokes.sort(key=lambda j: j.elo_rating, reverse=True)
        print(f"Tournament complete. {self.comparison_count} total comparisons made.")
        return jokes

def generate_basic_jokes(topic: str, top_n: int = 5) -> List[BasicJoke]:
    """Main function to generate and rank jokes"""
    # Generate jokes
    generator = BasicPlanSearchGenerator(topic)
    generator.build_observation_tree()
    jokes = generator.generate_all_jokes()
    
    # Rank with ELO system
    elo_system = BasicEloSystem()
    ranked_jokes = elo_system.run_tournament(jokes, rounds=3)
    
    return ranked_jokes[:top_n]

def interactive_mode():
    """Interactive joke generation mode"""
    print("\n🎭 Basic AI Joke Generator (Version 1)")
    print("=" * 50)
    print("Features: PlanSearch + Basic ELO Rating")
    
    while True:
        topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using the basic joke generator! 😄")
            break
        
        if not topic:
            print("Please enter a valid topic.")
            continue
        
        print(f"\nGenerating jokes about '{topic}'...")
        print("This may take 1-2 minutes...\n")
        
        try:
            start_time = time.time()
            top_jokes = generate_basic_jokes(topic, top_n=5)
            elapsed_time = time.time() - start_time
            
            print(f"\n🌟 Top 5 Basic Jokes about '{topic}':")
            print("=" * 50)
            
            for i, joke in enumerate(top_jokes, 1):
                print(f"\n{i}. {joke.text}")
                print(f"   ELO Rating: {joke.elo_rating:.0f}")
                path_display = ' → '.join(joke.path[:2])
                if len(joke.path) > 2:
                    path_display += '...'
                print(f"   Path: {path_display}")
            
            print(f"\n⏱️  Generated in {elapsed_time:.1f} seconds")
            
        except Exception as e:
            print(f"\nError generating jokes: {e}")
            print("Please try again with a different topic.")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] != "interactive":
        topic = sys.argv[1]
        print(f"\n🎭 Basic Joke Generation for: {topic}")
        print("=" * 50)
        
        top_jokes = generate_basic_jokes(topic, top_n=5)
        
        print("\n🌟 Top 5 Basic Jokes:")
        print("=" * 50)
        
        for i, joke in enumerate(top_jokes, 1):
            print(f"\n{i}. [ELO: {joke.elo_rating:.0f}] {joke.text}")
            path_display = ' → '.join(joke.path[:2])
            if len(joke.path) > 2:
                path_display += '...'
            print(f"   Generation Path: {path_display}")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()