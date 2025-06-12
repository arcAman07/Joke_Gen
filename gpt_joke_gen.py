#!/usr/bin/env python3

import os
import sys
import json
import random
import itertools
import hashlib
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    import openai
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

TEMPERATURE = 0.9
BRANCH_FACTOR = 5
MAX_DEPTH = 2
ELO_K_FACTOR = 32
INITIAL_ELO = 1200

@dataclass
class Joke:
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

llm_client = None

def llm_complete(prompt: str, n: int = 1, temperature: float = 0.9) -> List[str]:
    global llm_client
    if llm_client is None:
        llm_client = OpenAIClient()
    
    return llm_client.complete(prompt, n, temperature)

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
        
        for obs in self.observation_tree.get(1, []):
            joke_text = self.generate_joke_from_path([obs])
            jokes.append(Joke(text=joke_text, path=[obs]))
        
        for (obs_pair, derived_obs) in self.observation_tree.get(2, []):
            path = list(obs_pair) + [derived_obs]
            joke_text = self.generate_joke_from_path(path)
            jokes.append(Joke(text=joke_text, path=path))
        
        for obs in self.observation_tree.get(1, []):
            prompt = f"""Topic: {self.topic}
Observation: {obs}

Now generate a joke that CONTRADICTS or subverts this observation.
Just the joke, no explanation."""
            responses = llm_complete(prompt, n=1, temperature=TEMPERATURE)
            jokes.append(Joke(text=responses[0].strip(), path=[obs, "[contradicted]"]))
        
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
        
        total_comparisons = rounds * n_jokes
        
        for i in range(total_comparisons):
            if i % 10 == 0:
                print(f"  Comparison {i}/{total_comparisons}...")
            
            idx_a, idx_b = random.sample(range(n_jokes), 2)
            joke_a, joke_b = jokes[idx_a], jokes[idx_b]
            
            winner = self.judge_pair(joke_a, joke_b)
            
            new_rating_a, new_rating_b = self.update_ratings(joke_a, joke_b, winner)
            jokes[idx_a].elo_rating = new_rating_a
            jokes[idx_b].elo_rating = new_rating_b
            
            self.comparison_count += 1
        
        jokes.sort(key=lambda j: j.elo_rating, reverse=True)
        return jokes

def generate_top_jokes(topic: str, top_n: int = 5) -> List[Joke]:
    generator = PlanSearchJokeGenerator(topic)
    generator.build_observation_tree()
    
    jokes = generator.generate_all_jokes()
    print(f"\nGenerated {len(jokes)} jokes total")
    
    elo_system = EloRatingSystem()
    ranked_jokes = elo_system.run_tournament(jokes, rounds=3)
    
    return ranked_jokes[:top_n]

def interactive_joke_generator():
    print("\n🎭 Welcome to the AI Joke Generator!")
    print("=" * 50)
    
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
            print("=" * 50)
            
            for i, joke in enumerate(top_jokes, 1):
                print(f"\n{i}. {joke.text}")
                print(f"   Rating: {joke.elo_rating:.0f}")
            
            print(f"\n⏱️  Generated in {elapsed_time:.1f} seconds")
            
        except Exception as e:
            print(f"\nError generating jokes: {e}")
            print("Please try again with a different topic.")

def main(topic: str):
    print(f"\n🎭 Generating jokes about: {topic}")
    print("=" * 50)
    
    top_jokes = generate_top_jokes(topic, top_n=5)
    
    print("\n🌟 Top 5 Jokes:")
    print("=" * 50)
    
    for i, joke in enumerate(top_jokes, 1):
        print(f"\n{i}. [Rating: {joke.elo_rating:.0f}] {joke.text}")
        path_display = ' → '.join(joke.path[:2])
        if len(joke.path) > 2:
            path_display += '...'
        print(f"   Path: {path_display}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "interactive":
        topic = sys.argv[1]
        main(topic)
    else:
        interactive_joke_generator()