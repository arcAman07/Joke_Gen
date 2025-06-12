#!/usr/bin/env python3
"""
Basic AI Joke Generator with Multi-Provider Support
Supports OpenAI GPT, Anthropic Claude, and Google Gemini
Simple PlanSearch + ELO Rating System
"""

import os
import sys
import random
import itertools
import hashlib
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configuration
TEMPERATURE = 0.9
BRANCH_FACTOR = 5
MAX_DEPTH = 2
ELO_K_FACTOR = 32
INITIAL_ELO = 1200

print("🎭 Basic AI Joke Generator with Multi-Provider Support")
print("Supports: OpenAI GPT, Anthropic Claude, Google Gemini")


@dataclass
class Joke:
    text: str
    path: List[str]
    provider: str
    elo_rating: float = INITIAL_ELO
    id: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(f"{self.text}{self.provider}".encode()).hexdigest()[:8]


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.9) -> str:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            print("✅ OpenAI provider initialized")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    @property
    def name(self) -> str:
        return "OpenAI"
    
    def complete(self, prompt: str, temperature: float = 0.9) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "Error generating response"

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            print("✅ Anthropic provider initialized")
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    @property
    def name(self) -> str:
        return "Anthropic"
    
    def complete(self, prompt: str, temperature: float = 0.9) -> str:
        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return "Error generating response"

class GeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.genai = genai
            print("✅ Gemini provider initialized")
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    @property
    def name(self) -> str:
        return "Gemini"
    
    def complete(self, prompt: str, temperature: float = 0.9) -> str:
        try:
            generation_config = self.genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=400,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "Error generating response"

class FallbackProvider(LLMProvider):
    """Fallback provider when no API keys are available"""
    
    def __init__(self):
        print("⚠️  Warning: Using fallback mode with limited functionality.")
        self.templates = {
            "observations": [
                "People often have strong opinions about {topic}",
                "{topic} has changed significantly with technology",
                "There are many stereotypes about {topic}",
                "{topic} appears frequently in popular culture",
                "Most people misunderstand how {topic} actually works"
            ],
            "jokes": [
                "Why did the {topic} become a comedian? It had perfect timing!",
                "My {topic} is so advanced, it tells jokes I don't understand yet",
                "A {topic} walks into a bar. The bartender says, 'We don't serve your type here'",
                "What do you call a {topic} that tells jokes? A laugh-ing matter!",
                "The {topic} tried to tell a joke, but it was too technical for the audience"
            ]
        }
    
    @property
    def name(self) -> str:
        return "Fallback"
    
    def complete(self, prompt: str, temperature: float = 0.9) -> str:
        topic = self._extract_topic(prompt)
        
        if "observations" in prompt.lower():
            return "\n".join(template.format(topic=topic) 
                           for template in random.sample(self.templates["observations"], 3))
        elif "joke" in prompt.lower():
            template = random.choice(self.templates["jokes"])
            return template.format(topic=topic)
        elif "which joke is funnier" in prompt.lower():
            return random.choice(["A", "B"])
        else:
            return f"Generated response about {topic}"
    
    def _extract_topic(self, prompt: str) -> str:
        if "topic '" in prompt:
            start = prompt.find("topic '") + 7
            end = prompt.find("'", start)
            return prompt[start:end] if end > start else "something"
        return "something"

# ====================================================================
# PROVIDER MANAGER
# ====================================================================

class ProviderManager:
    """Manages multiple LLM providers with automatic initialization"""
    
    def __init__(self):
        self.providers = {}
        self.active_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Try to initialize available providers"""
        
        # Try OpenAI first
        try:
            if os.environ.get("OPENAI_API_KEY"):
                self.providers["openai"] = OpenAIProvider()
                if not self.active_provider:
                    self.active_provider = "openai"
        except Exception as e:
            print(f"OpenAI not available: {e}")
        
        # Try Anthropic
        try:
            if os.environ.get("ANTHROPIC_API_KEY"):
                self.providers["anthropic"] = AnthropicProvider()
                if not self.active_provider:
                    self.active_provider = "anthropic"
        except Exception as e:
            print(f"Anthropic not available: {e}")
        
        # Try Gemini
        try:
            if os.environ.get("GEMINI_API_KEY"):
                self.providers["gemini"] = GeminiProvider()
                if not self.active_provider:
                    self.active_provider = "gemini"
        except Exception as e:
            print(f"Gemini not available: {e}")
        
        # Fallback if no providers work
        if not self.providers:
            print("No API providers available, using fallback mode")
            self.providers["fallback"] = FallbackProvider()
            self.active_provider = "fallback"
        
        print(f"Active provider: {self.active_provider}")
        if len(self.providers) > 1:
            print(f"Available providers: {list(self.providers.keys())}")
    
    def get_provider(self, name: Optional[str] = None) -> LLMProvider:
        """Get specific provider or active one"""
        if name and name in self.providers:
            return self.providers[name]
        return self.providers[self.active_provider]
    
    def list_providers(self) -> List[str]:
        """List all available providers"""
        return list(self.providers.keys())

# Global provider manager
provider_manager = ProviderManager()

def llm_complete(prompt: str, provider: str = None, temperature: float = 0.9) -> str:
    """Complete prompt using specified or active provider"""
    target_provider = provider_manager.get_provider(provider)
    return target_provider.complete(prompt, temperature)

# ====================================================================
# PLANSEARCH JOKE GENERATOR
# ====================================================================

class BasicJokeGenerator:
    """Basic PlanSearch joke generator"""
    
    def __init__(self, topic: str, provider: str = None):
        self.topic = topic
        self.provider = provider or provider_manager.active_provider
        self.branch_factor = BRANCH_FACTOR
        self.max_depth = MAX_DEPTH
        self.observation_tree = {}
        self.jokes = []
    
    def generate_observations(self, context: Optional[List[str]] = None) -> List[str]:
        """Generate observations about the topic"""
        if context:
            prompt = f"""Given the topic '{self.topic}' and these observations:
{chr(10).join(f'- {obs}' for obs in context)}

Generate {self.branch_factor} NEW observations that combine or extend these ideas in unexpected ways.
Focus on contradictions, wordplay potential, and absurd connections."""
        else:
            prompt = f"""Generate {self.branch_factor} diverse observations about '{self.topic}'.
Include: stereotypes, wordplay opportunities, cultural references, technical aspects, and absurd angles.
Make them specific and comedy-oriented."""
        
        response = llm_complete(prompt, self.provider, TEMPERATURE)
        observations = []
        
        for line in response.split('\n'):
            cleaned = line.strip()
            if cleaned and not cleaned.startswith('-'):
                if cleaned.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    cleaned = cleaned[2:].strip()
                if cleaned:
                    observations.append(cleaned)
        
        return observations[:self.branch_factor]
    
    def build_observation_tree(self):
        """Build hierarchical observation tree"""
        print(f"Building observation tree for '{self.topic}' using {self.provider}...")
        
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
        print(f"Generated {len(level2_obs)} level-2 observation combinations")
    
    def generate_joke_from_path(self, observation_path: List[str]) -> str:
        """Generate a joke from a path of observations"""
        prompt = f"""Topic: {self.topic}

Observations:
{chr(10).join(f'- {obs}' for obs in observation_path)}

Create a single, short, punchy joke that cleverly uses these observations.
The joke should be self-contained and not require explanation.
Format: Just the joke text, no explanations or meta-commentary."""
        
        response = llm_complete(prompt, self.provider, TEMPERATURE)
        return response.strip()
    
    def generate_all_jokes(self) -> List[Joke]:
        """Generate jokes from all observation paths"""
        print("Generating jokes from observation paths...")
        jokes = []
        
        # Jokes from level 1 observations
        for obs in self.observation_tree.get(1, []):
            joke_text = self.generate_joke_from_path([obs])
            jokes.append(Joke(text=joke_text, path=[obs], provider=self.provider))
        
        # Jokes from level 2 observation combinations
        for (obs_pair, derived_obs) in self.observation_tree.get(2, []):
            path = list(obs_pair) + [derived_obs]
            joke_text = self.generate_joke_from_path(path)
            jokes.append(Joke(text=joke_text, path=path, provider=self.provider))
        
        # Contradiction jokes (subvert level 1 observations)
        for obs in self.observation_tree.get(1, []):
            prompt = f"""Topic: {self.topic}
Observation: {obs}

Now generate a joke that CONTRADICTS or subverts this observation.
Just the joke, no explanation."""
            response = llm_complete(prompt, self.provider, TEMPERATURE)
            jokes.append(Joke(text=response.strip(), path=[obs, "[contradicted]"], provider=self.provider))
        
        self.jokes = jokes
        print(f"Generated {len(jokes)} jokes total")
        return jokes

# ====================================================================
# ELO RATING SYSTEM
# ====================================================================

class EloRatingSystem:
    """ELO rating system for joke quality assessment"""
    
    def __init__(self, k_factor: float = ELO_K_FACTOR):
        self.k_factor = k_factor
        self.comparison_count = 0
        self.comparison_cache = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for ELO update"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, joke_a: Joke, joke_b: Joke, winner: str) -> Tuple[float, float]:
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
    
    def judge_pair(self, joke_a: Joke, joke_b: Joke) -> str:
        """Use LLM to judge which joke is funnier"""
        cache_key = (joke_a.id, joke_b.id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        prompt = f"""Which joke is funnier?

Joke A: {joke_a.text}

Joke B: {joke_b.text}

Reply with only 'A' or 'B'."""
        
        # Use consistent provider for judging
        judge_provider = joke_a.provider if joke_a.provider == joke_b.provider else None
        response = llm_complete(prompt, judge_provider, temperature=0.1)
        winner = response.strip().upper()
        
        if winner not in ["A", "B"]:
            winner = random.choice(["A", "B"])
        
        self.comparison_cache[cache_key] = winner
        return winner
    
    def run_tournament(self, jokes: List[Joke], rounds: int = 3) -> List[Joke]:
        """Run ELO tournament to rank jokes"""
        print(f"Running tournament with {len(jokes)} jokes...")
        
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
        
        jokes.sort(key=lambda j: j.elo_rating, reverse=True)
        print(f"Tournament complete. {self.comparison_count} total comparisons made.")
        return jokes

# ====================================================================
# MAIN FUNCTIONS
# ====================================================================

def generate_basic_jokes(topic: str, provider: str = None, top_n: int = 5) -> List[Joke]:
    """Main function to generate and rank jokes"""
    
    print(f"\n🎭 Generating jokes about '{topic}'")
    if provider:
        print(f"Using provider: {provider}")
    print("=" * 50)
    
    start_time = time.time()
    
    # Generate jokes
    generator = BasicJokeGenerator(topic, provider)
    generator.build_observation_tree()
    jokes = generator.generate_all_jokes()
    
    # Rank with ELO system
    elo_system = EloRatingSystem()
    ranked_jokes = elo_system.run_tournament(jokes, rounds=3)
    
    elapsed_time = time.time() - start_time
    
    # Display results
    print(f"\n🌟 Top {top_n} Jokes about '{topic}':")
    print("=" * 50)
    
    for i, joke in enumerate(ranked_jokes[:top_n], 1):
        print(f"\n{i}. {joke.text}")
        print(f"   Provider: {joke.provider}")
        print(f"   ELO Rating: {joke.elo_rating:.0f}")
        path_display = ' → '.join(joke.path[:2])
        if len(joke.path) > 2:
            path_display += '...'
        print(f"   Path: {path_display}")
    
    print(f"\n⏱️  Generated in {elapsed_time:.1f} seconds")
    print(f"📊 Statistics: {len(jokes)} jokes, {elo_system.comparison_count} comparisons")
    
    return ranked_jokes[:top_n]

def compare_providers(topic: str) -> None:
    """Compare joke generation across all available providers"""
    
    available_providers = provider_manager.list_providers()
    
    if len(available_providers) < 2:
        print("Need multiple providers for comparison")
        return
    
    print(f"\n🔬 Provider Comparison for '{topic}'")
    print(f"Testing providers: {', '.join(available_providers)}")
    print("=" * 60)
    
    results = {}
    
    for provider in available_providers:
        print(f"\n--- Testing {provider} ---")
        try:
            jokes = generate_basic_jokes(topic, provider, top_n=3)
            results[provider] = jokes
        except Exception as e:
            print(f"Error with {provider}: {e}")
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 30)
        
        for provider, jokes in results.items():
            avg_rating = sum(j.elo_rating for j in jokes) / len(jokes)
            print(f"{provider:12}: Avg ELO = {avg_rating:.0f}")
            print(f"              Best joke: {jokes[0].text[:60]}...")

def interactive_basic_mode():
    """Interactive joke generation mode"""
    
    print("\n🎭 Basic AI Joke Generator - Interactive Mode")
    print("=" * 50)
    
    available_providers = provider_manager.list_providers()
    print(f"Available providers: {', '.join(available_providers)}")
    
    while True:
        print("\nOptions:")
        print("1. Generate jokes (active provider)")
        print("2. Choose specific provider")
        print("3. Compare all providers")
        print("4. View provider status")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "5":
            print("\nThanks for using the basic joke generator! 😄")
            break
        
        elif choice == "4":
            print(f"\nProvider Status:")
            print(f"Active: {provider_manager.active_provider}")
            print(f"Available: {', '.join(available_providers)}")
            continue
        
        # Get topic for generation options
        if choice in ["1", "2", "3"]:
            topic = input("Enter topic: ").strip()
            if not topic:
                print("Please enter a valid topic.")
                continue
        
        if choice == "1":
            generate_basic_jokes(topic)
        
        elif choice == "2":
            print(f"Available providers: {', '.join(available_providers)}")
            provider = input("Choose provider: ").strip()
            if provider in available_providers:
                generate_basic_jokes(topic, provider)
            else:
                print("Invalid provider selected.")
        
        elif choice == "3":
            compare_providers(topic)
        
        else:
            print("Invalid choice. Please select 1-5.")

def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        # Command line mode
        topic = sys.argv[1]
        provider = sys.argv[2] if len(sys.argv) > 2 else None
        
        if provider == "compare":
            compare_providers(topic)
        elif provider and provider in provider_manager.list_providers():
            generate_basic_jokes(topic, provider)
        else:
            generate_basic_jokes(topic)
    else:
        # Interactive mode
        interactive_basic_mode()

if __name__ == "__main__":
    main()

# ====================================================================
# USAGE EXAMPLES
# ====================================================================

"""
# Set up API keys first:
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"  # optional
export GEMINI_API_KEY="your_key_here"     # optional

# Command line usage:
python basic_joke_gen.py "artificial intelligence"
python basic_joke_gen.py "cats" openai
python basic_joke_gen.py "programming" compare

# Interactive mode:
python basic_joke_gen.py

# Programmatic usage:
jokes = generate_basic_jokes("machine learning", "anthropic")
compare_providers("quantum computing")
"""