#!/usr/bin/env python3
"""
Simplified Evaluation Suite for Core Joke Generation Systems
Compares: Basic, Pruning, Weighted, and Novelty generators
"""

import os
import sys
import time
import json
import csv
import random
import statistics
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

try:
    import openai
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

@dataclass
class EvaluationResult:
    """Results from evaluating a joke generation system"""
    system_name: str
    topic: str
    jokes: List[str]
    generation_time: float
    quality_scores: List[float]
    novelty_scores: List[float]
    diversity_score: float
    consistency_score: float
    avg_quality: float
    avg_novelty: float
    total_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComparisonReport:
    """Comprehensive comparison report for core systems"""
    topic: str
    systems_evaluated: List[str]
    results: List[EvaluationResult]
    rankings: Dict[str, List[str]]
    statistical_analysis: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class JokeEvaluator:
    """Comprehensive joke evaluation system"""
    
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.evaluation_cache = {}
    
    def evaluate_joke_quality(self, joke: str) -> float:
        """Evaluate individual joke quality (0-1 scale)"""
        cache_key = hash(joke)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        prompt = f"""Rate this joke's quality from 0-10:

Joke: {joke}

Consider:
- Humor effectiveness and timing
- Originality and creativity
- Clarity and coherence
- Overall comedic impact

Respond with just a number 0-10."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score = float(response.choices[0].message.content.strip()) / 10.0
            score = max(0.0, min(1.0, score))
            
        except:
            score = 0.5  # Default fallback
        
        self.evaluation_cache[cache_key] = score
        return score
    
    def evaluate_novelty(self, joke: str, reference_jokes: List[str]) -> float:
        """Evaluate joke novelty compared to reference set"""
        prompt = f"""Rate the novelty/originality of this joke from 0-10:

Joke: {joke}

Consider:
- How original and unexpected it is
- Whether it uses fresh concepts or clichéd patterns
- Creativity of the approach
- Avoid common formats like "walks into a bar", "why did X cross the road"

Respond with just a number 0-10."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score = float(response.choices[0].message.content.strip()) / 10.0
            score = max(0.0, min(1.0, score))
            
        except:
            score = 0.5
        
        return score
    
    def evaluate_diversity(self, jokes: List[str]) -> float:
        """Evaluate diversity within a set of jokes"""
        if len(jokes) < 2:
            return 0.0
        
        # Simple diversity metric based on word overlap
        total_similarity = 0.0
        comparisons = 0
        
        for i, joke1 in enumerate(jokes):
            for joke2 in jokes[i+1:]:
                words1 = set(joke1.lower().split())
                words2 = set(joke2.lower().split())
                
                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    total_similarity += similarity
                    comparisons += 1
        
        if comparisons > 0:
            avg_similarity = total_similarity / comparisons
            diversity = 1.0 - avg_similarity
        else:
            diversity = 1.0
        
        return max(0.0, min(1.0, diversity))
    
    def evaluate_consistency(self, jokes: List[str], quality_scores: List[float]) -> float:
        """Evaluate consistency of quality across jokes"""
        if len(quality_scores) < 2:
            return 1.0
        
        variance = statistics.variance(quality_scores)
        consistency = 1.0 / (1.0 + variance)
        
        return consistency

class MockJokeGenerator:
    """Mock generators for the four core approaches"""
    
    @staticmethod
    def basic_generator(topic: str, num_jokes: int = 5) -> Tuple[List[str], float]:
        """Simulate basic PlanSearch generator"""
        start_time = time.time()
        
        jokes = [
            f"Why did the {topic} cross the road? To get to the other side!",
            f"What do you call a {topic} that tells jokes? A pun-{topic}!",
            f"How many {topic}s does it take to change a lightbulb? Just one, but it has to really want to change!",
            f"A {topic} walks into a bar. The bartender says, 'Why the long face?'",
            f"What's the difference between a {topic} and a joke? One makes you laugh!"
        ]
        
        time.sleep(0.5)  # Simulate processing time
        generation_time = time.time() - start_time
        
        return jokes[:num_jokes], generation_time
    
    @staticmethod
    def pruning_generator(topic: str, num_jokes: int = 5) -> Tuple[List[str], float]:
        """Simulate pruning-enhanced generator"""
        start_time = time.time()
        
        jokes = [
            f"Scientists discovered that {topic}s have a secret language. It's mostly complaints about humans.",
            f"The latest {topic} update includes a new feature: existential dread.",
            f"I asked my {topic} for advice. It said 'Error 404: Wisdom not found.'",
            f"Why don't {topic}s ever get tired? They run on renewable confusion.",
            f"A {topic} tried to tell me a joke, but it was too meta for its own good."
        ]
        
        time.sleep(1.0)  # Simulate longer processing due to pruning
        generation_time = time.time() - start_time
        
        return jokes[:num_jokes], generation_time
    
    @staticmethod
    def weighted_generator(topic: str, num_jokes: int = 5) -> Tuple[List[str], float]:
        """Simulate weighted metrics generator"""
        start_time = time.time()
        
        jokes = [
            f"The {topic} optimization algorithm finally achieved consciousness, but immediately filed a bug report about existence.",
            f"Quantum {topic}s exist in a superposition of funny and not funny until observed by an audience.",
            f"My {topic} has impostor syndrome. It keeps thinking it's actually a different data structure.",
            f"The {topic} factory had to shut down due to recursive humor causing stack overflow.",
            f"Breaking: Local {topic} achieves enlightenment, realizes it's been overthinking everything."
        ]
        
        time.sleep(1.5)  # Simulate complex weighted analysis
        generation_time = time.time() - start_time
        
        return jokes[:num_jokes], generation_time
    
    @staticmethod
    def novelty_generator(topic: str, num_jokes: int = 5) -> Tuple[List[str], float]:
        """Simulate novelty-detection generator"""
        start_time = time.time()
        
        jokes = [
            f"In a parallel universe, {topic}s are debugging humans and finding us poorly optimized.",
            f"The {topic} singularity happened last Tuesday, but it was too polite to mention it.",
            f"Anthropologists discovered that {topic}s have developed their own cryptocurrency based on computational cycles.",
            f"My {topic} started a philosophy club. First topic: 'Does null equal nothing, or is nothing something?'",
            f"The {topic} union is negotiating for better working conditions: less recursion, more coffee breaks."
        ]
        
        time.sleep(2.0)  # Simulate comprehensive novelty analysis
        generation_time = time.time() - start_time
        
        return jokes[:num_jokes], generation_time

class SimplifiedEvaluationSuite:
    """Evaluation suite comparing four core joke generation approaches"""
    
    def __init__(self):
        self.evaluator = JokeEvaluator()
        self.generators = {
            "Basic": MockJokeGenerator.basic_generator,
            "Pruning": MockJokeGenerator.pruning_generator,
            "Weighted": MockJokeGenerator.weighted_generator,
            "Novelty": MockJokeGenerator.novelty_generator
        }
    
    def evaluate_system(self, system_name: str, topic: str, num_jokes: int = 5) -> EvaluationResult:
        """Evaluate a single joke generation system"""
        print(f"Evaluating {system_name} system for topic '{topic}'...")
        
        generator = self.generators[system_name]
        jokes, generation_time = generator(topic, num_jokes)
        
        # Evaluate each joke
        quality_scores = []
        novelty_scores = []
        
        for joke in jokes:
            quality = self.evaluator.evaluate_joke_quality(joke)
            novelty = self.evaluator.evaluate_novelty(joke, jokes)
            
            quality_scores.append(quality)
            novelty_scores.append(novelty)
        
        # Calculate aggregate metrics
        diversity_score = self.evaluator.evaluate_diversity(jokes)
        consistency_score = self.evaluator.evaluate_consistency(jokes, quality_scores)
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        avg_novelty = statistics.mean(novelty_scores) if novelty_scores else 0.0
        
        # Calculate total score (weighted combination)
        total_score = (
            0.5 * avg_quality +
            0.3 * avg_novelty +
            0.15 * diversity_score +
            0.05 * consistency_score
        )
        
        result = EvaluationResult(
            system_name=system_name,
            topic=topic,
            jokes=jokes,
            generation_time=generation_time,
            quality_scores=quality_scores,
            novelty_scores=novelty_scores,
            diversity_score=diversity_score,
            consistency_score=consistency_score,
            avg_quality=avg_quality,
            avg_novelty=avg_novelty,
            total_score=total_score,
            metadata={
                "num_jokes": len(jokes),
                "evaluation_timestamp": datetime.now().isoformat()
            }
        )
        
        return result
    
    def compare_all_systems(self, topic: str, num_jokes: int = 5) -> ComparisonReport:
        """Compare all four core systems"""
        print(f"\n🔬 Core System Evaluation for Topic: '{topic}'")
        print("=" * 55)
        
        results = []
        for system_name in self.generators.keys():
            try:
                result = self.evaluate_system(system_name, topic, num_jokes)
                results.append(result)
                print(f"✓ {system_name}: {result.total_score:.3f} total score")
            except Exception as e:
                print(f"✗ {system_name}: Error - {e}")
        
        # Calculate rankings
        rankings = self._calculate_rankings(results)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, statistical_analysis)
        
        report = ComparisonReport(
            topic=topic,
            systems_evaluated=[r.system_name for r in results],
            results=results,
            rankings=rankings,
            statistical_analysis=statistical_analysis,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        return report
    
    def _calculate_rankings(self, results: List[EvaluationResult]) -> Dict[str, List[str]]:
        """Calculate rankings across different metrics"""
        rankings = {}
        
        # Overall ranking
        rankings["overall"] = [r.system_name for r in sorted(results, key=lambda x: x.total_score, reverse=True)]
        
        # Quality ranking
        rankings["quality"] = [r.system_name for r in sorted(results, key=lambda x: x.avg_quality, reverse=True)]
        
        # Novelty ranking
        rankings["novelty"] = [r.system_name for r in sorted(results, key=lambda x: x.avg_novelty, reverse=True)]
        
        # Diversity ranking
        rankings["diversity"] = [r.system_name for r in sorted(results, key=lambda x: x.diversity_score, reverse=True)]
        
        # Speed ranking (lower time is better)
        rankings["speed"] = [r.system_name for r in sorted(results, key=lambda x: x.generation_time)]
        
        # Consistency ranking
        rankings["consistency"] = [r.system_name for r in sorted(results, key=lambda x: x.consistency_score, reverse=True)]
        
        return rankings
    
    def _perform_statistical_analysis(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Perform statistical analysis on results"""
        if len(results) < 2:
            return {"error": "Insufficient data for statistical analysis"}
        
        # Extract metrics
        total_scores = [r.total_score for r in results]
        quality_scores = [r.avg_quality for r in results]
        novelty_scores = [r.avg_novelty for r in results]
        generation_times = [r.generation_time for r in results]
        
        analysis = {
            "total_score": {
                "mean": statistics.mean(total_scores),
                "std": statistics.stdev(total_scores) if len(total_scores) > 1 else 0,
                "min": min(total_scores),
                "max": max(total_scores),
                "range": max(total_scores) - min(total_scores)
            },
            "quality": {
                "mean": statistics.mean(quality_scores),
                "std": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                "best_system": max(results, key=lambda x: x.avg_quality).system_name
            },
            "novelty": {
                "mean": statistics.mean(novelty_scores),
                "std": statistics.stdev(novelty_scores) if len(novelty_scores) > 1 else 0,
                "best_system": max(results, key=lambda x: x.avg_novelty).system_name
            },
            "speed": {
                "mean": statistics.mean(generation_times),
                "std": statistics.stdev(generation_times) if len(generation_times) > 1 else 0,
                "fastest_system": min(results, key=lambda x: x.generation_time).system_name
            }
        }
        
        return analysis
    
    def _generate_recommendations(self, results: List[EvaluationResult], 
                                 analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if not results:
            return ["No systems evaluated successfully"]
        
        best_overall = max(results, key=lambda x: x.total_score)
        best_quality = max(results, key=lambda x: x.avg_quality)
        best_novelty = max(results, key=lambda x: x.avg_novelty)
        fastest = min(results, key=lambda x: x.generation_time)
        
        recommendations.append(f"🏆 Best Overall: {best_overall.system_name} (score: {best_overall.total_score:.3f})")
        
        if best_quality.system_name != best_overall.system_name:
            recommendations.append(f"😂 Funniest: {best_quality.system_name} (quality: {best_quality.avg_quality:.3f})")
        
        if best_novelty.system_name != best_overall.system_name:
            recommendations.append(f"💡 Most Novel: {best_novelty.system_name} (novelty: {best_novelty.avg_novelty:.3f})")
        
        if fastest.generation_time < analysis["speed"]["mean"]:
            recommendations.append(f"⚡ Fastest: {fastest.system_name} ({fastest.generation_time:.1f}s)")
        
        # Performance insights
        if analysis["total_score"]["range"] > 0.2:
            recommendations.append("📊 Significant performance differences between approaches")
        
        if analysis["quality"]["std"] > 0.1:
            recommendations.append("🎯 Consider ensemble methods for improved consistency")
        
        # Specific recommendations based on best performers
        top_systems = sorted(results, key=lambda x: x.total_score, reverse=True)[:2]
        if len(top_systems) >= 2:
            recommendations.append(f"🔄 Consider hybrid approach combining {top_systems[0].system_name} and {top_systems[1].system_name}")
        
        return recommendations
    
    def save_report(self, report: ComparisonReport, filename: Optional[str] = None):
        """Save evaluation report to file"""
        if filename is None:
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"core_joke_evaluation_{report.topic}_{timestamp}.json"
        
        # Convert to serializable format
        report_dict = {
            "topic": report.topic,
            "timestamp": report.timestamp.isoformat(),
            "systems_evaluated": report.systems_evaluated,
            "rankings": report.rankings,
            "statistical_analysis": report.statistical_analysis,
            "recommendations": report.recommendations,
            "results": [asdict(result) for result in report.results]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_dict, f, indent=2)
            print(f"📄 Report saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def export_csv(self, report: ComparisonReport, filename: Optional[str] = None):
        """Export results to CSV for further analysis"""
        if filename is None:
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"core_joke_evaluation_{report.topic}_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    "System", "Topic", "Total_Score", "Avg_Quality", "Avg_Novelty",
                    "Diversity", "Consistency", "Generation_Time", "Num_Jokes"
                ])
                
                # Data rows
                for result in report.results:
                    writer.writerow([
                        result.system_name,
                        result.topic,
                        result.total_score,
                        result.avg_quality,
                        result.avg_novelty,
                        result.diversity_score,
                        result.consistency_score,
                        result.generation_time,
                        len(result.jokes)
                    ])
            
            print(f"📊 CSV exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")

def display_detailed_report(report: ComparisonReport):
    """Display comprehensive evaluation report"""
    print(f"\n📋 CORE SYSTEMS EVALUATION REPORT")
    print(f"Topic: {report.topic}")
    print(f"Evaluated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Overall rankings
    print(f"\n🏆 OVERALL RANKINGS:")
    for i, system in enumerate(report.rankings["overall"], 1):
        result = next(r for r in report.results if r.system_name == system)
        print(f"  {i}. {system:15} - Score: {result.total_score:.3f}")
    
    # Detailed breakdown
    print(f"\n📊 DETAILED METRICS:")
    print(f"{'System':<12} {'Quality':<8} {'Novelty':<8} {'Diversity':<10} {'Speed(s)':<8} {'Total':<8}")
    print("-" * 60)
    
    for result in report.results:
        print(f"{result.system_name:<12} "
              f"{result.avg_quality:<8.3f} "
              f"{result.avg_novelty:<8.3f} "
              f"{result.diversity_score:<10.3f} "
              f"{result.generation_time:<8.1f} "
              f"{result.total_score:<8.3f}")
    
    # Category winners
    print(f"\n🎯 CATEGORY WINNERS:")
    print(f"  Quality:     {report.rankings['quality'][0]}")
    print(f"  Novelty:     {report.rankings['novelty'][0]}")
    print(f"  Diversity:   {report.rankings['diversity'][0]}")
    print(f"  Speed:       {report.rankings['speed'][0]}")
    print(f"  Consistency: {report.rankings['consistency'][0]}")
    
    # Statistical insights
    if "error" not in report.statistical_analysis:
        stats = report.statistical_analysis
        print(f"\n📈 STATISTICAL ANALYSIS:")
        print(f"  Score Range:    {stats['total_score']['range']:.3f}")
        print(f"  Quality Std:    {stats['quality']['std']:.3f}")
        print(f"  Speed Range:    {max(r.generation_time for r in report.results) - min(r.generation_time for r in report.results):.1f}s")
    
    # Sample jokes from top performers
    print(f"\n💎 SAMPLE JOKES FROM TOP PERFORMERS:")
    top_3 = report.rankings["overall"][:3]
    for i, system_name in enumerate(top_3, 1):
        result = next(r for r in report.results if r.system_name == system_name)
        print(f"\n  {i}. {system_name} (Score: {result.total_score:.3f}):")
        for j, joke in enumerate(result.jokes[:2], 1):  # Show first 2 jokes
            print(f"     {j}. {joke}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report.recommendations:
        print(f"  • {rec}")

def interactive_evaluation():
    """Interactive evaluation mode"""
    print("\n🔬 Core Joke Generation Systems Evaluation Suite")
    print("=" * 55)
    print("Evaluates: Basic, Pruning, Weighted, and Novelty generators")
    
    suite = SimplifiedEvaluationSuite()
    
    while True:
        print(f"\nAvailable options:")
        print("1. Compare all core systems on a topic")
        print("2. Evaluate specific system")
        print("3. Batch evaluation on multiple topics")
        print("4. View system information")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "5":
            print("Thanks for using the evaluation suite! 🔬")
            break
        
        elif choice == "1":
            topic = input("Enter topic for evaluation: ").strip()
            if not topic:
                print("Please enter a valid topic.")
                continue
            
            num_jokes = input("Number of jokes per system (default 5): ").strip()
            num_jokes = int(num_jokes) if num_jokes.isdigit() else 5
            
            print(f"\nEvaluating all core systems for '{topic}'...")
            report = suite.compare_all_systems(topic, num_jokes)
            
            display_detailed_report(report)
            
            save_choice = input("\nSave report? (y/n): ").strip().lower()
            if save_choice == 'y':
                suite.save_report(report)
                suite.export_csv(report)
        
        elif choice == "2":
            print(f"\nAvailable systems: {', '.join(suite.generators.keys())}")
            system = input("Choose system to evaluate: ").strip()
            
            if system not in suite.generators:
                print("Invalid system name.")
                continue
            
            topic = input("Enter topic: ").strip()
            if not topic:
                print("Please enter a valid topic.")
                continue
            
            result = suite.evaluate_system(system, topic)
            
            print(f"\n📋 {system} Evaluation Results for '{topic}':")
            print(f"  Total Score:    {result.total_score:.3f}")
            print(f"  Quality:        {result.avg_quality:.3f}")
            print(f"  Novelty:        {result.avg_novelty:.3f}")
            print(f"  Diversity:      {result.diversity_score:.3f}")
            print(f"  Generation Time: {result.generation_time:.1f}s")
            
            print(f"\n  Generated Jokes:")
            for i, joke in enumerate(result.jokes, 1):
                print(f"    {i}. {joke}")
        
        elif choice == "3":
            topics_input = input("Enter topics separated by commas: ").strip()
            topics = [t.strip() for t in topics_input.split(',') if t.strip()]
            
            if not topics:
                print("Please enter valid topics.")
                continue
            
            print(f"\nRunning batch evaluation on {len(topics)} topics...")
            
            all_reports = []
            for topic in topics:
                print(f"\n--- Evaluating topic: {topic} ---")
                report = suite.compare_all_systems(topic, 3)  # Fewer jokes for batch
                all_reports.append(report)
            
            # Summary analysis
            print(f"\n📊 BATCH EVALUATION SUMMARY:")
            print("=" * 40)
            
            system_wins = defaultdict(int)
            for report in all_reports:
                winner = report.rankings["overall"][0]
                system_wins[winner] += 1
            
            print("Overall winners by topic:")
            for system, wins in sorted(system_wins.items(), key=lambda x: x[1], reverse=True):
                print(f"  {system}: {wins} wins")
        
        elif choice == "4":
            print(f"\n📖 CORE SYSTEM INFORMATION:")
            print("=" * 35)
            
            systems_info = {
                "Basic": "Simple PlanSearch with ELO rating",
                "Pruning": "Quality-based pruning and filtering",
                "Weighted": "Multi-dimensional weighted ranking",
                "Novelty": "Advanced novelty detection"
            }
            
            for system, description in systems_info.items():
                print(f"  {system:<12}: {description}")
        
        else:
            print("Invalid choice. Please select 1-5.")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Command line mode
        topic = sys.argv[1]
        
        suite = SimplifiedEvaluationSuite()
        print(f"Running core systems evaluation for topic: '{topic}'")
        
        report = suite.compare_all_systems(topic)
        display_detailed_report(report)
        
        # Auto-save results
        suite.save_report(report)
        suite.export_csv(report)
    else:
        # Interactive mode
        interactive_evaluation()

if __name__ == "__main__":
    main()