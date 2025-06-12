#!/usr/bin/env python3
"""
Setup and Run Script for AI Joke Generation Research Project
Provides easy setup and execution of all joke generation variants
"""

import os
import sys
import subprocess
import time
from typing import Dict, List

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    requirements = [
        "openai>=1.0.0",
        "anthropic>=0.8.0", 
        "google-generativeai>=0.3.0",
        "numpy>=1.21.0"
    ]
    
    for requirement in requirements:
        try:
            print(f"Installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to install {requirement}: {e}")
    
    print("✅ Requirements installation completed")

def setup_api_keys():
    """Help user setup API keys"""
    print("\n🔑 API Key Setup")
    print("=" * 40)
    
    api_keys = {
        "OPENAI_API_KEY": "OpenAI GPT (Required for most features)",
        "ANTHROPIC_API_KEY": "Anthropic Claude (Optional, for multi-provider)",
        "GEMINI_API_KEY": "Google Gemini (Optional, for multi-provider)"
    }
    
    env_file_content = []
    
    for key, description in api_keys.items():
        current_value = os.environ.get(key)
        if current_value:
            print(f"✅ {key}: Already set")
            env_file_content.append(f"{key}={current_value}")
        else:
            print(f"\n{description}")
            value = input(f"Enter {key} (or press Enter to skip): ").strip()
            if value:
                os.environ[key] = value
                env_file_content.append(f"{key}={value}")
                print(f"✅ {key}: Set")
            else:
                print(f"⏭️  {key}: Skipped")
    
    # Create .env file
    if env_file_content:
        with open('.env', 'w') as f:
            f.write('\n'.join(env_file_content))
        print("\n📄 Created .env file with your API keys")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OpenAI API key is required for most functionality")
        return False
    
    return True

def display_menu():
    """Display main menu"""
    print("\n🎭 AI Joke Generation Research Project")
    print("=" * 50)
    print("Choose a variant to run:")
    print()
    print("1. Basic Generator        - Simple PlanSearch + ELO")
    print("2. Pruning Generator      - Quality-based pruning")
    print("3. Weighted Generator     - Multi-dimensional ranking")
    print("4. Novelty Generator      - Advanced novelty detection")
    print("5. Multi-Provider         - Cross-model validation")
    print("6. Self-Improving         - Adaptive learning")
    print("7. Evaluation Suite       - Compare all approaches")
    print()
    print("8. Demo All Systems       - Quick demonstration")
    print("9. Research Analysis      - Detailed comparison")
    print("10. Exit")
    print()

def run_generator(choice: str, topic: str = None):
    """Run selected generator"""
    
    generators = {
        "1": ("basic_joke_generator.py", "Basic Generator"),
        "2": ("pruning_joke_generator.py", "Pruning Generator"),
        "3": ("weighted_joke_generator.py", "Weighted Generator"),
        "4": ("novelty_joke_generator.py", "Novelty Generator"),
        "5": ("multi_provider_generator.py", "Multi-Provider Generator"),
        "6": ("self_improving_generator.py", "Self-Improving Generator"),
        "7": ("evaluation_suite.py", "Evaluation Suite")
    }
    
    if choice not in generators:
        print("❌ Invalid choice")
        return
    
    filename, name = generators[choice]
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found")
        print("Please ensure all generator files are in the current directory")
        return
    
    print(f"\n🚀 Running {name}...")
    print("-" * 40)
    
    try:
        if topic:
            subprocess.run([sys.executable, filename, topic])
        else:
            subprocess.run([sys.executable, filename, "interactive"])
    except KeyboardInterrupt:
        print("\n\n⏹️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running {name}: {e}")

def demo_all_systems():
    """Run quick demo of all systems"""
    demo_topic = input("Enter topic for demo (default: 'artificial intelligence'): ").strip()
    if not demo_topic:
        demo_topic = "artificial intelligence"
    
    print(f"\n🎪 Running demo for all systems with topic: '{demo_topic}'")
    print("=" * 60)
    
    generators = [
        ("basic_joke_generator.py", "Basic"),
        ("pruning_joke_generator.py", "Pruning"),
        ("weighted_joke_generator.py", "Weighted"),
        ("novelty_joke_generator.py", "Novelty"),
        ("multi_provider_generator.py", "Multi-Provider"),
        ("self_improving_generator.py", "Self-Improving")
    ]
    
    for filename, name in generators:
        if os.path.exists(filename):
            print(f"\n--- {name} Generator ---")
            try:
                subprocess.run([sys.executable, filename, demo_topic], timeout=120)
                time.sleep(2)  # Brief pause between generators
            except subprocess.TimeoutExpired:
                print(f"⏰ {name} generator timed out")
            except Exception as e:
                print(f"❌ Error with {name}: {e}")
        else:
            print(f"⏭️  Skipping {name} (file not found)")
    
    print("\n🏁 Demo completed!")

def research_analysis():
    """Run comprehensive research analysis"""
    print("\n🔬 Research Analysis Mode")
    print("=" * 40)
    
    topics = input("Enter topics for analysis (comma-separated): ").strip()
    if not topics:
        topics = "programming,cooking,cats,space exploration,coffee"
    
    topic_list = [t.strip() for t in topics.split(',')]
    
    print(f"\nRunning comprehensive analysis on {len(topic_list)} topics...")
    print("This will take several minutes and generate detailed reports.")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    if not os.path.exists("evaluation_suite.py"):
        print("❌ Evaluation suite not found")
        return
    
    # Run batch evaluation
    for topic in topic_list:
        print(f"\n--- Analyzing: {topic} ---")
        try:
            subprocess.run([sys.executable, "evaluation_suite.py", topic], timeout=300)
        except subprocess.TimeoutExpired:
            print(f"⏰ Analysis timed out for {topic}")
        except Exception as e:
            print(f"❌ Error analyzing {topic}: {e}")
    
    print("\n📊 Research analysis completed!")
    print("Check generated JSON and CSV files for detailed results.")

def check_files():
    """Check if all required files are present"""
    required_files = [
        "basic_joke_generator.py",
        "pruning_joke_generator.py", 
        "weighted_joke_generator.py",
        "novelty_joke_generator.py",
        "multi_provider_generator.py",
        "self_improving_generator.py",
        "evaluation_suite.py"
    ]
    
    missing_files = []
    for filename in required_files:
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        print("⚠️  Missing files:")
        for filename in missing_files:
            print(f"   - {filename}")
        return False
    
    print("✅ All generator files found")
    return True

def create_sample_files():
    """Create sample configuration files"""
    
    # Create sample .env file
    env_sample = """# API Keys for Joke Generation
# Get OpenAI API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_key_here

# Optional: Get Anthropic API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Get Gemini API key from: https://makersuite.google.com/app/apikey  
GEMINI_API_KEY=your_gemini_key_here
"""
    
    with open('.env.sample', 'w') as f:
        f.write(env_sample)
    
    # Create README
    readme = """# AI Joke Generation Research Project

This project implements and compares multiple approaches to AI-powered joke generation.

## Quick Start

1. Install Python 3.8+
2. Run: `python setup_and_run.py`
3. Follow the setup prompts
4. Choose a generator to test

## Generators Available

- **Basic**: Simple PlanSearch with ELO rating
- **Pruning**: Quality-based observation and joke filtering  
- **Weighted**: Multi-dimensional ranking (humor + diversity + novelty)
- **Novelty**: Advanced pattern detection and memorization prevention
- **Multi-Provider**: Cross-validation with multiple LLM providers
- **Self-Improving**: Adaptive learning with feedback loops

## Research Focus

This project addresses key questions in computational humor:
- How to generate truly novel vs. memorized jokes
- Multi-objective optimization of humor dimensions
- LLM-as-a-judge bias mitigation
- Cross-model consistency validation

## Files

- `*_joke_generator.py`: Individual generator implementations
- `evaluation_suite.py`: Comprehensive comparison framework
- `setup_and_run.py`: Easy setup and execution

## API Keys Required

At minimum, you need an OpenAI API key. Anthropic and Gemini keys are optional for multi-provider testing.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print("📄 Created sample configuration files:")
    print("   - .env.sample (copy to .env and add your API keys)")
    print("   - README.md")

def main():
    """Main execution function"""
    print("🎭 AI Joke Generation Research Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if files exist
    files_exist = check_files()
    
    # Setup phase
    setup_complete = False
    if not files_exist:
        print("\n⚠️  Some generator files are missing.")
        print("Please ensure all .py files are in the current directory.")
        create_sample_files()
        return
    
    # Install requirements
    install_choice = input("\nInstall/update requirements? (y/n): ").strip().lower()
    if install_choice == 'y':
        install_requirements()
    
    # Setup API keys
    api_setup = input("\nSetup API keys? (y/n): ").strip().lower()
    if api_setup == 'y':
        setup_complete = setup_api_keys()
    else:
        setup_complete = bool(os.environ.get("OPENAI_API_KEY"))
    
    if not setup_complete:
        print("\n⚠️  Setup incomplete. Some features may not work without API keys.")
        continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
        if continue_anyway != 'y':
            return
    
    create_sample_files()
    
    # Main loop
    while True:
        display_menu()
        choice = input("Select option (1-10): ").strip()
        
        if choice == "10":
            print("\n👋 Thanks for exploring AI joke generation research!")
            break
        
        elif choice in ["1", "2", "3", "4", "5", "6", "7"]:
            topic = input("Enter topic (or press Enter for interactive mode): ").strip()
            run_generator(choice, topic if topic else None)
        
        elif choice == "8":
            demo_all_systems()
        
        elif choice == "9":
            research_analysis()
        
        else:
            print("❌ Invalid choice. Please select 1-10.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()