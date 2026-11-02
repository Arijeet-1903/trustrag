"""
Loads secrets and settings from the .env file.
Import from here instead of typing API keys/passwords directly in your scripts.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the project root

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def check_config():
    """Quick sanity check that your .env is set up correctly."""
    missing = [
        name for name, val in {
            "NEO4J_URI": NEO4J_URI,
            "NEO4J_USERNAME": NEO4J_USERNAME,
            "NEO4J_PASSWORD": NEO4J_PASSWORD,
            "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        }.items() if not val
    ]
    if missing:
        print(f"Missing from your .env file: {', '.join(missing)}")
    else:
        print("All required settings are present.")


if __name__ == "__main__":
    check_config()
