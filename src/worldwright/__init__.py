__version__ = "0.0.1"

# Auto-load .env from cwd if present. No-op if python-dotenv isn't installed
# or if there's no .env file. Keeps ANTHROPIC_API_KEY easy to manage in dev.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv()
except ImportError:
    pass
