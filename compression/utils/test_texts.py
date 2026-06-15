import os
import json


# Function to load large texts from files
def load_large_text(filename, max_chars=None):
    """Load text from file with optional character limit."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    file_path = os.path.join(data_dir, filename)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            if max_chars:
                return text[:max_chars]
            return text
    else:
        return f"[File {filename} not found]"


# Basic test texts
test_texts = {
    "Simple repetitive":
    "The cat sat on the mat. " * 50,
    "Your arithmetic coding example":
    "The forecast was for the worst to come, and the worst was expected for the city. "
    "The following day, the city of Wilmington moved 6,200 people from the Wilmington airport to Gatineau.",
    "Shakespeare (structured)":
    "To be or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles "
    "and, by opposing, end them. To die—to sleep, no more; and by a sleep to say we end "
    "the heart-ache and the thousand natural shocks that flesh is heir to: 'tis a consummation "
    "devoutly to be wish'd.",
    "Technical text":
    "import numpy as np\nfrom sklearn.model_selection import train_test_split\n"
    "def process_data(X, y):\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
    "    return X_train, X_test, y_train, y_test\n" * 10,
    "Random-like text":
    "The algorithm processes 47892 data points using methodology XQ-7734B with parameters "
    "alpha=0.00234, beta=0.91847, gamma=2.3049. Results indicate 67.3% efficiency improvement "
    "over baseline ZX-991A configuration in 84.7% of test cases." * 5,
    "Very short":
    "Hello world!",
    "Long repetitive":
    "Na " * 1000 + "Batman!"
}

# Large test texts loaded from files
large_test_texts = {
    "Bible (first 10k chars)":
    lambda: load_large_text("bible_kjv.txt", 10000),
    "Bible (first 50k chars)":
    lambda: load_large_text("bible_kjv.txt", 50000),
    "Bible (full)":
    lambda: load_large_text("bible_kjv.txt"),
    "Dictionary (first 10k chars)":
    lambda: load_large_text("dictionary.txt", 10000),
    "Dictionary (first 50k chars)":
    lambda: load_large_text("dictionary.txt", 50000),
    "Dictionary (full)":
    lambda: load_large_text("dictionary.txt"),
    "Generated 100k tokens (first 10k chars)":
    lambda: load_large_text("generated_100k.txt", 10000),
    "Generated 100k tokens (first 50k chars)":
    lambda: load_large_text("generated_100k.txt", 50000),
    "Generated 100k tokens (first 100k chars)":
    lambda: load_large_text("generated_100k.txt", 100000),
    "Generated 100k tokens (full)":
    lambda: load_large_text("generated_100k.txt"),
}

# Combined test texts (for backward compatibility)
all_test_texts = test_texts.copy()


# Function to get a test text (handles both regular and lazy-loaded texts)
def get_test_text(name):
    """Get a test text by name, loading from file if necessary."""
    if name in test_texts:
        return test_texts[name]
    elif name in large_test_texts:
        return large_test_texts[name]()
    else:
        raise KeyError(f"Test text '{name}' not found")


default_text = "The forecast was for the worst to come, and the worst was expected for the city. The following day, the city of Wilmington moved 6,200 people from the Wilmington airport to Gatineau."


def load_text_from_file(file_path: str, text_index: int = 0):
    """Load text and tokens from a generated text file.
    
    Args:
        file_path: Path to JSON file
        text_index: Which text to use if multiple (default: 0)
        
    Returns:
        Tuple of (text, tokens, metadata)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    metadata = {
        'source_file': file_path,
        'generating_model': data.get('model', 'unknown')
    }

    # Handle different file formats
    if 'texts' in data and data['texts']:
        # Format from generate_model_texts.py
        if text_index >= len(data['texts']):
            raise ValueError(
                f"Text index {text_index} out of range (file has {len(data['texts'])} texts)"
            )

        text_data = data['texts'][text_index]
        metadata['prompt'] = text_data.get('prompt', '')
        metadata['num_texts'] = len(data['texts'])
        metadata['selected_index'] = text_index

        if 'generated_tokens' in text_data:
            return text_data['generated_text'], text_data[
                'generated_tokens'], metadata
        else:
            # If no tokens, we'll need to tokenize
            return text_data['generated_text'], None, metadata
    elif 'generated_text' in data:
        # Simple format
        return data['generated_text'], data.get('generated_tokens',
                                                None), metadata
    else:
        # Plain text file
        return data if isinstance(data, str) else str(data), None, metadata
