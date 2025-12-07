import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Replace HTML entities like &amp; or &#123;
    text = re.sub(r'&(#?[\w\d]+);', ' ', text)

    # Tokenize (handles mixed Bangla-English properly)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        tokens = word_tokenize(text)

    clean_tokens = []
    for token in tokens:
        # Keep Bangla words
        if re.match(r'^[\u0980-\u09FF]+$', token):
            clean_tokens.append(token)
        # Keep English words (like "Python", "Engineer")
        elif re.match(r'^[A-Za-z]+$', token):
            clean_tokens.append(token)
        # Keep technical words like "C++", "C#", ".NET"
        elif re.match(r'^[A-Za-z0-9\+\#\.]+$', token):
            clean_tokens.append(token)

    # Join back
    return " ".join(clean_tokens)

def tokenize_whitespace_remove_special(text):
    if not isinstance(text, str):
        return ""

    # Split by whitespace
    tokens = text.split()

    # Remove tokens that consist entirely of special characters
    clean_tokens = [tok for tok in tokens if re.search(r'[A-Za-z0-9\u0980-\u09FF]', tok)]

    return " ".join(clean_tokens)

def log_normalize(x):
    """Apply logarithmic normalization to a single value."""
    if pd.isna(x) or x is None:
        return 0.0
    try:
        x = float(x)
        # Handle negative and zero values
        if x < 0:
            return 0.0
        elif x == 0:
            return 0.0
        else:
            return np.log1p(x)
    except (ValueError, TypeError):
        return 0.0
