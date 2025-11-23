"""
Data Generation Script for PII NER
Generates realistic noisy STT transcripts with PII entities
"""

import json
import random
from typing import List, Dict, Tuple

# Seed for reproducibility
random.seed(42)


# ==== DATA SOURCES ====

FIRST_NAMES = [
    "ramesh", "priya", "amit", "sneha", "rajesh", "anita", "vijay", "kavita",
    "suresh", "meera", "arun", "pooja", "mahesh", "nisha", "deepak", "ritu",
    "rahul", "sunita", "kiran", "anjali", "rohan", "neha", "sanjay", "rekha",
    "manoj", "swati", "vinod", "sapna", "prakash", "divya", "ashok", "geeta",
    "john", "sarah", "michael", "emma", "david", "olivia", "james", "sophia",
    "robert", "isabella", "william", "mia", "richard", "charlotte", "thomas", "amelia"
]

LAST_NAMES = [
    "sharma", "verma", "gupta", "kumar", "singh", "patel", "mehta", "reddy",
    "krishnan", "nair", "iyer", "rao", "das", "shah", "joshi", "desai",
    "agarwal", "chopra", "malhotra", "kapoor", "bose", "sen", "menon", "pillai",
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis",
    "rodriguez", "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas"
]

CITIES = [
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad",
    "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal", "visakhapatnam", "patna",
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia", "san antonio",
    "london", "paris", "tokyo", "sydney", "dubai", "singapore", "toronto", "berlin"
]

LOCATIONS = [
    "park street", "mg road", "nehru nagar", "gandhi avenue", "railway station",
    "central square", "market area", "residency road", "lake view", "hill station",
    "airport terminal", "bus stand", "shopping mall", "cyber park", "tech park",
    "main street", "first avenue", "second street", "oak road", "pine boulevard"
]

EMAIL_DOMAINS = [
    "gmail dot com", "yahoo dot com", "outlook dot com", "hotmail dot com",
    "rediffmail dot com", "protonmail dot com", "icloud dot com", "aol dot com"
]

# Patterns for noisy STT


def generate_credit_card() -> Tuple[str, str]:
    """Generate a credit card number in noisy STT format"""
    patterns = [
        # Full digits
        lambda: f"{random.randint(4000, 5999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}",
        # Spelled out
        lambda: " ".join([str(random.randint(0, 9)) for _ in range(16)]),
        # Mixed
        lambda: f"four {random.randint(100, 999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(100, 999)}",
        # With words
        lambda: f"{random.randint(4000, 5999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} double {random.randint(0, 9)} {random.randint(0, 9)}",
    ]
    text = random.choice(patterns)()
    return text, text


def generate_phone() -> Tuple[str, str]:
    """Generate a phone number in noisy STT format"""
    patterns = [
        # 10 digits continuous
        lambda: "".join([str(random.randint(0, 9)) for _ in range(10)]),
        # 10 digits with spaces
        lambda: " ".join([str(random.randint(0, 9)) for _ in range(10)]),
        # Grouped format (3-3-4)
        lambda: f"{random.randint(700, 999)} {random.randint(100, 999)} {random.randint(1000, 9999)}",
        # Grouped format (4-3-3)
        lambda: f"{random.randint(7000, 9999)} {random.randint(100, 999)} {random.randint(100, 999)}",
        # Starting with common prefixes
        lambda: f"nine {' '.join([str(random.randint(0, 9)) for _ in range(9)])}",
        lambda: f"eight {' '.join([str(random.randint(0, 9)) for _ in range(9)])}",
        lambda: f"seven {' '.join([str(random.randint(0, 9)) for _ in range(9)])}",
        # Mixed continuous and spaced
        lambda: f"{random.randint(90, 99)}{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)}",
        # Double/triple patterns
        lambda: f"double {random.randint(0, 9)} {' '.join([str(random.randint(0, 9)) for _ in range(8)])}",
        lambda: f"{random.randint(0, 9)} {random.randint(0, 9)} triple {random.randint(0, 9)} {' '.join([str(random.randint(0, 9)) for _ in range(4)])}",
    ]
    text = random.choice(patterns)()
    return text, text


def generate_email() -> Tuple[str, str]:
    """Generate an email in noisy STT format"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    domain = random.choice(EMAIL_DOMAINS)
    
    # More explicit email patterns with "at" and "dot" keywords
    patterns = [
        f"{first} dot {last} at {domain}",
        f"{first}{random.randint(10, 99)} at {domain}",
        f"{first} underscore {last} at {domain}",
        f"{first} {last} at {domain}",
        f"{first} dot {last[:1]} at {domain}",
        f"{first}{last} at {domain}",
    ]
    
    text = random.choice(patterns)
    return text, text


def generate_person_name() -> Tuple[str, str]:
    """Generate a person name"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    
    patterns = [
        f"{first} {last}",
        f"{first}",
        f"{last}",
        f"{first} {last[0]}",
    ]
    
    text = random.choice(patterns)
    return text, text


def generate_date() -> Tuple[str, str]:
    """Generate a date in noisy STT format"""
    day = random.randint(1, 28)
    month = random.choice(["january", "february", "march", "april", "may", "june", 
                           "july", "august", "september", "october", "november", "december"])
    year = random.randint(2020, 2025)
    
    patterns = [
        f"{day:02d} {month} {year}",
        f"{month} {day}",
        f"{day} {month}",
        f"{day:02d} {month[:3]} {year}",
        lambda: f"{random.randint(1, 28):02d} {random.randint(1, 12):02d} {year}",
    ]
    
    pattern = random.choice(patterns)
    text = pattern() if callable(pattern) else pattern
    return text, text


def generate_city() -> Tuple[str, str]:
    """Generate a city name"""
    city = random.choice(CITIES)
    return city, city


def generate_location() -> Tuple[str, str]:
    """Generate a location"""
    location = random.choice(LOCATIONS)
    return location, location


# ==== SENTENCE TEMPLATES ====

TEMPLATES = [
    # Credit card templates
    "my credit card number is {CREDIT_CARD}",
    "card number is {CREDIT_CARD}",
    "the card is {CREDIT_CARD}",
    "please charge {CREDIT_CARD}",
    "my card details are {CREDIT_CARD}",
    
    # Phone templates
    "call me on {PHONE}",
    "my number is {PHONE}",
    "reach me at {PHONE}",
    "phone number {PHONE}",
    "you can call {PHONE}",
    "contact number is {PHONE}",
    
    # Email templates
    "email me at {EMAIL}",
    "my email is {EMAIL}",
    "send it to {EMAIL}",
    "email id is {EMAIL}",
    "reach me at {EMAIL}",
    
    # Person name templates
    "my name is {PERSON_NAME}",
    "i am {PERSON_NAME}",
    "this is {PERSON_NAME}",
    "speaking to {PERSON_NAME}",
    "talk to {PERSON_NAME}",
    
    # Date templates
    "on {DATE}",
    "the date is {DATE}",
    "scheduled for {DATE}",
    "travelling on {DATE}",
    "meeting on {DATE}",
    
    # City templates
    "i live in {CITY}",
    "calling from {CITY}",
    "i am in {CITY}",
    "located in {CITY}",
    "travelling to {CITY}",
    
    # Location templates
    "near {LOCATION}",
    "at {LOCATION}",
    "close to {LOCATION}",
    "meet at {LOCATION}",
    
    # Multi-entity templates
    "my name is {PERSON_NAME} and my email is {EMAIL}",
    "i am {PERSON_NAME} calling from {CITY}",
    "{PERSON_NAME} at {EMAIL} phone {PHONE}",
    "card {CREDIT_CARD} name {PERSON_NAME}",
    "email {EMAIL} number {PHONE}",
    "{PERSON_NAME} from {CITY} number is {PHONE}",
    "i am {PERSON_NAME} my number is {PHONE} and email is {EMAIL}",
    "travelling to {CITY} on {DATE}",
    "call {PERSON_NAME} at {PHONE}",
    "send email to {PERSON_NAME} at {EMAIL}",
    "meeting {PERSON_NAME} in {CITY} on {DATE}",
    "my details are name {PERSON_NAME} email {EMAIL} phone {PHONE}",
    "card number is {CREDIT_CARD} and email is {EMAIL}",
    "i will be in {CITY} on {DATE} call me on {PHONE}",
    "{PERSON_NAME} staying near {LOCATION} in {CITY}",
]


def generate_example(example_id: int) -> Dict:
    """Generate a single training example"""
    
    # Select a random template
    template = random.choice(TEMPLATES)
    
    # Find all entity types in template
    entity_types = []
    for entity_type in ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"]:
        if f"{{{entity_type}}}" in template:
            entity_types.append(entity_type)
    
    # Generate entity values
    generators = {
        "CREDIT_CARD": generate_credit_card,
        "PHONE": generate_phone,
        "EMAIL": generate_email,
        "PERSON_NAME": generate_person_name,
        "DATE": generate_date,
        "CITY": generate_city,
        "LOCATION": generate_location,
    }
    
    text = template
    entities = []
    
    for entity_type in entity_types:
        entity_text, _ = generators[entity_type]()
        
        # Find position before replacement
        placeholder = f"{{{entity_type}}}"
        start = text.find(placeholder)
        
        # Replace placeholder
        text = text.replace(placeholder, entity_text, 1)
        
        # Calculate end position
        end = start + len(entity_text)
        
        entities.append({
            "start": start,
            "end": end,
            "label": entity_type
        })
    
    return {
        "id": f"utt_{example_id:04d}",
        "text": text,
        "entities": sorted(entities, key=lambda x: x["start"])
    }


def generate_dataset(num_examples: int, start_id: int = 1) -> List[Dict]:
    """Generate a dataset with specified number of examples"""
    dataset = []
    for i in range(num_examples):
        example = generate_example(start_id + i)
        dataset.append(example)
    return dataset


def save_dataset(dataset: List[Dict], filepath: str):
    """Save dataset to JSONL file"""
    with open(filepath, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(dataset)} examples to {filepath}")


# ==== MAIN ====

if __name__ == "__main__":
    print("Generating PII NER datasets...\n")
    
    # Generate training data (800 examples)
    print("Generating training data...")
    train_data = generate_dataset(num_examples=800, start_id=1)
    save_dataset(train_data, "data/train.jsonl")
    
    # Generate dev data (150 examples)
    print("Generating dev data...")
    dev_data = generate_dataset(num_examples=150, start_id=5001)
    save_dataset(dev_data, "data/dev.jsonl")
    
    print("\nData generation complete!")
    print(f"   - Training examples: {len(train_data)}")
    print(f"   - Dev examples: {len(dev_data)}")
    print("\nSample examples:")
    print(json.dumps(train_data[0], indent=2))
    print(json.dumps(train_data[1], indent=2))

