"""
The 40 entities the RAG system must answer about (20 people + 20 places).

The first 10 of each are the assignment's required minimum set.
The remaining 10 of each are extras to give the retriever more diversity.
"""

PEOPLE = [
    # Assignment-required minimum
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # Extras (10)
    "Mustafa Kemal Atatürk",
    "Mahatma Gandhi",
    "Steve Jobs",
    "Charles Darwin",
    "Isaac Newton",
    "Pablo Picasso",
    "Vincent van Gogh",
    "Stephen Hawking",
    "Ludwig van Beethoven",
    "Wolfgang Amadeus Mozart",
]

PLACES = [
    # Assignment-required minimum
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Giza pyramid complex",  # canonical Wikipedia title for "Pyramids of Giza"
    "Mount Everest",
    # Extras (10)
    "Sagrada Família",
    "Stonehenge",
    "Petra",
    "Angkor Wat",
    "Acropolis of Athens",
    "Christ the Redeemer (statue)",
    "Big Ben",
    "Sydney Opera House",
    "Niagara Falls",
    "Cappadocia",
]


def all_entities() -> list[tuple[str, str]]:
    """Return [(title, type), ...] for all 40 entities."""
    return [(p, "person") for p in PEOPLE] + [(pl, "place") for pl in PLACES]
