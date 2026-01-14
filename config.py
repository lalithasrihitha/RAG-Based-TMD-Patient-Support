from dataclasses import dataclass

"""
https://medlineplus.gov/temporomandibulardisorders.html
"""

@dataclass(frozen=True)
class RAGConfig:
    ARTICLE: str = "temporomandibular-disorders"
    COLLECTION: str = "temporomandibular_disorders"

CONFIG = RAGConfig()

