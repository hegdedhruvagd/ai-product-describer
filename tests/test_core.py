"""Validates the checked-in product corpus and its training-ready records."""
from collections import Counter
import re

from app.core import CORPUS_PATH, load_product_corpus


def test_bundled_corpus_exists_and_loads():
    """The checked-in JSONL dataset exists and the shared loader returns records."""
    assert CORPUS_PATH.is_file()
    records = load_product_corpus()
    assert records


def test_every_product_record_has_required_nonempty_fields():
    """Every record includes the fields needed to condition model training."""
    required_fields = {
        "product_name", "category", "key_ingredients",
        "target_skin_hair_type", "size", "description",
    }
    for record in load_product_corpus():
        assert required_fields <= record.keys()
        assert all(isinstance(record[field], str) and record[field].strip()
                   for field in ("product_name", "category", "target_skin_hair_type", "size", "description"))
        assert isinstance(record["key_ingredients"], list)
        assert record["key_ingredients"]
        assert all(isinstance(ingredient, str) and ingredient.strip()
                   for ingredient in record["key_ingredients"])


def test_corpus_size_and_description_variety_are_suitable_for_training():
    """The corpus has the requested scale and non-empty, varied descriptions."""
    records = load_product_corpus()
    descriptions = [record["description"] for record in records]
    assert 100 <= len(records) <= 150
    assert all(len(description.split()) >= 30 for description in descriptions)
    assert len(set(descriptions)) >= 100
    vocabulary = {word.lower() for description in descriptions for word in re.findall(r"[a-zA-Z]+", description)}
    assert len(vocabulary) >= 100


def test_all_personal_care_categories_have_balanced_coverage():
    """All nine intended categories occur with enough examples for the small corpus."""
    distribution = Counter(record["category"] for record in load_product_corpus())
    expected_categories = {
        "Face cleanser", "Moisturiser", "Sunscreen", "Serum", "Face mask",
        "Body lotion", "Shampoo", "Conditioner", "Lip balm",
    }
    assert set(distribution) == expected_categories
    assert all(count >= 10 for count in distribution.values())

