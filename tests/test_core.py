"""Milestone tests. Remove a skip as the corresponding milestone is implemented."""
from collections import Counter
import re

import pytest

from app.core import CORPUS_PATH, load_product_corpus


# Milestone 1: Collect a small corpus of product descriptions for a specific category.
def test_milestone_1_dataset_exists_and_loads():
    assert CORPUS_PATH.is_file()
    records = load_product_corpus()
    assert records


def test_milestone_1_record_fields_are_complete():
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


def test_milestone_1_corpus_size_and_description_variety():
    records = load_product_corpus()
    descriptions = [record["description"] for record in records]
    assert 100 <= len(records) <= 150
    assert all(len(description.split()) >= 30 for description in descriptions)
    assert len(set(descriptions)) >= 100
    vocabulary = {word.lower() for description in descriptions for word in re.findall(r"[a-zA-Z]+", description)}
    assert len(vocabulary) >= 100


def test_milestone_1_categories_have_reasonable_coverage():
    distribution = Counter(record["category"] for record in load_product_corpus())
    expected_categories = {
        "Face cleanser", "Moisturiser", "Sunscreen", "Serum", "Face mask",
        "Body lotion", "Shampoo", "Conditioner", "Lip balm",
    }
    assert set(distribution) == expected_categories
    assert all(count >= 10 for count in distribution.values())

