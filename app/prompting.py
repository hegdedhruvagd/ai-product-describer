"""Shared prompt formatting for training examples and user descriptions."""


def format_product_prompt(product: dict, *, include_description: bool = False) -> str:
    ingredients = product.get("key_ingredients", [])
    if isinstance(ingredients, str):
        ingredients_text = ingredients
    else:
        ingredients_text = ", ".join(ingredients)
    lines = [
        f"Product: {product.get('product_name', '')}",
        f"Category: {product.get('category', '')}",
        f"Ingredients: {ingredients_text}",
        f"For: {product.get('target_skin_hair_type', '')}",
        f"Size: {product.get('size', '')}",
        "Description:",
    ]
    if include_description:
        lines[-1] += f" {product.get('description', '')}"
    return "\n".join(lines)
