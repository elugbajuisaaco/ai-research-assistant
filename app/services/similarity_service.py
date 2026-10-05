import math

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    if not vector_a or not vector_b:
        raise ValueError("Vectors cannot be empty.")
        
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have the same dimension. Got {len(vector_a)} and {len(vector_b)}."
        )
        
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
        
    return float(dot_product / (magnitude_a * magnitude_b))
