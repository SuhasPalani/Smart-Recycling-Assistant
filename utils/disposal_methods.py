def determine_disposal(detected_object, material_insight):
    """
    Use a simple string search to analyze the material insight and recommend a disposal method.
    """
    disposal_methods = {
        "plastic": "Recycle it in a plastic recycling bin if accepted by local facilities.",
        "paper": "Recycle it in a paper recycling bin.",
        "food": "Compost it if organic and uncooked, otherwise dispose of it in landfill waste.",
        "metal": "Recycle it in a metal recycling bin.",
        "glass": "Recycle it in a glass recycling bin.",
    }
    
    # Check for keywords in the material insight
    for keyword in disposal_methods:
        if keyword in material_insight.lower():
            return disposal_methods[keyword]
    
    return "Unknown disposal method."
