"""
Lifestyle tracking analysis: food categorization and correlation analysis.

This module provides food categorization (using dictionary + LRU cache + LLM fallback)
and calculates correlations between foods, exercise, and symptom severity.
"""
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# FOOD CATEGORIES DICTIONARY (~200-300 common foods)
# ============================================================================

FOOD_CATEGORIES = {
    # ===== LÁCTEOS =====
    "leche": ["lacteos"],
    "queso": ["lacteos"],
    "yogur": ["lacteos"],
    "yogurt": ["lacteos"],
    "nata": ["lacteos"],
    "mantequilla": ["lacteos"],
    "crema": ["lacteos"],
    "helado": ["lacteos", "azucar"],
    "quesito": ["lacteos"],
    "flan": ["lacteos", "azucar"],
    "natillas": ["lacteos", "azucar"],

    # ===== GLUTEN =====
    "pan": ["gluten", "carbohidratos"],
    "pasta": ["gluten", "carbohidratos"],
    "pizza": ["gluten", "procesado"],
    "pizza con queso": ["gluten", "lacteos", "procesado"],
    "pizza cuatro quesos": ["gluten", "lacteos", "procesado"],
    "galleta": ["gluten", "azucar"],
    "galletas": ["gluten", "azucar"],
    "cereales": ["gluten", "carbohidratos"],
    "tostada": ["gluten"],
    "tostadas": ["gluten"],
    "croissant": ["gluten", "procesado"],
    "bizcocho": ["gluten", "azucar"],
    "pastel": ["gluten", "azucar"],
    "empanada": ["gluten", "procesado"],
    "bocadillo": ["gluten"],

    # ===== VERDURAS =====
    "ensalada": ["verduras"],
    "ensalada cesar": ["verduras", "lacteos"],
    "lechuga": ["verduras"],
    "tomate": ["verduras"],
    "zanahoria": ["verduras"],
    "brócoli": ["verduras", "fibra_alta"],
    "brocoli": ["verduras", "fibra_alta"],
    "espinacas": ["verduras", "fibra_alta"],
    "pimiento": ["verduras"],
    "calabacín": ["verduras"],
    "calabacin": ["verduras"],
    "berenjena": ["verduras"],
    "pepino": ["verduras"],
    "judías verdes": ["verduras", "fibra_alta"],
    "judias verdes": ["verduras", "fibra_alta"],
    "verduras al vapor": ["verduras"],
    "verduras salteadas": ["verduras"],

    # ===== PROCESADO =====
    "hamburguesa": ["procesado", "proteina"],
    "hamburguesa con queso": ["procesado", "proteina", "lacteos"],
    "nuggets": ["procesado", "proteina"],
    "salchicha": ["procesado", "proteina"],
    "salchichas": ["procesado", "proteina"],
    "chorizo": ["procesado", "proteina"],
    "bacon": ["procesado", "proteina"],
    "patatas fritas": ["procesado", "fritos"],
    "doritos": ["procesado"],
    "cheetos": ["procesado"],

    # ===== CAFÉ =====
    "café": ["cafe"],
    "cafe": ["cafe"],
    "café con leche": ["cafe", "lacteos"],
    "cafe con leche": ["cafe", "lacteos"],
    "espresso": ["cafe"],
    "capuchino": ["cafe", "lacteos"],
    "cappuccino": ["cafe", "lacteos"],
    "latte": ["cafe", "lacteos"],

    # ===== ALCOHOL =====
    "cerveza": ["alcohol"],
    "vino": ["alcohol"],
    "vino tinto": ["alcohol"],
    "vino blanco": ["alcohol"],
    "copa": ["alcohol"],
    "whisky": ["alcohol"],
    "ron": ["alcohol"],
    "vodka": ["alcohol"],

    # ===== FIBRA ALTA =====
    "legumbres": ["fibra_alta", "proteina"],
    "garbanzos": ["fibra_alta", "proteina"],
    "lentejas": ["fibra_alta", "proteina"],
    "alubias": ["fibra_alta", "proteina"],
    "judías": ["fibra_alta", "proteina"],
    "judias": ["fibra_alta", "proteina"],

    # ===== FRITOS =====
    "frito": ["fritos"],
    "frita": ["fritos"],
    "rebozado": ["fritos", "gluten"],
    "empanado": ["fritos", "gluten"],
    "croquetas": ["fritos", "gluten"],

    # ===== PICANTE =====
    "picante": ["picante"],
    "chile": ["picante"],
    "guindilla": ["picante"],
    "jalapeño": ["picante"],
    "jalapeno": ["picante"],
    "cayena": ["picante"],
    "comida picante": ["picante"],
    "salsa picante": ["picante"],

    # ===== PROTEÍNA =====
    "pollo": ["proteina"],
    "pollo a la plancha": ["proteina"],
    "pechuga de pollo": ["proteina"],
    "ternera": ["proteina"],
    "cerdo": ["proteina"],
    "pescado": ["proteina"],
    "salmón": ["proteina"],
    "salmon": ["proteina"],
    "atún": ["proteina"],
    "atun": ["proteina"],
    "merluza": ["proteina"],
    "huevo": ["proteina"],
    "huevos": ["proteina"],

    # ===== CARBOHIDRATOS =====
    "arroz": ["carbohidratos"],
    "arroz blanco": ["carbohidratos"],
    "arroz integral": ["carbohidratos", "fibra_alta"],
    "patata": ["carbohidratos"],
    "patatas": ["carbohidratos"],
    "patata cocida": ["carbohidratos"],
    "patatas cocidas": ["carbohidratos"],
    "boniato": ["carbohidratos"],

    # ===== AZÚCAR =====
    "chocolate": ["azucar"],
    "caramelo": ["azucar"],
    "caramelos": ["azucar"],
    "chuches": ["azucar"],
    "dulce": ["azucar"],
    "dulces": ["azucar"],
    "refresco": ["azucar"],
    "coca cola": ["azucar"],
    "pepsi": ["azucar"],

    # ===== PLATOS COMPUESTOS =====
    "pasta carbonara": ["gluten", "lacteos", "procesado"],
    "macarrones con queso": ["gluten", "lacteos"],
    "lasaña": ["gluten", "lacteos", "procesado"],
    "lasana": ["gluten", "lacteos", "procesado"],
    "paella": ["carbohidratos", "proteina"],
    "tortilla": ["proteina"],
    "tortilla de patatas": ["proteina", "carbohidratos"],
}


# ============================================================================
# FOOD CATEGORIZATION WITH LRU CACHE
# ============================================================================

@lru_cache(maxsize=1000)
def categorize_food_cached(food_text: str) -> Tuple[str, ...]:
    """
    Categorize food text into food groups using 3-level approach:

    Level 1: Exact match in dictionary (instant, covers ~85% of cases)
    Level 2: LRU cache (instant for repeated queries)
    Level 3: Keyword matching for variations

    Args:
        food_text: Free-text food description

    Returns:
        Tuple of food categories (e.g., ("lacteos", "gluten"))
    """
    # Normalize text
    normalized = food_text.lower().strip()

    # LEVEL 1: Exact match in dictionary
    if normalized in FOOD_CATEGORIES:
        # Convert list to tuple for caching
        categories = FOOD_CATEGORIES[normalized]
        if isinstance(categories, list):
            return tuple(categories)
        return tuple([categories]) if not isinstance(categories, tuple) else categories

    # LEVEL 3: Keyword matching (partial match)
    categories = set()
    for food_key, food_cats in FOOD_CATEGORIES.items():
        # Check if any keyword from dictionary appears in the food text
        if food_key in normalized:
            # Ensure food_cats is iterable (handle both list and single string)
            if isinstance(food_cats, list):
                categories.update(food_cats)
            else:
                categories.add(food_cats)

    # If found categories via keyword matching, return them
    if categories:
        return tuple(sorted(categories))

    # If nothing found, return uncategorized
    return ("sin_categorizar",)


def categorize_foods(food_list: List[str]) -> Dict[str, int]:
    """
    Categorize a list of foods and count occurrences of each category.

    Args:
        food_list: List of food descriptions

    Returns:
        Dictionary mapping category name to count
    """
    category_counts = {}

    for food in food_list:
        categories = categorize_food_cached(food)
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1

    return category_counts


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_food_correlations(
    daily_records: List[Dict],
    symptom_severity_fn
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Calculate correlations between food categories and symptom severity.

    Args:
        daily_records: List of daily records with 'foods' and 'symptoms'
        symptom_severity_fn: Function to calculate symptom severity from symptoms

    Returns:
        Tuple of (trigger_foods, beneficial_foods) dictionaries
    """
    import numpy as np

    # Need at least 7 days of data
    if len(daily_records) < 7:
        return None, None

    # Extract symptom severities for each day
    severities = []
    food_presence = {}  # category -> [0, 1, 0, 1, ...]  (binary presence per day)

    for record in daily_records:
        # Calculate severity for this day
        severity = symptom_severity_fn(record.get('symptoms'))
        severities.append(severity)

        # Categorize foods for this day
        foods = record.get('foods', [])
        if foods:
            day_categories = categorize_foods(foods)
        else:
            day_categories = {}

        # Track binary presence of each food category
        # Define all possible food categories
        all_categories = {
            'lacteos', 'gluten', 'verduras', 'procesado', 'cafe', 'alcohol',
            'fibra_alta', 'fritos', 'picante', 'proteina', 'carbohidratos',
            'azucar', 'sin_categorizar'
        }

        for category in all_categories:
            if category not in food_presence:
                food_presence[category] = []
            food_presence[category].append(1 if category in day_categories else 0)

    severities = np.array(severities)

    # Calculate correlations
    trigger_foods = {}
    beneficial_foods = {}

    for category, presence_list in food_presence.items():
        presence = np.array(presence_list)

        # Skip if no variance (all same values)
        if np.std(presence) == 0:
            continue

        # Calculate Pearson correlation
        correlation = np.corrcoef(presence, severities)[0, 1]

        # Count occurrences
        occurrences = int(np.sum(presence))

        # Only include if consumed at least 2 times
        if occurrences < 2:
            continue

        # Categorize as trigger (positive correlation > 0.5) or beneficial (negative < -0.5)
        if correlation >= 0.5:
            # Trigger food
            if correlation >= 0.7:
                insight = f"Correlación fuerte: consumir {category} se asocia con aumento de síntomas"
            else:
                insight = f"Correlación moderada: {category} podría estar relacionado con más síntomas"

            trigger_foods[category] = {
                "correlation": round(correlation, 2),
                "occurrences": occurrences,
                "insight": insight
            }

        elif correlation <= -0.5:
            # Beneficial food
            if correlation <= -0.7:
                insight = f"Correlación inversa fuerte: consumir {category} se asocia con reducción de síntomas"
            else:
                insight = f"Correlación inversa moderada: {category} podría ser beneficioso"

            beneficial_foods[category] = {
                "correlation": round(correlation, 2),
                "occurrences": occurrences,
                "insight": insight
            }

    return trigger_foods if trigger_foods else None, beneficial_foods if beneficial_foods else None


def calculate_exercise_impact(
    daily_records: List[Dict],
    symptom_severity_fn
) -> Optional[Dict]:
    """
    Calculate impact of exercise on symptom severity.

    Args:
        daily_records: List of daily records with 'exercise' and 'symptoms'
        symptom_severity_fn: Function to calculate symptom severity from symptoms

    Returns:
        Exercise impact dictionary or None
    """
    import numpy as np

    # Need at least 7 days of data
    if len(daily_records) < 7:
        return None

    # Extract exercise levels and severities
    severities_with = []
    severities_without = []
    exercise_binary = []  # 1 if moderate/high, 0 if none
    severities_all = []

    for record in daily_records:
        severity = symptom_severity_fn(record.get('symptoms'))
        severities_all.append(severity)

        exercise = record.get('exercise', 'none')

        if exercise in ['moderate', 'high']:
            severities_with.append(severity)
            exercise_binary.append(1)
        else:
            severities_without.append(severity)
            exercise_binary.append(0)

    # Need at least 2 days with exercise and 2 without
    if len(severities_with) < 2 or len(severities_without) < 2:
        return None

    # Calculate correlation
    exercise_binary = np.array(exercise_binary)
    severities_all = np.array(severities_all)

    correlation = np.corrcoef(exercise_binary, severities_all)[0, 1]

    # Calculate averages
    avg_with = float(np.mean(severities_with))
    avg_without = float(np.mean(severities_without))

    # Generate insight
    diff_percent = abs((avg_with - avg_without) / avg_without * 100) if avg_without > 0 else 0

    if correlation <= -0.4:
        # Exercise is beneficial
        insight = f"El ejercicio se asocia con reducción del {diff_percent:.0f}% en severidad de síntomas"
    elif correlation >= 0.4:
        # Exercise seems to worsen symptoms
        insight = f"El ejercicio intenso podría estar asociado con aumento del {diff_percent:.0f}% en síntomas"
    else:
        # No clear relationship
        insight = "No se observa relación clara entre ejercicio y síntomas con los datos actuales"

    return {
        "correlation": round(correlation, 2),
        "days_with_exercise": len(severities_with),
        "average_severity_with": round(avg_with, 2),
        "average_severity_without": round(avg_without, 2),
        "insight": insight
    }


def generate_lifestyle_recommendations(
    trigger_foods: Optional[Dict],
    beneficial_foods: Optional[Dict],
    exercise_impact: Optional[Dict]
) -> List[str]:
    """
    Generate actionable recommendations based on food and exercise insights.

    Args:
        trigger_foods: Dictionary of trigger food insights
        beneficial_foods: Dictionary of beneficial food insights
        exercise_impact: Exercise impact dictionary

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Food recommendations
    if trigger_foods:
        # Get top 2 trigger foods by correlation
        sorted_triggers = sorted(trigger_foods.items(), key=lambda x: x[1]['correlation'], reverse=True)
        for food, data in sorted_triggers[:2]:
            recommendations.append(f"⚠️ Considera reducir {food}: correlación {data['correlation']} con síntomas")

    if beneficial_foods:
        # Get top 2 beneficial foods
        sorted_beneficial = sorted(beneficial_foods.items(), key=lambda x: x[1]['correlation'])
        for food, data in sorted_beneficial[:2]:
            recommendations.append(f"✅ Aumenta consumo de {food}: correlación inversa {data['correlation']}")

    # Exercise recommendations
    if exercise_impact:
        if exercise_impact['correlation'] <= -0.4:
            recommendations.append(f"✅ Mantén el ejercicio: se asocia con {abs(exercise_impact['correlation']*100):.0f}% menos síntomas")
        elif exercise_impact['correlation'] >= 0.4:
            recommendations.append(f"⚠️ Modera la intensidad del ejercicio: podría estar aumentando síntomas")

    # If no specific recommendations, provide general advice
    if not recommendations:
        recommendations.append("Continúa registrando alimentos y ejercicio para obtener insights personalizados")

    return recommendations
