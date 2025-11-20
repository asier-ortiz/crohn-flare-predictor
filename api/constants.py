"""
Constants for IBD classification and cluster mapping.
Based on Montreal Classification for Crohn's Disease and Ulcerative Colitis.
"""

# Montreal Classification for Crohn's Disease (Location)
CROHN_MONTREAL_LOCATIONS = {
    'L1': {
        'name': 'Ileal',
        'description': 'Íleon terminal ± ciego',
        'cluster': 0,  # Maps to Cluster 0 (high pain, low diarrhea)
    },
    'L2': {
        'name': 'Colonic',
        'description': 'Solo colon',
        'cluster': 2,  # Maps to Cluster 2 (high diarrhea, blood)
    },
    'L3': {
        'name': 'Ileocolonic',
        'description': 'Íleon terminal + colon',
        'cluster': 1,  # Maps to Cluster 1 (mixed symptoms)
    },
    'L4': {
        'name': 'Upper GI',
        'description': 'Tracto gastrointestinal superior',
        'cluster': 1,  # Maps to Cluster 1 (default mixed)
    }
}

# Montreal Classification for Ulcerative Colitis (Extent)
UC_MONTREAL_EXTENT = {
    'E1': {
        'name': 'Proctitis',
        'description': 'Afectación limitada al recto',
        'cluster': 0,  # Maps to Cluster 0 (mild symptoms, rectal bleeding)
    },
    'E2': {
        'name': 'Left-sided colitis',
        'description': 'Hasta flexura esplénica',
        'cluster': 1,  # Maps to Cluster 1 (moderate symptoms)
    },
    'E3': {
        'name': 'Pancolitis',
        'description': 'Afectación de todo el colon',
        'cluster': 2,  # Maps to Cluster 2 (severe symptoms, extensive)
    }
}

# Simplified mapping: Montreal → Cluster
CROHN_MONTREAL_TO_CLUSTER = {
    'L1': 0,
    'L2': 2,
    'L3': 1,
    'L4': 1,
}

UC_MONTREAL_TO_CLUSTER = {
    'E1': 0,
    'E2': 1,
    'E3': 2,
}

# All Montreal codes (for validation)
VALID_CROHN_LOCATIONS = list(CROHN_MONTREAL_LOCATIONS.keys())
VALID_UC_EXTENTS = list(UC_MONTREAL_EXTENT.keys())
VALID_MONTREAL_CODES = VALID_CROHN_LOCATIONS + VALID_UC_EXTENTS

# IBD Types
IBD_TYPES = ['crohn', 'ulcerative_colitis']

# Cluster descriptions (for UI/reporting)
CROHN_CLUSTER_DESCRIPTIONS = {
    0: 'Patrón Ileal (L1-like): Alto dolor abdominal, diarrea moderada',
    1: 'Patrón Ileocolónico (L3-like): Síntomas balanceados',
    2: 'Patrón Colónico (L2-like): Alta diarrea, sangre en heces',
}

UC_CLUSTER_DESCRIPTIONS = {
    0: 'Patrón Proctitis (E1-like): Afectación rectal, síntomas leves',
    1: 'Patrón Left-sided (E2-like): Síntomas moderados',
    2: 'Patrón Pancolitis (E3-like): Síntomas severos, extensos',
}


def get_cluster_from_montreal(montreal_code: str) -> int:
    """
    Get cluster ID from Montreal classification code.

    Args:
        montreal_code: Montreal code (L1-L4 for Crohn, E1-E3 for UC)

    Returns:
        Cluster ID (0-2)

    Raises:
        ValueError: If invalid Montreal code
    """
    if montreal_code in CROHN_MONTREAL_TO_CLUSTER:
        return CROHN_MONTREAL_TO_CLUSTER[montreal_code]
    elif montreal_code in UC_MONTREAL_TO_CLUSTER:
        return UC_MONTREAL_TO_CLUSTER[montreal_code]
    else:
        raise ValueError(f"Invalid Montreal code: {montreal_code}")


def get_ibd_type_from_montreal(montreal_code: str) -> str:
    """
    Determine IBD type from Montreal code.

    Args:
        montreal_code: Montreal code (L1-L4 for Crohn, E1-E3 for UC)

    Returns:
        IBD type: 'crohn' or 'ulcerative_colitis'

    Raises:
        ValueError: If invalid Montreal code
    """
    if montreal_code in VALID_CROHN_LOCATIONS:
        return 'crohn'
    elif montreal_code in VALID_UC_EXTENTS:
        return 'ulcerative_colitis'
    else:
        raise ValueError(f"Invalid Montreal code: {montreal_code}")


def get_montreal_description(montreal_code: str) -> str:
    """
    Get human-readable description of Montreal classification.

    Args:
        montreal_code: Montreal code (L1-L4 for Crohn, E1-E3 for UC)

    Returns:
        Description string

    Raises:
        ValueError: If invalid Montreal code
    """
    if montreal_code in CROHN_MONTREAL_LOCATIONS:
        return f"{CROHN_MONTREAL_LOCATIONS[montreal_code]['name']} - {CROHN_MONTREAL_LOCATIONS[montreal_code]['description']}"
    elif montreal_code in UC_MONTREAL_EXTENT:
        return f"{UC_MONTREAL_EXTENT[montreal_code]['name']} - {UC_MONTREAL_EXTENT[montreal_code]['description']}"
    else:
        raise ValueError(f"Invalid Montreal code: {montreal_code}")
