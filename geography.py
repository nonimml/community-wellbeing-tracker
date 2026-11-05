"""Canonical province-to-district mapping for Sri Lanka."""

PROVINCE_DISTRICTS = {
    "Western": ("Colombo", "Gampaha", "Kalutara"),
    "Central": ("Kandy", "Matale", "Nuwara Eliya"),
    "Southern": ("Galle", "Matara", "Hambantota"),
    "Northern": ("Jaffna", "Kilinochchi", "Mannar", "Mullaitivu", "Vavuniya"),
    "Eastern": ("Batticaloa", "Ampara", "Trincomalee"),
    "North Western": ("Kurunegala", "Puttalam"),
    "North Central": ("Anuradhapura", "Polonnaruwa"),
    "Uva": ("Badulla", "Monaragala"),
    "Sabaragamuwa": ("Ratnapura", "Kegalle"),
}

PROVINCES = tuple(PROVINCE_DISTRICTS)
DISTRICT_TO_PROVINCE = {
    district: province
    for province, districts in PROVINCE_DISTRICTS.items()
    for district in districts
}
DISTRICTS = tuple(DISTRICT_TO_PROVINCE)


def is_valid_geography(province: str, district: str) -> bool:
    """Return whether a district belongs to the supplied province."""
    return DISTRICT_TO_PROVINCE.get(district) == province
