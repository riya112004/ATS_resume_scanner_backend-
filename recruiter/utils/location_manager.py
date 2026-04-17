import re
from typing import Dict, Optional

class LocationManager:
    """
    Lightweight Location Utility.
    Reliant on AI-extracted structured data (city, state, country).
    No static mappings or hardcoded lists.
    """

    def clean_text(self, text: str) -> str:
        """Standard cleanup for comparison."""
        if not text: return ""
        return re.sub(r'[^a-zA-Z0-9]', '', str(text).lower().strip())

    def calculate_location_score(self, search_query: str, res_city: str, res_state: str, res_country: str) -> float:
        """
        Calculates location match score based on AI-extracted fields.
        1.0 = Strong Match (Search term found in City/State/Country)
        0.0 = No Match
        """
        if not search_query:
            return 1.0 # Default if no location search

        # Normalize all inputs
        query_parts = [self.clean_text(p) for p in search_query.split(",") if p.strip()]
        res_values = [
            self.clean_text(res_city),
            self.clean_text(res_state),
            self.clean_text(res_country)
        ]

        # Check if ANY part of search query matches ANY AI-extracted location field
        for part in query_parts:
            if any(part == val for val in res_values if val):
                return 1.0
            
            # Sub-string match (e.g., "Savannah" in "Savannah GA")
            if any(part in val or val in part for val in res_values if val):
                return 0.8 # Slightly lower for partial matches

        return 0.0

loc_manager = LocationManager()
