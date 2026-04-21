import math
from datetime import datetime
from dateutil import parser as date_parser
from typing import List, Dict, Optional

class ExperienceCalculator:
    """
    A production-ready engine to calculate unique work experience by merging 
    overlapping date ranges and normalizing various date formats.
    """

    @staticmethod
    def normalize_date(date_str: str) -> Optional[datetime]:
        """
        Strictly normalizes date strings. Returns None if unparseable.
        """
        if not date_str:
            return None
        
        # Check for explicit "Present" indicators
        if any(word in str(date_str).lower() for word in ["present", "current", "now", "today"]):
            return datetime.now()
        
        try:
            parsed_date = date_parser.parse(str(date_str))
            # Safety: If date is in the far future (AI error), treat as invalid
            if parsed_date.year > datetime.now().year + 1:
                return None
            return parsed_date
        except (ValueError, TypeError):
            return None

    @staticmethod
    def calculate_months(start: datetime, end: datetime) -> int:
        """Calculates total months between two dates, inclusive."""
        if not start or not end or start > end:
            return 0
        # Total months = (years * 12) + months + 1 (to include the start/end month)
        return (end.year - start.year) * 12 + (end.month - start.month) + 1

    def calculate_total_experience(self, experience_entries: List[Dict]) -> Dict:
        """
        Logic with strict validation:
        1. Skips entries with invalid start/end dates.
        2. Only allows 'now' for current roles.
        3. Merges overlapping ranges correctly.
        """
        if not experience_entries:
            return self._empty_response()

        processed_intervals = []
        for entry in experience_entries:
            start_str = entry.get("startDate") or entry.get("start_date", "")
            end_str = entry.get("endDate") or entry.get("end_date", "")
            
            start = self.normalize_date(start_str)
            end = self.normalize_date(end_str)
            
            # CRITICAL FIX: Skip if start or end date is missing or invalid
            if not start or not end:
                continue
                
            # Ensure start <= end
            if start > end:
                start, end = end, start
                
            processed_intervals.append({
                "start": start,
                "end": end
            })

        if not processed_intervals:
            return self._empty_response()

        # Step 1: Sort intervals by start date
        processed_intervals.sort(key=lambda x: x["start"])

        # Step 2: Merge overlapping intervals
        merged_ranges = []
        if processed_intervals:
            current_range = {
                "start": processed_intervals[0]["start"],
                "end": processed_intervals[0]["end"]
            }

            for next_entry in processed_intervals[1:]:
                # If next starts before or exactly when current ends (or is adjacent/touching)
                # We use -1 month logic if we want to merge "touching" months, 
                # but standard overlap is next_entry["start"] <= current_range["end"]
                if next_entry["start"] <= current_range["end"]:
                    # Update end date if the next one extends further
                    current_range["end"] = max(current_range["end"], next_entry["end"])
                else:
                    # No overlap, push finished range and start new one
                    merged_ranges.append(current_range)
                    current_range = {
                        "start": next_entry["start"],
                        "end": next_entry["end"]
                    }
            merged_ranges.append(current_range)

        # Step 3: Calculate total unique months
        total_unique_months = sum(
            self.calculate_months(r["start"], r["end"]) for r in merged_ranges
        )

        # Step 4: Formatting
        years = total_unique_months // 12
        remaining_months = total_unique_months % 12
        
        # Mathematical decimal format (e.g., 1yr 11mo = 1.91)
        decimal_val = round(total_unique_months / 12, 1)

        return {
            "total_months": total_unique_months,
            "readable": f"{years} year{'s' if years != 1 else ''} {remaining_months} month{'s' if remaining_months != 1 else ''}",
            "decimal": decimal_val,
            "merged_ranges": [
                {
                    "start_date": r["start"].strftime("%b %Y"),
                    "end_date": r["end"].strftime("%b %Y"),
                    "duration_months": self.calculate_months(r["start"], r["end"])
                } for r in merged_ranges
            ]
        }

    def _empty_response(self):
        return {
            "total_months": 0, 
            "readable": "0 years 0 months", 
            "decimal": 0.0, 
            "merged_ranges": []
        }

exp_manager = ExperienceCalculator()
