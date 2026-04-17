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
    def normalize_date(date_str: str) -> datetime:
        """
        Normalizes various string formats into a datetime object.
        Handles 'Present', 'Current', and various date string variations.
        """
        if not date_str or any(word in date_str.lower() for word in ["present", "current", "now"]):
            return datetime.now()
        
        try:
            return date_parser.parse(date_str)
        except (ValueError, TypeError):
            return datetime.now()

    @staticmethod
    def calculate_months(start: datetime, end: datetime) -> int:
        """Calculates total months between two dates, inclusive."""
        if start > end:
            return 0
        # Total months = (years * 12) + months + 1 (to include the start/end month)
        return (end.year - start.year) * 12 + (end.month - start.month) + 1

    def calculate_total_experience(self, experience_entries: List[Dict]) -> Dict:
        """
        Main engine logic:
        1. Normalizes all entries.
        2. Sorts by start date.
        3. Merges overlapping or touching intervals.
        4. Calculates final totals in multiple formats.
        """
        if not experience_entries:
            return self._empty_response()

        processed_intervals = []
        for entry in experience_entries:
            start_str = entry.get("startDate") or entry.get("start_date", "")
            end_str = entry.get("endDate") or entry.get("end_date", "")
            
            start = self.normalize_date(start_str)
            end = self.normalize_date(end_str)
            
            # Basic validation: ensure start <= end
            if start > end:
                start, end = end, start
                
            processed_intervals.append({
                "start": start,
                "end": end,
                "company": entry.get("company", "Unknown")
            })

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
