import math
from datetime import datetime
from dateutil import parser as date_parser
from typing import List, Dict, Optional

class SeekerExperienceCalculator:
    """
    Dedicated experience engine for Job Seekers.
    Calculates unique work experience by merging overlapping date ranges.
    Strictly validates dates to prevent false experience inflation.
    """

    @staticmethod
    def normalize_date(date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        
        # Check for explicit "Present" indicators
        if any(word in str(date_str).lower() for word in ["present", "current", "now", "today"]):
            return datetime.now()
        
        try:
            parsed_date = date_parser.parse(str(date_str))
            # Safety: If date is in the far future, treat as invalid
            if parsed_date.year > datetime.now().year + 1:
                return None
            return parsed_date
        except (ValueError, TypeError):
            return None

    @staticmethod
    def calculate_months(start: datetime, end: datetime) -> int:
        if not start or not end or start > end:
            return 0
        return (end.year - start.year) * 12 + (end.month - start.month) + 1

    def calculate_total_experience(self, experience_entries: List[Dict]) -> Dict:
        if not experience_entries:
            return {"total_months": 0, "readable": "0 years 0 months", "decimal": 0.0}

        processed_intervals = []
        for entry in experience_entries:
            start = self.normalize_date(entry.get("start_date") or entry.get("startDate"))
            end = self.normalize_date(entry.get("end_date") or entry.get("endDate"))
            
            # CRITICAL FIX: Skip entry if start or end date is missing or invalid
            if not start or not end:
                continue

            if start > end: start, end = end, start
            processed_intervals.append({"start": start, "end": end})

        if not processed_intervals:
             return {"total_months": 0, "readable": "0 years 0 months", "decimal": 0.0}

        processed_intervals.sort(key=lambda x: x["start"])

        merged_ranges = []
        if processed_intervals:
            current_range = processed_intervals[0]
            for next_entry in processed_intervals[1:]:
                if next_entry["start"] <= current_range["end"]:
                    current_range["end"] = max(current_range["end"], next_entry["end"])
                else:
                    merged_ranges.append(current_range)
                    current_range = next_entry
            merged_ranges.append(current_range)

        total_months = sum(self.calculate_months(r["start"], r["end"]) for r in merged_ranges)
        decimal_val = round(total_months / 12, 1)

        return {
            "total_months": total_months,
            "decimal": decimal_val,
            "readable": f"{total_months // 12} years {total_months % 12} months"
        }

seeker_exp_manager = SeekerExperienceCalculator()
