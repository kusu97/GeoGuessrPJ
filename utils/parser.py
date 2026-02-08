'''
This file is adapted from:
https://github.com/ccmdi/geobench/blob/main/scripts/parser.py
Original author: ccmdi
License: MIT

A script for extracting geographic information from MLLM responses.
'''

from models.base import Prediction
import re

def parse_response(response: str) -> Prediction:
    # Look for the final answer section that matches the required format
    final_answer_match = re.search(
        # Country part:
        r"(?:^|\n)(?:\*\*)?(?:country|Country)(?:\*\*)?:\s*"  # Keyword (optionally **keyword**), colon, initial space
        r"(\*+)?\s*([^,\r\n]+?)\s*(\*+)?"                  # Optional value wrapper (* or ** etc.), internal spaces, country name, internal spaces, optional value wrapper
        r"\s*(?:\n|$)"                                      # Trailing space and newline/end
        # Separator:
        r".*?"
        # Latitude part:
        r"(?:^|\n)(?:\*\*)?(?:lat|Lat)(?:\*\*)?:\s*"          # Keyword (optionally **keyword**), colon, initial space
        r"(\*+)?\s*([-+]?\d+\.?\d*?)\s*(\*+)?"              # Optional value wrapper (* or ** etc.), internal spaces, lat number, internal spaces, optional value wrapper
        r"\s*(?:\n|$)"                                      # Trailing space and newline/end
        # Separator:
        r".*?"
        # Longitude part:
        r"(?:^|\n)(?:\*\*)?(?:lng|Lng)(?:\*\*)?:\s*"          # Keyword (optionally **keyword**), colon, initial space
        r"(\*+)?\s*([-+]?\d+\.?\d*?)\s*(\*+)?"              # Optional value wrapper (* or ** etc.), internal spaces, lng number, internal spaces, optional value wrapper
        r"\s*(?:\n|$)",                                     # Trailing space and newline/end
        response,
        re.MULTILINE | re.DOTALL
    )
    if not final_answer_match:
        raise ValueError("Response missing required fields in final answer format")

    try:
        country = final_answer_match.group(2).strip() # Country name is now group 2
        lat_str = final_answer_match.group(5).strip() # Lat number is now group 5
        lng_str = final_answer_match.group(8).strip() # Lng number is now group 8
        
        lat = float(lat_str)
        lng = float(lng_str)
    except (AttributeError, IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse final answer: {e}")
    
    if not -90 <= lat <= 90:
        raise ValueError(f"Invalid latitude value: {lat} (must be between -90 and 90)")
    if not -180 <= lng <= 180:
        raise ValueError(f"Invalid longitude value: {lng} (must be between -180 and 180)")
        
    return Prediction(country=country, lat=lat, lng=lng)


def parse_response_with_exception_handler(response: str) -> Prediction | None:
    try:
        return parse_response(response)
    except ValueError:
        # return None if parse_response fails
        return None


if __name__ == '__main__':
    pass