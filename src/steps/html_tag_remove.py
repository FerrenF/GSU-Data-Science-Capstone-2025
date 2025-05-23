import re
import logging
from datetime import datetime

from core.step import Step

class RemoveHTMLTagsStep(Step):
    name = "remove_html_tags"

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        """
        Removes HTML tags from 'title' and 'text' fields using regular expressions.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def remove_html_tags(text):
            if not isinstance(text, str):
                return text
            # Replace <br> and <br /> tags with a period
            text = re.sub(r'<br\s*/?>', '.', text, flags=re.IGNORECASE)
            # Remove all other HTML tags
            clean_text = re.sub(r'<.*?>', '', text)
            return clean_text

        logging.info("Removing HTML tags from 'title' and 'text' columns...")
        df["title"] = df["title"].apply(remove_html_tags)
        df["text"] = df["text"].apply(remove_html_tags)

        data["dataset"] = df
        logging.info("HTML tags removed from 'title' and 'text'.")
        return data