import random

class AIScriptWriter:
    def __init__(self):
        self.story_elements = {
            "genre": ["Sci-Fi", "Horror", "Fantasy"],
            "characters": ["AI explorer", "Lost astronaut", "Time traveler"],
            "conflict": ["finds a portal", "escapes a dystopia", "unlocks forbidden knowledge"]
        }

    def generate_script(self):
        return f"{random.choice(self.story_elements['characters'])} {random.choice(self.story_elements['conflict'])} in a {random.choice(self.story_elements['genre'])} world."

if __name__ == "__main__":
    script_ai = AIScriptWriter()
    print("Generated AI Script:", script_ai.generate_script())
