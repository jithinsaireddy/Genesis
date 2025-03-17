import random

class StoryGenerator:
    def __init__(self):
        self.themes = ["sci-fi", "horror", "fantasy", "mystery", "cyberpunk"]
        self.characters = ["AI explorer", "Time traveler", "Rogue android", "Lost astronaut"]
        self.events = ["discovers a secret", "finds a portal", "escapes a dystopia", "unlocks forbidden AI knowledge"]

    def generate_story(self):
        theme = random.choice(self.themes)
        character = random.choice(self.characters)
        event = random.choice(self.events)
        return f"In a {theme} world, {character} {event}."

    def generate_multiple_stories(self, num_stories=5):
        """Generate multiple unique stories"""
        stories = []
        for _ in range(num_stories):
            stories.append(self.generate_story())
        return stories

class InteractiveStoryAI:
    def __init__(self):
        self.paths = {
            "start": ["You wake up in a futuristic city.", "You find yourself in a dystopian wasteland."],
            "choice_1": ["Explore the city", "Look for other survivors"],
            "choice_2": ["Enter the AI lab", "Escape to the mountains"]
        }

    def generate_story(self, user_choice):
        return self.paths[user_choice]

# Example usage
if __name__ == "__main__":
    # Example of using original StoryGenerator
    generator = StoryGenerator()
    print("Random Story:", generator.generate_story())
    
    # Example of using InteractiveStoryAI
    story_ai = InteractiveStoryAI()
    story = story_ai.generate_story("start")
    print(f"AI Story Path: {story}")

    print("\nMultiple Stories:")
    for i, story in enumerate(generator.generate_multiple_stories(3), 1):
        print(f"Story {i}: {story}")
