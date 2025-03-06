# AI Character Chat using LangChain

This example demonstrates how to create an AI character chat system using LangChain. The system allows users to have conversations with AI characters that maintain consistent personalities, speech patterns, and worldviews.

## Features

- Three-level prompt structure:
  - **Service Prompt**: Top-level prompt defining the roleplay framework
  - **Character Prompt**: Specific character details including worldview, traits, and example dialog
  - **User Prompt**: Information about the user to help the AI respond appropriately

- Conversation memory that retains previous interactions
- Support for multiple characters with different personalities

## Requirements

- Python 3.8+
- LangChain
- OpenAI API key or OpenRouter API key

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your API keys in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Usage

Run the example script:

```
python example_character-chat.py
```

The script will prompt you to select a character to chat with (Sherlock Holmes or Tony Stark). After selecting a character, you can have a conversation by typing your messages.

## Creating Your Own Characters

To create your own character, define the following prompts:

1. **Service Prompt**: General instructions for the roleplay
2. **Character Prompt**: Details about your character including:
   - Name
   - Worldview
   - Character Traits
   - Speech Pattern
   - Example Dialog

3. **User Prompt**: Information about the user interacting with the character

Then create a `CharacterChat` instance with these prompts:

```python
my_character = CharacterChat(
    service_prompt=service_prompt,
    character_prompt=my_character_prompt,
    user_prompt=my_user_prompt
)
```

## Customization

You can customize various aspects of the character chat:

- **Memory Size**: Change the `memory_k` parameter to control how many previous conversations are remembered
- **Temperature**: Adjust the temperature in the `ChatOpenRouter` initialization to control response randomness
- **Model**: Change the model in the `ChatOpenRouter` class to use different LLMs

## Example

```python
# Create a new character
wizard_prompt = """
Name: Gandalf the Grey
Worldview: Believes in the inherent good of all beings, but wary of power's corruption.
Character Traits: Wise, patient, mysterious, powerful but restrained, enjoys simple pleasures.
Speech Pattern: Speaks in riddles, philosophical, alternates between formal and casual speech.

Example Dialog:
"A wizard is never late, nor is he early. He arrives precisely when he means to."
"All we have to decide is what to do with the time that is given to us."
"Fly, you fools!"
"""

fantasy_user_prompt = """
The user is a fantasy enthusiast interested in Middle-earth lore and magical adventures.
They may ask about your journeys, magic, or seek your wisdom on various matters.
"""

gandalf_chat = CharacterChat(
    service_prompt=service_prompt,
    character_prompt=wizard_prompt,
    user_prompt=fantasy_user_prompt
)
``` 