import openai 
from dotenv import load_dotenv

load_dotenv() 


def get_prompt_response(prompt, is_revision = False) -> dict:
    """
    Takes a prompt and returns response dictionary 
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system", 
            "content": "Your role is to answer the user prompt as accurately and concisely as possible."
        }, {
            "role": "user", 
            "content": prompt
        }]
    )
    generated_response = response['choices'][0]['message']['content']

    prefix = "revised_" if is_revision else ""

    prompt_response = {
        prefix + "prompt": prompt, 
        prefix + "response": generated_response
    }
    return prompt_response


def get_prompt_revision(data: dict = {"prompt": None, "response": None}, optimization_criteria: list = ["conciseness", "readability", "informative tone"]):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": 
                f"You review prompts given to GPT-3.5-Turbo and the responses the model produces. \
                  Rewrite the prompt so that the model response is optimized for the following characteristics: {', '.join(optimization_criteria)}."
            }, {
            "role": "user", 
            "content": 
                f"Rewrite the prompt based on the original prompt and response given below. \
                  Return ONLY your revised prompt ready to be given to an LLM. \
                  Here is the original prompt: {data['prompt']}. \
                  Here is the LLM response: {data['response']}." 
        }]
    )
    revised_prompt = response['choices'][0]['message']['content']
    return revised_prompt