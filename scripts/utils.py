import openai 
from dotenv import load_dotenv

load_dotenv() 


def get_prompt_response(prompt, is_revision = False) -> dict:
    """
    Takes a prompt and returns response dictionary 
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your role is to answer the user prompt as accurately and concisely as possible."},
            {"role": "user", "content": prompt}
        ]
    )
    generated_response = response['choices'][0]['message']['content']

    prefix = "revised_" if is_revision else ""

    prompt_response = {
        prefix + "prompt": prompt, 
        prefix + "response": generated_response
    }
    return prompt_response


def get_prompt_revision(data: dict = {"prompt": None, "response": None}):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your role is to review LLM prompts and their responses. \
                                           You have a critical eye for refining prompts to produce the \
                                           most concise, accurate, and detailed reponses possible."
             },
            {"role": "user", "content": f"Revise the prompt so that the response is as clear and concise as possible. \
                                            Here is the original prompt: {data['prompt']}. \
                                            Here is the LLM response: {data['response']}. \
                                            Rewrite the prompt."
            } 
        ]
    )
    revised_prompt = response['choices'][0]['message']['content']
    return revised_prompt