import openai
from dotenv import load_dotenv
load_dotenv() 


def generate_prompt(prompt_model, prompt=None):
    response = openai.ChatCompletion.create(
        model=prompt_model,
        messages=[
            {"role": "system", "content": "You are a generator function from a Generative Adversarial Network (GAN)."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def generate_response(prompt, response_model):
    response = openai.ChatCompletion.create(
        model=response_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def analyze_response(prompt, response, critique_model):
    analysis_prompt = f"Given this prompt: {prompt}, and this response: {response}, give feedback on how you would re-write the prompt based on the information you'd want in the response."
    
    response = openai.ChatCompletion.create(
        model=critique_model,
        messages=[
            {"role": "system", "content": "You are a discriminator function from a Generative Adversarial Network (GAN)."},
            {"role": "user", "content": analysis_prompt}
        ]
    )
    return response.choices[0].message['content']

def main():
    prompt_model = "gpt-3.5-turbo"
    response_model = "gpt-3.5-turbo"
    critique_model = "gpt-4o"
    iterations = 5
    
    initial_prompt = generate_prompt(prompt_model)
    print(f"Initial Prompt: {initial_prompt}")
    
    current_prompt = initial_prompt
    for i in range(iterations):
        print(f"\nIteration {i+1}:")
        
        response = generate_response(current_prompt, response_model)
        print(f"Response: {response}")
        
        critique = analyze_response(current_prompt, response, critique_model)
        print(f"Critique: {critique}")
        
        current_prompt = generate_prompt(prompt_model, scaffolded_input=initial_prompt + critique)
        print(f"Refined Prompt: {current_prompt}")
        
    # Compare final refined response with a vanilla prompt to GPT-4
    final_response = generate_response(current_prompt, response_model)
    print(f"\nFinal Refined Response: {final_response}")
    
    vanilla_response = generate_response(initial_prompt, critique_model)
    print(f"\nVanilla GPT-4o Response: {vanilla_response}")

if __name__ == "__main__":
    main()
