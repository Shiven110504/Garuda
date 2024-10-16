#!/usr/bin/env python3

from flask import Flask, request, jsonify
import cv2
import os
import subprocess

app = Flask(__name__)

# Ensure the tmp directory exists
os.makedirs('tmp', exist_ok=True)

@app.route('/process', methods=['POST'])
def process_video():
    data = request.get_json()
    if not data or 'video_path' not in data:
        return jsonify({'error': 'No video path provided'}), 400

    video_path = data['video_path']
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file does not exist'}), 400

    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frame rate of the video
    frame_interval = frame_rate * 5  # Process every 5 seconds (frame_rate * 5)
    all_responses = []
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Process every 5 seconds
        if current_frame % frame_interval == 0:
            # Save the frame as an image
            frame_path = f"tmp/frame_{frame_index}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_index += 1

            # Concatenate all previous responses into one string
            previous_responses_text = "\n".join(all_responses)

            # Call the OpenAI API for the current frame, passing all previous responses for chain-of-thought reasoning
            prompt = create_chain_of_thought_prompt(frame_path, previous_responses_text)
            response = process_image_and_prompt(frame_path, prompt)
            if response:
                print(f"Response for frame {current_frame}: {response}")
                all_responses.append(response)  # Store response

    # After processing the last frame, generate the final summary
    final_summary_prompt = create_final_summary_prompt("\n".join(all_responses))
    final_summary = process_image_and_prompt(None, final_summary_prompt)
    green_solutions_prompt = create_green_solutions_prompt(final_summary)
    green_solutions = process_green_solutions(green_solutions_prompt)
    cap.release()
    
    # Return JSON data for each frame and the final summary
    result = {
        'frames': all_responses,
        'summary': final_summary,
        'green_solutions': green_solutions
    }
    return jsonify(result), 200


def process_image_and_prompt(image_path, prompt):
    try:
        # Prepare the command to run Ollama with the Llava model
        command = f'ollama run llava'

        # Create the multi-line prompt
        multi_line_prompt = f"""
        {prompt}
        """
        if image_path:
            # If there's an image, add the image path to the prompt
            multi_line_prompt += f"\n{image_path}"

        # Run the command and capture the output
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        stdout, stderr = process.communicate(input=multi_line_prompt)

        # Check for errors
        if process.returncode != 0:
            return f'Error: {stderr}'

        # Return the output from Ollama
        return stdout.strip()

    except Exception as e:
        return f'Error: {str(e)}'
    
def process_green_solutions(prompt):
    try:
        # Prepare the command to run Ollama with the Llava model
        command = f'ollama run llava'

        # Create the multi-line prompt
        multi_line_prompt = f"""
        {prompt}
        """

        # Run the command and capture the output
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        stdout, stderr = process.communicate(input=multi_line_prompt)

        # Check for errors
        if process.returncode != 0:
            return f'Error: {stderr}'

        # Return the output from Ollama
        return stdout.strip()

    except Exception as e:
        return f'Error: {str(e)}'


def create_final_summary_prompt(all_responses):
    # Chain of thought prompt for the final summary
    prompt = (
        """
        You are provided with a series of real-time observations from a disaster scene, detailing structural damage, 
        infrastructure impact, accessibility challenges, rescue needs, and other key factors, reported frame by 
        frame. Using these observations, generate a concise yet comprehensive AI-driven summary that includes:
        1. Overall severity: Assess the total extent of the disaster and determine the critical response level required.
        2. Damage types and affected areas: Identify and categorize the types of damage (e.g., structural, environmental)
        and highlight the affected infrastructure.
        3. Rescue priorities: Summarize the immediate rescue needs observed, including areas where survivors may be trapped
        or in critical danger.
        4. Accessibility and hazards: Highlight any accessibility issues, such as blocked routes or unsafe zones, and
        areas posing risks to rescue teams.
        5. Additional concerns: Note any other important factors that could impact rescue operations or require special
        attention.

        Here is the information collected from the video:{all_responses}
        """
    )
    return prompt


def create_chain_of_thought_prompt(image_path, all_responses=None):
    # Create a chain of thought prompt for processing an individual image
    prompt = (
        """
        Analyze the image and provide the following details as a JSON object (only choose one category for each):
        1. Damage Severity: ('minor', 'moderate', 'severe', or 'catastrophic').
        2. Critical Response Level: (1 to 5, where 5 is the most urgent, based on the severity of damage and
        immediate danger to lives).
        3. Damage Type: ('structural', 'fire-related', 'flood-related', 'landslide', 'wind damage',
        'explosion-related', or 'other'). Can be more specific (e.g., 'collapsed buildings',
        'burning structures', 'flooded areas', 'damaged roads', etc.) but only choose one category.
        4. Infrastructure Affected: (most affected infrastructure, e.g., 'roads', 'bridges',
        'buildings', 'power lines', 'communication towers', 'none').
        5. Rescue Needed: ('yes' or 'no').
        6. Accessibility: ('accessible', 'partially blocked', or 'blocked').
        7. Additional Hazards: (most applicable secondary danger such as 'gas leaks', 'downed power lines',
        or 'chemical spills', 'none').
        8. Suggested Equipment: (optional list of most recommended tool/vehicle needed for the scene, e.g., 'cranes',
        'boats', 'rescue helicopters', 'none').
        """
        "\nExpected Output:\n"
        "{\n"
        "  \"damage_severity\": \"\",\n"
        "  \"critical_response_level\": 0,\n"
        "  \"damage_type\": \"\",\n"
        "  \"infrastructure_affected\": [],\n"
        "  \"rescue_needed\": \"\",\n"
        "  \"accessibility\": \"\"\n"
        "  \"additional_hazards\": \"\"\n"
        "  \"suggested_equipment\": \"\"\n"
        "}"
    )

    # Add all previous responses to the prompt if they exist for chain of thought
    if all_responses:
        prompt += f"\nPrevious responses:\n{all_responses}"

    return prompt

def create_green_solutions_prompt(final_summary):
    prompt = (
        f"""
        Based on the following summary of the disaster, provide a list of green solutions that can be implemented
        to mitigate the damage and improve the situation: {final_summary}
        Address the following aspects:
        1. Immediate damage control: Eco-friendly methods to reduce harm and protect ecosystems.
        2. Waste management: Sustainable practices for handling debris, waste, and hazardous materials.
        3. Energy solutions: Renewable energy options to support recovery and reduce dependency on fossil fuels.
        4. Water management: Strategies to conserve water and ensure clean, accessible water supplies.
        5. Infrastructure rebuilding: Use of sustainable materials and green technologies in rebuilding efforts.
        6. Long-term resilience: Solutions that enhance environmental sustainability and disaster preparedness for the future.  
        """
    )
    return prompt


if __name__ == '__main__':
    app.run(host='localhost', port=8081)