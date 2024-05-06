import json
import os
from io import BytesIO
import base64
import argparse
from copy import deepcopy
from tqdm import tqdm
import openai
from PIL import Image

SYSTEM_PROMPT1 = \
"""You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of Person 1 and Person 2 who are physically in contact with each other, create a Markdown table with the columns "Person 1 Body Part" and "Person 2 Body Part", listing the body parts of the two people that are guaranteed to be in contact with each other, from the following list. ALL body parts that you list must be from this list.
Body parts: "chest", "stomach", "waist", "back", "hand", "arm", "shoulder", "foot", "leg", "leg", "head", "neck", "butt"

If a contact point could be in either of two parts, then use a comma-separated list to list the two parts (e.g. "back, waist"). If the contact could be in three or more parts, do not include this row in the table.

If there are no contact points between these body parts that the description directly implies, your table should contain only the column names and no other rows.

First, write your reasoning. Then write the Markdown table."""
SYSTEM_PROMPT1_noperson1 = \
"""You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of two people who are physically in contact with each other, create a Markdown table with the columns "Person 1 Body Part" and "Person 2 Body Part", listing the body parts of the two people that are guaranteed to be in contact with each other, from the following list. ALL body parts that you list must be from this list. You can choose which person is Person 1 and which is Person 2.
Body parts: "chest", "stomach", "waist", "back", "hand", "arm", "shoulder", "foot", "leg", "leg", "head", "neck", "butt"

If a contact point could be in either of two parts, then use a comma-separated list to list the two parts (e.g. "back, waist"). If the contact could be in three or more parts, do not include this row in the table.

If there are no contact points between these body parts that the description directly implies, your table should contain only the column names and no other rows.

First, write your reasoning. Then write the Markdown table."""

"""Body parts: "chest", "stomach", "front waist", "front shoulder", "lower back", "upper back", "hand", "arm", "foot", "leg", "head", "neck", "butt"
Note that "lower back" is the back side of the waist and stomach, and "upper back" is the back side of the chest and shoulders."""

SYSTEM_PROMPT2 = \
"""You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of Person 1 and Person 2 who are physically in contact with each other, create a Markdown table with the columns "Person 1 Body Part" and "Person 2 Body Part", listing the body parts of the two people that are guaranteed to be in contact with each other, from the following list. ALL body parts that you list must be from this list.
Body parts: "chest", "stomach", "waist (front)", "waist (back)", "shoulder (front)", "shoulder (back)", "back", "hand", "arm", "foot", "leg", "head", "neck", "butt"
Note that "back" includes the entire area of the back.

Do your best to resolve ambiguity between body parts using commonsense. If you are unable to decide between two or three body parts for a contact point, then use a comma-separated list to list the parts (e.g. "head, neck"). If the contact could be in four or more parts, do not include this row in the table.

Only include contact points that are directly implied by the description.
If there are no contact points between these body parts that the description directly implies, your table should contain only the column names and no other rows.

First, write your reasoning. Then write the Markdown table."""

SYSTEM_PROMPT2_noperson1 = \
"""You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of two people who are physically in contact with each other, create a Markdown table with the columns "Person 1 Body Part" and "Person 2 Body Part", listing the body parts of the two people that are guaranteed to be in contact with each other, from the following list. ALL body parts that you list must be from this list. You can choose which person is Person 1 and which is Person 2.
Body parts: "chest", "stomach", "front waist", "front shoulder", "back waist", "back shoulder", "back middle", "hand", "arm", "foot", "leg", "head", "neck", "butt"
Note that "back middle" is the area on the back above the waist and below the shoulders.

If a contact point could be in either of two parts, then use a comma-separated list to list the two parts (e.g. "head, neck"). If the contact could be in three or more parts, do not include this row in the table.

Only include contact points that are directly implied by the description.
If there are no contact points between these body parts that the description directly implies, your table should contain only the column names and no other rows.

First, write your reasoning. Then write the Markdown table."""

SYSTEM_PROMPT3  = \
"""You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of Person 1 and Person 2 who are physically in contact with each other, create a Markdown table with the columns "Person 1 Body Part" and "Person 2 Body Part", listing the body parts of the two people that are guaranteed to be in contact with each other, from the following list. ALL body parts that you list must be from this list.
Body parts: "chest", "stomach", "waist (front)", "waist (back)", "shoulder (front)", "shoulder (back)", "back", "hand", "arm", "foot", "leg", "head", "neck", "butt"
Note that "back" includes the entire area of the back.

Do your best to resolve ambiguity between body parts using commonsense. If you are unable to decide between two or three body parts for a contact point, then use a comma-separated list to list the parts (e.g. "head, neck"). If the contact could be in four or more parts, do not include this row in the table.

Include all contact points that are directly implied by the description, not just those that are explicitly mentioned.
If there are no contact points between these body parts that the description implicitly or explicitly implies, your table should contain only the column names and no other rows.

First, write your reasoning. Then write the Markdown table."""

SYSTEM_PROMPT3_noperson1  = \
"""You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of two people who are physically in contact with each other, create a Markdown table with the columns "Person 1 Body Part" and "Person 2 Body Part", listing the body parts of the two people that are guaranteed to be in contact with each other, from the following list. ALL body parts that you list must be from this list. You can choose which person is Person 1 and which is Person 2.
Body parts: "chest", "stomach", "waist (front)", "waist (back)", "shoulder (front)", "shoulder (back)", "back", "hand", "arm", "foot", "leg", "head", "neck", "butt"
Note that "back" includes the entire area of the back.

Do your best to resolve ambiguity between body parts using commonsense. If you are unable to decide between two or three body parts for a contact point, then use a comma-separated list to list the parts (e.g. "head, neck"). If the contact could be in four or more parts, do not include this row in the table.

Include all contact points that are directly implied by the description, not just those that are explicitly mentioned.
If there are no contact points between these body parts that the description implicitly or explicitly implies, your table should contain only the column names and no other rows.

First, write your reasoning. Then write the Markdown table."""

SYSTEM_PROMPT4_noperson1  = \
"""You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of two people who are physically in contact with each other, create a Markdown table with the columns "Person 1 Body Part" and "Person 2 Body Part", listing the body parts of the two people that are guaranteed to be in contact with each other, from the following list. ALL body parts that you list must be from this list. You can choose which person is Person 1 and which is Person 2.
Body parts: "chest", "stomach", "waist (front)", "waist (back)", "shoulder (front)", "shoulder (back)", "back", "hand", "arm", "foot", "leg", "head", "neck", "butt"
Note that "back" includes the entire area of the back.

Include all contact points that are directly implied by the description, not just those that are explicitly mentioned.
If there are no contact points between these body parts that the description implicitly or explicitly implies, your table should contain only the column names and no other rows.

First, write your reasoning. Then write the Markdown table."""

YOGA_SYSTEM_PROMPT4 = \
"""You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of a yoga pose, create a Markdown table with the columns "Body Part 1" and "Body Part 2", listing the body parts of the person that are guaranteed to be in contact with each other, from the following list. ALL body parts that you list must be from this list.
Body parts: "head", "back", "shoulder", "arm", "hand", "leg", "foot", "stomach", "butt", "ground"
Note that "back" includes the entire area of the back.

Include all contact points that are directly implied by the description, not just those that are explicitly mentioned.
If there are no contact points between these body parts that the description implicitly or explicitly implies, your table should contain only the column names and no other rows.

First, write your reasoning. Then write the Markdown table."""

VISION_SYSTEM_PROMPT1 = \
"""You are a helpful assistant. You follow all directions correctly and precisely.
For each image, identify all pairs of body parts of Person 1 and Person 2 that are touching.
Write all of these in a Markdown table where the first column is "Person 1 Body Part" and the second column is "Person 2 Body Part".
You can pick which is Person 1 and which is Person 2.
The list of possible body parts is: head, neck, chest, stomach, waist, back, shoulder, arm, hand, leg, foot, butt.

List all pairs you are confident about.
If there are no pairs of body parts that you are confident about, output no pairs.
"""

VISION_SYSTEM_PROMPT2 = \
"""You are a helpful assistant. You follow all directions correctly and precisely.
For each image, identify all pairs of body parts of Person 1 and Person 2 that are touching.
Write all of these in a Markdown table where the first column is "Person 1 Body Part" and the second column is "Person 2 Body Part".
You can pick which is Person 1 and which is Person 2.
The list of possible body parts is: head, neck, chest, stomach, waist (back), waist (front), back, shoulder (back), shoulder (front), arm, hand, leg, foot, butt.
Do not include left/right.

List ALL pairs you are confident about.
If you are not confident about any pairs, output an empty table.
Carefully write your reasoning first, and then write the Markdown table.
"""

VISION_SYSTEM_PROMPT2_LR = \
"""You are a helpful assistant. You follow all directions correctly and precisely.
For each image, identify all pairs of body parts of Person 1 and Person 2 that are touching.
Write all of these in a Markdown table where the first column is "Person 1 Body Part" and the second column is "Person 2 Body Part".
You can pick which is Person 1 and which is Person 2.
The list of possible body parts is: head, neck, chest, stomach, waist (back), waist (front), back, shoulder (back), shoulder (front), arm, hand, leg, foot, butt.
Include left/right when appropriate.

Only list pairs you are absolutely certain about.
If you are not certain about any pairs, output an empty table.
Carefully write your reasoning first, and then write the Markdown table.
"""

VISION_SYSTEM_PROMPT3 = \
"""You are a helpful assistant. You follow all directions correctly and precisely.
For each image, identify all pairs of body parts of Person 1 and Person 2 that are touching.
Write all of these in a Markdown table where the first column is "Person 1 Body Part" and the second column is "Person 2 Body Part".
The list of possible body parts is: head, neck, chest, stomach, waist (back), waist (front), back, shoulder (back), shoulder (front), arm, hand, leg, foot, butt.
Do not include left/right. For waist/shoulder, think carefully about front vs. back.

List ALL pairs you are confident about.
If you are not confident about any pairs, output an empty table.
Carefully write your reasoning first, and then write the Markdown table.
"""

VISION_SYSTEM_PROMPT4 = \
"""You are a helpful assistant. You follow all directions correctly and precisely.
For each image, identify all pairs of body parts of Person 1 and Person 2 that are touching.
Write all of these in a Markdown table where the first column is "Person 1 Body Part" and the second column is "Person 2 Body Part".
The list of possible body parts is: head, neck, chest, stomach, back, shoulder (front), arm, hand, leg, foot, butt.
Do not include left/right. Note that "back" includes the entire back side of the torso.

List ALL pairs you are confident about.
If you are not confident about any pairs, output an empty table.
Carefully write your reasoning first, and then write the Markdown table.
"""

CHI3D_VISION_SYSTEM_PROMPT2_ACTIONS = \
"""You are a helpful assistant. You follow all directions correctly and precisely.
In each image, the two people are engaging in one of these actions: grab, handshake, hold hands, hug, kick, pose, or push.
For each image, first identify which action is occurring.
Then identify all pairs of body parts of Person 1 and Person 2 that are touching.
Write all of these in a Markdown table where the first column is "Person 1 Body Part" and the second column is "Person 2 Body Part".
You can pick which is Person 1 and which is Person 2.
The list of possible body parts is: head, neck, chest, stomach, waist (back), waist (front), back, shoulder (back), shoulder (front), arm, hand, leg, foot, butt.
Do not include left/right.

Only list pairs you are absolutely certain about.
If you are not certain about any pairs, output an empty table.
Carefully write your reasoning first, and then write the Markdown table.
"""

CHI3D_VISION_SYSTEM_PROMPT4_ACTIONS = \
"""You are a helpful assistant. You follow all directions correctly and precisely.
In each image, the two people are engaging in one of these actions: grab, handshake, hold hands, hug, kick, pose, or push.
For each image, first identify which action is occurring.
Then identify all pairs of body parts of Person 1 and Person 2 that are touching.
Write all of these in a Markdown table where the first column is "Person 1 Body Part" and the second column is "Person 2 Body Part".
The list of possible body parts is: head, neck, chest, stomach, back, shoulder (front), arm, hand, leg, foot, butt.
Do not include left/right. Note that "back" includes the entire back side of the torso.

List ALL pairs you are confident about.
If you are not confident about any pairs, output an empty table.
Carefully write your reasoning first, and then write the Markdown table."""

YOGA_VISION_SYSTEM_PROMPT1 = \
"""You are a helpful assistant. You answer all questions carefully and correctly.
Describe the yogi's pose.
Then list the body part(s) of the yogi that are touching the ground, one per line. You must choose from these:
```
head
neck
chest
stomach
back
hip
shoulder
arm
hand
knee
thigh
foot
butt
```
Only list a part if you're certain about it.
"""

YOGA_VISION_SYSTEM_PROMPT2 = \
"""You are a helpful assistant. You answer all questions carefully and correctly.
Identify which body parts of the yogi are directly touching each other in this image (if any).
Write each pair in a Markdown table with two columns.
Each body part MUST be from this list:
```
head
neck
chest
stomach
back
hip
shoulder
arm
hand
leg
foot
butt
ground
```
Do not write "left" or "right".

Only list a pair if you're confident about it.
If you are not confident on any pairs, output an empty table.
Carefully write your reasoning first, and then write the Markdown table.
"""

YOGA_VISION_SYSTEM_PROMPT3  = \
"""You are a helpful assistant. You answer all questions carefully and correctly.
Describe the yogi's pose.
Then list the body part(s) of the yogi that are touching the ground, one per line. You must choose from these:
```
head
chest
stomach
back
shoulder
arm
hand
knee
foot
butt
```
Use plural if both limbs are touching the ground for hand/foot/arm/knee.
Only list a part if you're certain about it.
"""

YOGA_VISION_SYSTEM_PROMPT4 = \
"""You are a helpful assistant. You answer all questions carefully and correctly.
Identify which body parts of the yogi are directly touching each other in this image (if any).
Write each pair in a Markdown table with two columns.
Each body part MUST be from this list:
```
head
back
shoulder
arm
hand
leg
foot
stomach
butt
ground
```
Do not write "left" or "right".
Use the plural form if the contact is true of both body parts (for hand/foot/arm/leg).

Only list a pair if you're confident about it.
If you are not confident on any pairs, output an empty table.
Carefully write your reasoning first, and then write the Markdown table."""

YOGA_VISION_SYSTEM_PROMPT5 = \
"""You are a helpful assistant. You answer all questions carefully and correctly.
Identify which yoga pose this is a variation of.
Then list the body part(s) of the yogi that are touching the ground in both this image and the standard version of the pose (the intersection of the two).
Write the list in a Markdown block with one part per line.
EACH PART MUST BE EXACTLY EQUAL TO ONE OF THESE:
```
head
chest
stomach
back
shoulder
elbow
hand
knee
foot
butt
```
Use plural if both limbs are touching the ground for hand/foot/elbow/knee.
Only list a part if you're certain about it.
"""

YOGA_VISION_SYSTEM_PROMPT6 = \
"""You are a helpful assistant. You answer all questions carefully and correctly.
Identify which body parts of the yogi are directly touching each other in this image (if any).
Write each pair in a Markdown table with two columns.
Each body part MUST be from this list:
```
head
back
shoulder
arm
hand
leg
foot
stomach
butt
ground
```
Do not write "left" or "right".

Only list a pair if you're certain about it.
If you are not certain on any pairs, output an empty table.
Carefully describe the yoga pose first, and then write the Markdown table."""

YOGA_VISION_SYSTEM_PROMPT7 = \
"""You are a helpful assistant. You answer all questions carefully and correctly.
Identify which body part(s) of the yogi are directly touching the ground.
Write the list in a Markdown block with one part per line.
EACH PART MUST BE EXACTLY EQUAL TO ONE OF THESE:
```
head
stomach
back
shoulder
elbow
hand
knee
foot
butt
```
Do not include "left" or "right".

Describe and name the yoga pose, and then write the Markdown list with the header "Parts On Ground".
Note that the pose may differ from the standard version, so pay close attention.
Only list a part if you're certain about it.
"""

YOGA_VISION_SYSTEM_PROMPT8 = \
"""You are a helpful assistant. You answer all questions carefully and correctly.
Identify which body parts of the yogi are touching each other in this image (if any).
Write each pair in a Markdown table with two columns.
Each body part MUST be from this list:
```
head
back
shoulder
arm
hand
leg
foot
stomach
butt
ground
```
Do not write "left" or "right".

Describe and name the yoga pose, and then write the Markdown table.
Note that the pose may differ from the standard version, so pay close attention.
Only list a part if you're certain about it."""

YOGA_VISION_SYSTEM_PROMPTS = {
    1: YOGA_VISION_SYSTEM_PROMPT1,
    2: YOGA_VISION_SYSTEM_PROMPT2,
    3: YOGA_VISION_SYSTEM_PROMPT3,
    4: YOGA_VISION_SYSTEM_PROMPT4,
    5: YOGA_VISION_SYSTEM_PROMPT5,
    6: YOGA_VISION_SYSTEM_PROMPT6,
    7: YOGA_VISION_SYSTEM_PROMPT7,
    8: YOGA_VISION_SYSTEM_PROMPT8,
}

REWRITE_CAPTION_PROMPT = \
"""Rewrite the caption below so that it doesn't mention "left" or "right" to describe any hand, arm, foot, or leg. The revised caption should otherwise be identical. Write only the revised caption and no other text.

"""

PROMPT1 = "Description: DESCRIPTION"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-path")
    parser.add_argument("--output-path")
    parser.add_argument("--gpt-model-name", default="gpt-4-1106-preview")
    parser.add_argument("--simple-gpt-model-name")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--prompt-choice")
    parser.add_argument("--single-person", action="store_true")
    parser.add_argument("--images-dir")
    parser.add_argument("--ext", choices=["png", "jpg"])
    args = parser.parse_args()
    if args.simple_gpt_model_name is None:
        args.simple_gpt_model_name = args.gpt_model_name
    client = openai.OpenAI()

    os.makedirs('/'.join(args.output_path.split('/')[:-1]), exist_ok=True)

    if args.keys_path is not None:
        with open(args.keys_path) as f:
            keys = json.load(f)
    outputs = {}
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path) as f:
                outputs = json.load(f)
        except json.decoder.JSONDecodeError:
            pass
    finished_keys = set(list(outputs.keys()))
    vision_prompts = {
        '1': VISION_SYSTEM_PROMPT1,
        '2': VISION_SYSTEM_PROMPT2,
        '2lr': VISION_SYSTEM_PROMPT2_LR,
        '2actions': CHI3D_VISION_SYSTEM_PROMPT2_ACTIONS,
        '3': VISION_SYSTEM_PROMPT3,
        '4': VISION_SYSTEM_PROMPT4,
        '4actions': CHI3D_VISION_SYSTEM_PROMPT4_ACTIONS,
    }
    for i in range(1, 9):
        vision_prompts['yoga'+str(i)] = YOGA_VISION_SYSTEM_PROMPTS[i]
    for key in tqdm(keys):
        print(key)
        if key in finished_keys:
            continue
        outputs[key] = []
        if args.vision:
            prompt = vision_prompts[args.prompt_choice]
            img = Image.open(os.path.join(args.images_dir, key+'.'+args.ext))
            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            try:
                gpt_response = client.chat.completions.create(
                    model=args.gpt_model_name,
                    messages=[
                        {'role': 'system', 'content': prompt},
                        {'role': 'user', 'content': [
                            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_str}', 'detail': 'high'}}
                        ]}
                    ],
                    temperature=0.7,
                    n=args.num_samples,
                    max_tokens=500
                )
            except (openai.BadRequestError, openai.InternalServerError):
                print('Bad request error', key)
                continue
            for choice in gpt_response.choices:
                output = {
                    'table_response': choice.message.content
                }
                outputs[key].append(output)
        else:
            examples = keys[key]
            for ex in examples:
                if isinstance(ex, str):
                    example = {'caption': ex}
                else:
                    assert isinstance(example, dict)
                print('CAPTION:', example['caption'])
                if 'rewritten_caption' not in example:
                    gpt_response = client.chat.completions.create(
                        model=args.simple_gpt_model_name,
                        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': REWRITE_CAPTION_PROMPT+example['caption']}],
                        temperature=0,
                        n=1,
                        max_tokens=200
                    )
                    rewritten_caption = gpt_response.choices[0].message.content.strip()
                    example['rewritten_caption'] = rewritten_caption
                print('REWRITTEN CAPTION:', example['rewritten_caption'])
                gpt_response = client.chat.completions.create(
                    model=args.gpt_model_name,
                    messages=[
                        {'role': 'system', 'content': (SYSTEM_PROMPT4_noperson1 if not args.single_person else YOGA_SYSTEM_PROMPT4)},
                        {'role': 'user', 'content': PROMPT1.replace('DESCRIPTION', example['rewritten_caption'])}
                    ],
                    temperature=0,
                    n=args.num_samples,
                    max_tokens=500
                )
                for choice in gpt_response.choices:
                    print('CHOICE', choice.message.content)
                    output = deepcopy(example)
                    output['table_response'] = choice.message.content
                    outputs[key].append(output)
        with open(args.output_path, 'w') as fout:
            json.dump(outputs, fout)
        finished_keys.add(key)
