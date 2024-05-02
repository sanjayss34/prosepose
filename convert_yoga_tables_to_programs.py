import argparse
from collections import defaultdict
from tqdm import tqdm
import json
from pattern.text.en import singularize

"""CANDIDATES = {
    'left shoulder', 'right shoulder', 'left arm', 'right arm', 'left hand', 'right hand', 'left leg', 'right leg', 'left foot', 'right foot', 'waist', 'back', 'stomach', 'chest', 'head', 'neck', 'butt'
}"""
CANDIDATES = {
    'left hand': [3, 33, 72],
    'right hand': [23, 26, 27, 40],
    'left arm': [5, 11, 17, 34, 49, 57, 69],
    'right arm': [6, 8, 24, 32, 37, 44],
    'left foot': [1, 67],
    'right foot': [42, 65],
    'left leg': [15, 20, 21, 25, 30, 45, 53, 56, 60, 64],
    'right leg': [12, 16, 18, 19, 22, 28, 36, 46, 55, 62],
    'back': [13, 35, 38, 41, 63, 70, 2, 29, 10, 48, 14, 43],
    'head': [39, 51, 52, 66],
    'neck': [14, 43, 47, 54],
    'butt': [9, 71],
    'waist': [10, 48, 50, 68, 74],
    'waist (back)': [10, 48],
    'waist (front)': [50, 68, 74],
    'left shoulder (front)': [73],
    'left shoulder (back)': [2],
    'right shoulder (front)': [31],
    'right shoulder (back)': [29],
    'left shoulder': [2, 73],
    'right shoulder': [29, 31],
    'chest': [7, 58, 59, 61],
    'stomach': [0, 4, 68, 74],
}
CANDIDATES = set(CANDIDATES.keys())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--constraint-threshold", type=float, default=10)
    args = parser.parse_args()

    with open(args.input_path) as f:
        examples = json.load(f)
    keys = list(set(examples.keys()))
    # if args.fix_hand_region:
    #     CANDIDATES['left hand'] = [3, 33, 72, 17]
    #     CANDIDATES['left arm'] = [5, 11, 34, 49, 57, 69]
    outputs = {}
    for key in tqdm(keys):
        program = "import torch\ndef loss(vertices1, VERTEX_LIST_MAP):\n    total_loss = torch.tensor(0.0)\n"
        divisor = 0
        if key in examples:
            lst = []
            constraints = defaultdict(float)
            pairs = []
            for example in examples[key]:
                lines = example['table_response'].split('\n')
                for line in lines:
                    if len(line.split('|')) >= 3:
                        part1 = line.lower().split('|')[1].strip()
                        part2 = line.lower().split('|')[2].strip()
                        part1_base = singularize(part1)
                        part2_base = singularize(part2)
                        if (part1_base in CANDIDATES or 'left '+part1_base in CANDIDATES) and (part2_base in CANDIDATES or 'left '+part2_base in CANDIDATES):
                            pairs.append(frozenset([part1_base, part2_base]))
                            constraints[pairs[-1]] += 1
            outputs[args.prefix+key+args.suffix] = [] # [{'code': program}]
            for example in examples[key]:
                basic_program = "import torch\ndef loss(vertices1, VERTEX_LIST_MAP):\n    return torch.tensor(0.0)"
                program = "import torch\ndef loss(vertices1, VERTEX_LIST_MAP):\n    total_loss = 0.0\n"
                num_lines = len(program.split('\n'))
                lines = example['table_response'].split('\n')
                count = 0
                for line in lines:
                    if len(line.split('|')) >= 3:
                        part1 = line.lower().split('|')[1].strip()
                        part2 = line.lower().split('|')[2].strip()
                        part1_base = singularize(part1)
                        part2_base = singularize(part2)
                        if (part1_base in CANDIDATES or 'left '+part1_base in CANDIDATES) and (part2_base in CANDIDATES or 'left '+part2_base in CANDIDATES):
                            pair = frozenset([part1_base, part2_base])
                            if constraints[pair] >= args.constraint_threshold:
                                part1 = [list(pair)[0]]
                                if len(pair) == 1:
                                    part2 = [list(pair)[0]]
                                else:
                                    part2 = [list(pair)[1]]
                                if 'left '+part1[0] in CANDIDATES:
                                    part1 = ['left '+part1[0], 'right '+part1[0]]
                                if 'left '+part2[0] in CANDIDATES:
                                    part2 = ['left '+part2[0], 'right '+part2[0]]
                                program += f'    loss_term{count} = torch.stack([\n'
                                for p1 in part1:
                                    for p2 in part2:
                                        if p1 == p2:
                                            continue
                                        program += f'        min_distance(vertices1[VERTEX_LIST_MAP[\"{p1}\"]], vertices1[VERTEX_LIST_MAP[\"{p2}\"]]),\n'
                                program += '    ]).min()\n'
                                program += f'    total_loss += loss_term{count}\n'
                                count += 1
                if len(program.split('\n')) == num_lines:
                    program = basic_program
                else:
                    program += '    return total_loss'
                outputs[args.prefix+key+args.suffix].append({'code': program})
        # example = examples[key][0]
        # example['code'] = program
    with open(args.output_path, 'w') as fout:
        json.dump(outputs, fout)
