import argparse
from collections import defaultdict
import itertools
from tqdm import tqdm
import json
from copy import deepcopy
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
    'stomach': [0, 4, 68, 74]
}
CANDIDATES = set(CANDIDATES.keys())

def get_part_list(term, left_right_specified, delimiter):
    if not left_right_specified or len(term) == 0:
        return term.split(delimiter)
    parts = []
    initial_parts = term.split(delimiter)
    prefix = ""
    if term.split()[0] in {'left', 'right'}:
        initial_parts = ' '.join(term.split()[1:]).split(delimiter)
        prefix = term.split()[0]+' '
    for part in initial_parts:
        if prefix+part in CANDIDATES:
            parts.append(prefix+part)
        elif part in CANDIDATES:
            parts.append(part)
        else:
            if part.split()[0] in {'left', 'right'}:
                if ' '.join(part.split()[1:]) in CANDIDATES:
                    parts.append(' '.join(part.split()[1:]))
            elif part.split('(')[0].strip() in CANDIDATES:
                parts.append(part.split('(')[0].strip())
    return parts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--delimiter", default="/")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--max-program-length", default=None, type=int)
    parser.add_argument("--not-both-orders", action="store_true")
    args = parser.parse_args()

    with open(args.input_path) as f:
        examples = json.load(f)
    print('Initial length:', len(examples))
    outputs = {}
    for key in tqdm(examples):
        lst = []
        for example in examples[key]:
            pairs = []
            pairs_sans_lr = []
            if isinstance(example, str):
                example = {'table_response': example}
            # print(example['table_response'])
            lines = example['table_response'].split('\n')
            for line in lines:
                if len(line.split('|')) >= 3:
                    term1 = line.lower().split('|')[1].strip()
                    term2 = line.lower().split('|')[2].strip()
                    parts1 = get_part_list(term1, False, args.delimiter)
                    parts2 = get_part_list(term2, False, args.delimiter)
                    line_pairs = []
                    line_pairs_sans_lr = []
                    for part1 in parts1:
                        if len(part1) == 0:
                            continue
                        lr_part1 = False
                        plural1 = False
                        part1_base = part1
                        if singularize(part1) != part1 and 'left '+singularize(part1) in CANDIDATES:
                            lr_part1 = True
                            plural1 = True
                            part1_base = singularize(part1)
                        elif 'left '+part1 in CANDIDATES:
                            lr_part1 = True
                        elif part1_base not in CANDIDATES:
                            if part1.split('(')[0].strip() in CANDIDATES:
                                part1_base = part1 = part1.split('(')[0].strip()
                            continue
                        for part2 in parts2:
                            if len(part2) == 0:
                                continue
                            lr_part2 = False
                            plural2 = False
                            part2_base = part2
                            if singularize(part2) != part2 and 'left '+singularize(part2) in CANDIDATES:
                                lr_part2 = True
                                plural2 = True
                                part2_base = singularize(part2)
                            elif 'left '+part2 in CANDIDATES:
                                lr_part2 = True
                            elif part2_base not in CANDIDATES:
                                if part2.split('(')[0].strip() in CANDIDATES:
                                    part2_base = part2 = part2.split('(')[0].strip()
                                continue
                            if plural1:
                                if plural2 or lr_part2:
                                    line_pairs.append([(prefix1+' '+part1_base, prefix2+' '+part2_base) for prefix1, prefix2 in zip(['left', 'left'], ['left', 'right'])])
                                    line_pairs.append([(prefix1+' '+part1_base, prefix2+' '+part2_base) for prefix1, prefix2 in zip(['right', 'right'], ['left', 'right'])])
                                else:
                                    line_pairs.append([('left '+part1_base, part2)])
                                    line_pairs.append([('right '+part1_base, part2)])
                            elif plural2:
                                if lr_part1:
                                    line_pairs.append([(prefix1+' '+part1_base, prefix2+' '+part2_base) for prefix1, prefix2 in zip(['left', 'right'], ['left', 'left'])])
                                    line_pairs.append([(prefix1+' '+part1_base, prefix2+' '+part2_base) for prefix1, prefix2 in zip(['left', 'right'], ['right', 'right'])])
                                else:
                                    line_pairs.append([(part1, 'left '+part2_base)])
                                    line_pairs.append([(part1, 'right '+part2_base)])
                            elif lr_part1 and lr_part2:
                                line_pairs.append([(prefix1+' '+part1_base, prefix2+' '+part2_base) for prefix1 in ['left', 'right'] for prefix2 in ['left', 'right']])
                            elif lr_part1:
                                line_pairs.append([(prefix1+' '+part1_base, part2_base) for prefix1 in ['left', 'right']])
                            elif lr_part2:
                                line_pairs.append([(part1_base, prefix2+' '+part2_base) for prefix2 in ['left', 'right']])
                            else:
                                line_pairs.append([(part1_base, part2_base)])
                            line_pairs_sans_lr.append([part1_base, part2_base])
                            if plural1 or plural2:
                                line_pairs_sans_lr.append([part1_base, part2_base])
                    pairs += line_pairs
                    pairs_sans_lr += line_pairs_sans_lr
            default_program = "import torch\ndef loss(vertices1, vertices2, VERTEX_LIST_MAP):\n    return torch.tensor(0.0)"
            program = default_program
            over_limit = False
            if len(pairs) > 0:
                program = "import torch\ndef loss(vertices1, vertices2, VERTEX_LIST_MAP):\n    total_loss = torch.tensor(0.0, dtype=vertices1.dtype, device=vertices1.device)\n"
                lr_possibilities = ['left', 'right']
                possible_losses = []
                base_names_lr = [part.split('left ')[1] for part in CANDIDATES if part[:5] == 'left ']
                base_names1 = []
                base_names2 = []
                counts1 = defaultdict(int)
                counts2 = defaultdict(int)
                pair_counts = defaultdict(int)
                for pair in pairs_sans_lr:
                    if pair[0] in base_names_lr:
                        base_names1.append(pair[0]+'|'+str(counts1[pair[0]]))
                        counts1[pair[0]] += 1
                    else:
                        assert pair[0] in CANDIDATES
                        base_names1.append(pair[0])
                    if pair[1] in base_names_lr:
                        base_names2.append(pair[1]+'|'+str(counts2[pair[1]]))
                        counts2[pair[1]] += 1
                    else:
                        assert pair[1] in CANDIDATES
                        base_names2.append(pair[1])
                    pair_counts[(base_names1[-1].split('|')[0], base_names2[-1].split('|')[0])] += 1

                if args.max_program_length is not None and 2**max(sum(counts1.values()), sum(counts2.values())) > args.max_program_length:
                  over_limit = True
                else:
                  combinations1 = list(itertools.product(["left", "right"], repeat=sum(counts1.values())))
                  combinations2 = list(itertools.product(["left", "right"], repeat=sum(counts2.values())))
                  lr_parts1 = list(counts1.keys())
                  lr_parts2 = list(counts2.keys())
                  pair_losses = {}
                  for combo1 in combinations1:
                      full_names1 = []
                      index1 = 0
                      for name in base_names1:
                          if '|' in name:
                              full_names1.append(combo1[index1]+' '+name.split('|')[0])
                              index1 += 1
                          else:
                              full_names1.append(name)
                      assert all([name in CANDIDATES for name in full_names1])
                      if any([counts1[part] > 1 and ('left '+part not in full_names1 or 'right '+part not in full_names1) for part in base_names_lr]):
                          continue
                      for combo2 in combinations2:
                          full_names2 = []
                          index2 = 0
                          for name in base_names2:
                              if '|' in name:
                                  full_names2.append(combo2[index2]+' '+name.split('|')[0])
                                  index2 += 1
                              else:
                                  full_names2.append(name)
                          assert all([name in CANDIDATES for name in full_names2])
                          if any([counts2[part] > 1 and ('left '+part not in full_names2 or 'right '+part not in full_names2) for part in base_names_lr]):
                              continue
                          for part1, part2 in zip(full_names1, full_names2):
                              if (part1, part2) not in pair_losses:
                                  pair_losses[(part1, part2)] = f'loss_{part1.replace(" ", "_").replace("(", "").replace(")", "")}_{part2.replace(" ", "_").replace("(", "").replace(")", "")}'
                                  program += f'    {pair_losses[(part1, part2)]} = min_distance(vertices1[VERTEX_LIST_MAP[\"{part1}\"]], vertices2[VERTEX_LIST_MAP[\"{part2}\"]])\n'
                          program += f'    loss_term_{len(possible_losses)} = '+"+".join([pair_losses[(part1, part2)] for part1, part2 in zip(full_names1, full_names2)])+'\n'
                          # program += f'    loss_term_{len(possible_losses)} = '+"+".join([f"min_distance(vertices1[VERTEX_LIST_MAP[\"{part1}\"]], vertices2[VERTEX_LIST_MAP[\"{part2}\"]])" for part1, part2 in zip(full_names1, full_names2)])+'\n'
                          possible_losses.append(f'loss_term_{len(possible_losses)}')
                          if args.max_program_length is not None and len(program) > args.max_program_length:
                              break
                      if args.max_program_length is not None and len(program) > args.max_program_length:
                          break
                  program += '    total_loss = torch.stack(['+', '.join(possible_losses)+']).min()\n'
                program += '    return total_loss'
            example['code'] = program
            if args.max_program_length is not None and (over_limit or len(program) > args.max_program_length):
                example['code'] = default_program
            lst.append(example)
        outputs[args.prefix+key+args.suffix] = lst
    # Condense the list of programs into a single program for each example
    both_orders = not args.not_both_orders
    outputs2 = {}
    for key in outputs:
        new_program = "import torch\ndef loss(vertices1, vertices2, VERTEX_LIST_MAP):\n    total_loss = torch.tensor(0.0, dtype=vertices1.dtype)\n"
        loss_pairs = set()
        end_lines = ""
        for i, datum in enumerate(outputs[key]):
            lines = datum['code'].split('\n')
            for line in lines:
                if line.strip()[:5] == 'loss_' and line.strip()[:9] != 'loss_term':
                    line1 = line.replace('loss_', 'lossa_')
                    line2 = line.replace('vertices1', 'vertices3').replace('vertices2', 'vertices1').replace('vertices3', 'vertices2').replace('loss_', 'lossb_')
                    loss_pairs.add(line1+'\n')
                    if both_orders:
                      loss_pairs.add(line2+'\n')
                elif line.strip()[:9] == 'loss_term':
                    end_lines += f'    program{i}_'+line.strip().replace('loss_', 'lossa_')+'\n'
                    if both_orders:
                      end_lines += f'    program{i}_'+line.strip().replace('loss_', 'lossb_')+'\n'
                elif line.strip()[:10] == 'total_loss':
                    end_lines += line.replace('loss_term', f'program{i}_lossa_term').replace('total_loss ', f'program{i}_total_lossa ')+'\n'
                    if both_orders:
                      end_lines += line.replace('loss_term', f'program{i}_lossb_term').replace('total_loss ', f'program{i}_total_lossb ')+'\n'
                      end_lines += f"    total_loss += min(program{i}_total_lossa, program{i}_total_lossb)\n"
                    else:
                      end_lines += f"    total_loss += program{i}_total_lossa\n"
        for pair in sorted(list(loss_pairs)):
            new_program += pair
        new_program += end_lines
        new_program += f"    return total_loss / {len(outputs[key])}"
        outputs2[key] = [{'code': new_program}]
        for k in outputs[key][0]:
            if k != 'code':
                outputs2[key][0][k] = [datum[k] for datum in outputs[key]]
        assert outputs2[key][0]['code'] == new_program
    outputs = outputs2
    print('Final length:', len(outputs))
    with open(args.output_path, 'w') as fout:
        json.dump(outputs, fout)
