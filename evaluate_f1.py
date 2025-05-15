import json
import numpy as np
import cv2
import argparse
import sys
import os
import pickle as pkl
from tqdm import tqdm
from PIL import Image

fine_grained_names = {
    0: "left upper belly",
    1: "left foot",
    2: "back left shoulder",
    3: "left fingers",
    4: "front right upper belly",
    5: "front left forearm",
    6: "back right upper arm",
    7: "right upper chest",
    8: "back right forearm",
    9: "right butt",
    10: "right back",
    11: "back left elbow",
    12: "back right knee",
    13: "right middle back",
    14: "back left neck",
    15: "back left upper thigh",
    16: "front right knee",
    17: "left hand",
    18: "back right lower leg",
    19: "front right upper thigh",
    20: "front left lower thigh",
    21: "back left lower leg",
    22: "front right shin",
    23: "right palm",
    24: "back right elbow",
    25: "front left shin",
    26: "right fingers",
    27: "right hand",
    28: "back right lower thigh",
    29: "back right shoulder",
    30: "back left knee",
    31: "front right shoulder",
    32: "front right forearm",
    33: "left fingers",
    34: "back left elbow",
    35: "left middle back",
    36: "back right shin",
    37: "front right upper arm",
    38: "right upper back",
    39: "back left skull",
    40: "right fingers",
    41: "left upper back",
    42: "right foot",
    43: "back right neck",
    44: "front right elbow",
    45: "front left upper thigh",
    46: "front right upper thigh",
    47: "front right neck",
    48: "left lower back",
    49: "front left upper arm",
    50: "front pelvis",
    51: "front right skull",
    52: "back right skull",
    53: "back left lower thigh",
    54: "front left neck",
    55: "back right upper thigh",
    56: "back left shin",
    57: "back left upper arm",
    58: "right upper chest",
    59: "left upper chest",
    60: "front left lower leg",
    61: "left lower chest",
    62: "front right lower leg",
    63: "right upper back",
    64: "front left knee",
    65: "right foot",
    66: "front left skull",
    67: "left foot",
    68: "right lower belly",
    69: "back left forearm",
    70: "left upper back",
    71: "left butt",
    72: "left palm",
    73: "front left shoulder",
    74: "left lower belly",
}
inv_coarse_region_names = {
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
coarse_region_names = {
  i: [key for key in inv_coarse_region_names if i in inv_coarse_region_names[key]][0]
  for i in fine_grained_names
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-name")
    parser.add_argument("--ignore-left-right", action="store_true")
    parser.add_argument("--ignore-order", action="store_true")
    parser.add_argument("--tables-path")
    parser.add_argument("--table-prefix", default="")
    parser.add_argument("--table-suffix", default="")
    parser.add_argument("--gt-path")
    parser.add_argument("--contact-threshold", type=float, default=0.01)
    parser.add_argument("--match-overlapping-regions", action="store_true")
    parser.add_argument("--path-to-results-pkl")
    parser.add_argument("--single-person", action="store_true")
    parser.add_argument("--output-path")
    args = parser.parse_args()
    
    with open(args.tables_path) as f:
      tables = json.load(f)
      if len(args.table_suffix) > 0:
        tables = {key[len(args.table_prefix):-len(args.table_suffix)]: tables[key] for key in tables}
    with open(args.gt_path) as f:
      gt_data = [json.loads(line) for line in f.readlines()]
    names_to_include = []
    if args.path_to_results_pkl is not None:
      with open(args.path_to_results_pkl, 'rb') as f:
        res_pkl = pkl.load(f)
      names_to_include = [name[len(args.table_prefix):-len(args.table_suffix)] if len(args.table_suffix) > 0 else name for name in res_pkl['info']['img_names']]
    for datum in gt_data:
      assert datum['question_id'] in tables, datum['question_id']
    predictions_dict = {}
    for key in tables:
      predictions_dict[key] = []
      for response in tables[key]:
        lines = response['table_response'].split('\n')
        coarse_pairs = set()
        for line in lines:
          if line.count('|') == 3:
            parts = [part.lower().strip() for part in line.split('|')]
            regions = []
            for part in parts[1:3]:
              if part in inv_coarse_region_names:
                regions.append(part)
              elif args.ignore_left_right and 'left '+part in inv_coarse_region_names:
                regions.append(part)
            if len(regions) == 2:
              coarse_pairs.add(tuple(regions))
        predictions_dict[key].append(coarse_pairs)

    micro_f1s = []
    macro_f1s = []
    micro_precisions = []
    macro_precisions = []
    micro_recalls = []
    macro_recalls = []
    ids = []
    for datum in gt_data:
      if len(names_to_include) > 0 and datum['question_id'] not in names_to_include:
        continue
      coarse_pairs = set()
      sentences = [sent.strip() for sent in datum['answer'].split('.') if len(sent.strip()) > 0]
      for sent in sentences:
        if 'is touching' in sent:
          coarse1 = sent.split('is touching')[0].split('Person 1\'s')[1].strip()
          coarse2 = sent.split('Person 2\'s')[1].strip()
          if args.ignore_left_right:
            if coarse1.split()[0] in {'left', 'right'}:
              coarse1 = coarse1.split(coarse1.split()[0])[1].strip()
            if coarse2.split()[0] in {'left', 'right'}:
              coarse2 = coarse2.split(coarse2.split()[0])[1].strip()
          coarse_pairs.add((coarse1, coarse2))
      macro_f1s.append([])
      macro_precisions.append([])
      macro_recalls.append([])
      if len(predictions_dict[datum['question_id']]) == 0:
        macro_f1s[-1].append(0)
        micro_f1s.append(0)
        macro_precisions[-1].append(0)
        micro_precisions.append(0)
        macro_recalls[-1].append(0)
        micro_recalls.append(0)
      for pred_set in predictions_dict[datum['question_id']]:
        best_f1 = 0
        best_f1_precision = 0
        best_f1_recall = 0
        orderings = [[0, 1]]
        if args.ignore_order:
          orderings = [[0, 1], [1, 0]]
        for o in orderings:
          pred_set_o = set([(pair[o[0]], pair[o[1]]) for pair in pred_set])
          num_precision_good = 0
          num_recall_good = 0
          for pair1 in pred_set_o:
            for pair2 in (coarse_pairs.union(set([(pair[1], pair[0]) for pair in coarse_pairs])) if args.single_person else coarse_pairs):
              if any([any([region in inv_coarse_region_names[name2] for region in inv_coarse_region_names[name1]]) for name1 in ['left '+pair1[0], 'right '+pair1[0], pair1[0]] for name2 in ['left '+pair2[0], 'right '+pair2[0], pair2[0]] if name1 in inv_coarse_region_names and name2 in inv_coarse_region_names]):
                if any([any([region in inv_coarse_region_names[name2] for region in inv_coarse_region_names[name1]]) for name1 in ['left '+pair1[1], 'right '+pair1[1], pair1[1]] for name2 in ['left '+pair2[1], 'right '+pair2[1], pair2[1]] if name1 in inv_coarse_region_names and name2 in inv_coarse_region_names]):
                  num_precision_good += 1
                  break
          for pair1 in coarse_pairs:
            for pair2 in (pred_set_o.union(set([(pair[1], pair[0]) for pair in pred_set_o])) if args.single_person else pred_set_o):
              if any([any([region in inv_coarse_region_names[name2] for region in inv_coarse_region_names[name1]]) for name1 in ['left '+pair1[0], 'right '+pair1[0], pair1[0]] for name2 in ['left '+pair2[0], 'right '+pair2[0], pair2[0]] if name1 in inv_coarse_region_names and name2 in inv_coarse_region_names]):
                if any([any([region in inv_coarse_region_names[name2] for region in inv_coarse_region_names[name1]]) for name1 in ['left '+pair1[1], 'right '+pair1[1], pair1[1]] for name2 in ['left '+pair2[1], 'right '+pair2[1], pair2[1]] if name1 in inv_coarse_region_names and name2 in inv_coarse_region_names]):
                  num_recall_good += 1
                  break
          common = pred_set_o.intersection(coarse_pairs)
          if len(pred_set_o) > 0:
            if args.match_overlapping_regions:
              precision = num_precision_good / len(pred_set_o)
            else:
              precision = len(common) / len(pred_set_o)
          else:
            precision = 0
          if len(coarse_pairs) > 0:
            if args.match_overlapping_regions:
              recall = num_recall_good / len(coarse_pairs)
            else:
              recall = len(common) / len(coarse_pairs)
          else:
            recall = 0
          if precision+recall > 0:
            f1 = 2 * precision * recall / (precision+recall)
          elif len(coarse_pairs)+len(pred_set_o) == 0:
            precision = 1
            recall = 1
            f1 = 1
          else:
            f1 = 0
          if f1 > best_f1:
            best_f1_precision = precision
            best_f1_recall = recall
          best_f1 = max(best_f1, f1)
          assert not np.isnan(best_f1)
        if best_f1_precision < 1 and best_f1 > 0.999:
          print(best_f1_precision, best_f1_recall, best_f1)
        micro_f1s.append(best_f1)
        macro_f1s[-1].append(best_f1)
        micro_precisions.append(best_f1_precision)
        macro_precisions[-1].append(best_f1_precision)
        micro_recalls.append(best_f1_recall)
        macro_recalls[-1].append(best_f1_recall)
      ids.append(datum['question_id'])
    print('Micro F1:', np.mean(micro_f1s))
    print([(i, len(lst)) for lst, i in zip(macro_f1s, ids) if len(lst) < 20])
    print('Macro F1:', np.mean([np.mean(lst) for lst in macro_f1s]))
    print('Micro Precision:', np.mean(micro_precisions))
    print('Micro Recall:', np.mean(micro_recalls))
    if args.output_path is not None:
        with open(args.output_path, 'w') as fout:
            json.dump({img_name: np.mean(lst) for img_name, lst in zip(ids, macro_f1s)}, fout)
        with open(args.output_path.replace('f1s', 'precisions'), 'w') as fout:
            json.dump({img_name: np.mean(lst) for img_name, lst in zip(ids, macro_precisions)}, fout)
