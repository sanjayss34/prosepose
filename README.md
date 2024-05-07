# ProsePose
<b> Pose Priors from Language Models </b>\
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://prosepose.github.io) [![arXiv](https://img.shields.io/badge/arXiv-2405.03689-00ff00.svg)](https://arxiv.org/abs/2405.03689)

This is the repository for the paper [Pose Priors from Language Models](https://arxiv.org/abs/2405.03689), which demonstrates that (multimodal) language models encode information about human pose that we can operationalize to improve 3D pose reconstruction.

## Data
Download the necessary files from BUDDI and pre-processed data by running `bash fetch_data.sh`. The script requires [gdown](https://pypi.org/project/gdown/). The dataset structure should look like this:
```
|-- datasets/
    |-- original/
        |-- FlickrCI3D_Signatures/
        |-- CHI3D/
        |-- Hi4D/
        |-- moyo_cam05_centerframe/
    |-- processed/
        |-- FlickrCI3D_Signatures/
        |-- CHI3D/
        |-- Hi4D/
        |-- moyo_cam05_centerframe/
```
You will also need to download the original datasets from the dataset providers (e.g. to access the raw images):
[FlickrCI3D](https://ci3d.imar.ro/flickrci3d)
[CHI3D](https://ci3d.imar.ro/chi3d)
[Hi4D](https://yifeiyin04.github.io/Hi4D/#dataset)
[MOYO](https://moyo.is.tue.mpg.de/)
Put the contents of each dataset download in the `original` subdirectory corresponding to that dataset (e.g. `datasets/original/FlickrCI3D_Signatures/`).

## Installation
Please install [Anaconda/Miniconda](https://docs.anaconda.com/free/miniconda/index.html).

We provide a packed conda environment. It will be downloaded when you run the script mentioned in the previous section. You can then activate it as follows:
```
tar -xvzf prosepose_env.tar.gz
source prosepose_env/bin/activate
```

## Preprocessing
To save cropped versions of images for each dataset, run:
```
python prepare_flickrci3d_images.py
python prepare_chi3d_images.py
python prepare_hi4d_images.py
python prepare_moyo_images.py
```
Please look at the scripts to see the default image directory (for the original images) or specify the path to your image directory.

## Running ProsePose
There are three steps in our pipeline: (1) Generating constraints (using a large multimodal model), (2) Converting the constraints into loss functions, and (3) Running optimization using the constraints.
We also provide our JSON files of generated loss functions in `datasets/processed/*/` if you'd like to run only Step 3.
### Generating constraints and loss functions
To generate constraints with GPT4-V for each dataset, you can run the following (for the validation sets):
```
python gpt_table.py --vision --keys-path datasets/processed/Hi4D/val_keys.json --output-path gpt4v_hi4d_val/table_20samples.json --num-samples 20 --prompt-choice 2 --images-dir datasets/processed/Hi4D/cropped_images --gpt-model-name gpt-4-vision-preview --ext jpg
python gpt_table.py --vision --keys-path datasets/processed/FlickrCI3D_Signatures/val_keys.json --output-path gpt4v_flickrci3d_val/table_20samples.json --num-samples 20 --prompt-choice 2 --images-dir datasets/processed/FlickrCI3D_Signatures/cropped_images --gpt-model-name gpt-4-vision-preview --ext png
python gpt_table.py --vision --keys-path datasets/processed/CHI3D/trainval_keys.json --output-path gpt4v_chi3d_trainval/table_20samples.json --num-samples 20 --prompt-choice 2 --images-dir datasets/processed/CHI3D/cropped_images --gpt-model-name gpt-4-vision-preview --ext jpg
python gpt_table.py --vision --keys-path datasets/processed/moyo_cam05_centerframe/trainval_keys.json --output-path gpt4v_moyo_trainval/table_20samples.json --num-samples 20 --prompt-choice yoga8 --images-dir datasets/processed/moyo_cam05_centerframe/cropped_images --gpt-model-name gpt-4-vision-preview --ext jpg
```
To convert those constraints into loss functions, you can run:
```
python convert_tables_to_programs.py --input-path gpt4v_hi4d_val/table_20samples.json --output-path gpt4v_hi4d_val/programs_from_prompt2.json --suffix 0
python convert_tables_to_programs.py --input-path gpt4v_flickrci3d_val/table_20samples.json --output-path gpt4v_flickrci3d_val/programs_from_prompt2.json
python convert_tables_to_programs.py --input-path gpt4v_chi3d_trainval/table_20samples.json --output-path gpt4v_chi3d_trainval/programs_from_prompt2.json --suffix _0
python convert_yoga_tables_to_programs.py --input-path gpt4v_moyo_trainval/table_20samples.json --output-path gpt4v_moyo_trainval/programs_from_prompt8.json --constraint-threshold 10
```
Note: the `constraint-threshold` in the last line corresponds to the threshold f in the paper.

If you get an error about downloading nltk when running one of the conversion scripts and you are using an Ubuntu system, running these should fix the issue:
```
sudo update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs/
```

### Optimization with Predicted Constraints
To optimize pose reconstructions using the generated loss functions, you can run:
```
python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/prosepose.yaml --start-idx 0 --exp-opts logging.base_folder=flickrci3d_val/prosepose_gpt4v/optimization logging.run=bev_gpt4v_prompt2_20samples datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['flickrci3ds'] datasets.test_names=[]
python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/prosepose.yaml --start-idx 0 --exp-opts logging.base_folder=hi4d_val/prosepose_gpt4v/optimization logging.run=bev_gpt4v_prompt2_20samples datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['hi4d'] datasets.test_names=[]
python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/prosepose.yaml --start-idx 0 --exp-opts logging.base_folder=chi3d_trainval/prosepose_gpt4v/optimization logging.run=bev_gpt4v_prompt2_20samples datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['chi3d'] datasets.test_names=[]
python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/prosepose_moyo.yaml --start-idx 0 --exp-opts logging.base_folder=moyo_trainval/prosepose_gpt4v/optimization logging.run=gpt4v_prompt8_20samples
```

## Running Baselines
You can run baselines just like in the [BUDDI](https://github.com/muelea/buddi) repo:
```
python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/heuristic_02.yaml --start-idx 0 --exp-opts logging.base_folder=flickrci3d_val/heuristic_02/optimization logging.run=heuristic_02 datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['flickrci3ds'] datasets.test_names=[]
python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev.yaml --start-idx 0 --exp-opts logging.base_folder=flickrci3d_val/buddi_cond_bev/optimization logging.run=buddi_cond_bev datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['flickrci3ds'] datasets.test_names=[]

python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/heuristic_02.yaml --start-idx 0 --exp-opts logging.base_folder=hi4d_val/heuristic_02/optimization logging.run=heuristic_02 datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['hi4d'] datasets.test_names=[]
python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev.yaml --start-idx 0 --exp-opts logging.base_folder=hi4d_val/buddi_cond_bev/optimization logging.run=buddi_cond_bev datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['hi4d'] datasets.test_names=[]

python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/heuristic_02.yaml --start-idx 0 --exp-opts logging.base_folder=chi3d_trainval/heuristic_02/optimization logging.run=heuristic_02 datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['chi3d'] datasets.test_names=[]
python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev.yaml --start-idx 0 --exp-opts logging.base_folder=chi3d_trainval/buddi_cond_bev/optimization logging.run=buddi_cond_bev datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['chi3d'] datasets.test_names=[]

python -m llib.methods.hhcs_optimization.main --exp-cfg llib/methods/hhcs_optimization/configs/moyo_baseline.yaml --start-idx 0 --exp-opts logging.base_folder=moyo_trainval/hmr2_opt/optimization logging.run=hmr2_opt
```

## Evaluation
You can use these command templates to evaluate a method. The `--backoff-predictions-folder` is only necessary for our method, as the other methods don't involve a backoff case.
```
python -m llib.methods.hhcs_optimization.evaluation.hi4d_eval --exp-cfg llib/methods/hhcs_optimization/evaluation/hi4d_eval.yaml --predictions-folder PATH_TO_METHOD_DIR --eval-split val --print_result --backoff-predictions-folder PATH_TO_HEURISTIC_02_DIR
python -m llib.methods.hhcs_optimization.evaluation.flickrci3ds_eval --exp-cfg llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.yaml -gt PATH_TO_PSEUDO_GT_DIR -p PATH_TO_METHOD_DIR --backup-predicted-folder PATH_TO_HEURISTIC_02_DIR --flickrci3ds-split val
python -m llib.methods.hhcs_optimization.evaluation.chi3d_eval --exp-cfg llib/methods/hhcs_optimization/evaluation/chi3d_eval.yaml --predictions-folder PATH_TO_METHOD_DIR --eval-split val --print_result --backoff-predictions-folder PATH_TO_HEURISTIC_02_DIR --programs-path CUSTOM_LOSS_FUNCTIONS_PATH --threshold 2
python -m llib.methods.hhcs_optimization.evaluation.moyo_eval --exp-cfg llib/methods/hhcs_optimization/evaluation/moyo_eval.yaml --predictions-folder PATH_TO_METHOD_DIR --eval-split train --print_result --backoff-predictions-folder PATH_TO_HMR2_OPTIM_DIR
```
Note: the threshold for the CHI3D command corresponds to the threshold t in the paper.

# Citation
```
@article{subramanian2024pose,
    title={Pose Priors from Language Models},
    author={Subramanian, Sanjay and Ng, Evonne and M{\“u}ller, Lea and Klein, Dan and Ginosar, Shiry and Darrell, Trevor},
    journal={arXiv preprint TODO},
    year={2024}}
```

# Acknowledgements
This repository is adapted from [BUDDI](https://github.com/muelea/buddi).
