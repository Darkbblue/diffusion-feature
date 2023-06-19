# diffusion-feature
Use pretrained Stable Diffusion model to extract features for various vision tasks.  

This repo is derived from Diffusion Classifier (https://github.com/diffusion-classifier/diffusion-classifier). I implement the "diffusion feature" mentioned in their paper appendix and used as a baseline by them. You can regard this code as a community implementation.  

**Although I have ensured this code can run, I haven't yet tested the extracted features on image classification tasks. If anyone wants to do the testing, you have my thanks.**

## Setup
```bash
conda create -n diffusion-classifier python=3.9

conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

conda install diffusers tqdm nvidia::cudatoolkit tensorboard flake8 ipykernel pytest seaborn

conda install -c xformers xformers
conda install -c nvidia cuda-nvcc
conda install -c huggingface transformers
conda install -c conda-forge accelerate ipdb pytest-env wandb
```

Or you can follow the instruction in the original repo, which does not work for me though.

## Usage
```bash
python3 main.py --t 100 --input_dir path/to/your/input/ --output_dir path/to/your/output/ --n_workers 3 --worker_idx 0
```

You can use tumex or screen to start multiple tasks and set their corresponding worker_idx (0, 1, ..., n-1) to enable multiprocess inference, which is strongly recommended.

## How I Implement it and How to Modify this Code
I copy the UNet2DConditionModel class from my installed diffusers package (anaconda3/envs/xxx/lib/python3.9/site-packages/diffusers/models/unet_2d_condition.py), and make a new class based on it. Actually I just copy the original forward function and return the tensor in the mid block instead.  

Then I modify main.py to enable feature extraction with batched computation.  

If you want to run this code, it may be wise to check your own diffusers package and make sure the modified UNet2DConditionModel is compatible with it. If not, it should be easy to make your own version in the same way as me. Additionally, you may want to modify how main.py load and save files to your preference.  
