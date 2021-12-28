#!/usr/bin/env bash

# dark roof
# CUDA_VISIBLE_DEVICES=1 python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --input ~/data/dark_roof_7611_copy/ --tile 1200 --output ~/data/dark_roof_7611_results --outscale 2 --suffix ""

# construction site
# CUDA_VISIBLE_DEVICES=0 python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --input ~/data/homedepot.81989_sample100/ --tile 1200 --output ~/data/homedepot.81989_results/ --outscale 2 --suffix ""

# outdoor
# CUDA_VISIBLE_DEVICES=2 python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --input ~/data/domebuilds.80486_sample100/ --tile 1200 --output ~/data/domebuilds.80486_results --outscale 2 --suffix ""


## Start of parameter test launches

# cpu-x2plus-scale1
# python inference_realesrgan.py --input ~/data/realesrgan_parameters_test/original/ --usecpu --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --outscale 1 --output ~/data/realesrgan_parameters_test/cpu-x2plus-scale1 --suffix ""

# cpu-x2plus-scale1-half
# python inference_realesrgan.py --input ~/data/realesrgan_parameters_test/original/ --usecpu --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --outscale 1 --half --output ~/data/realesrgan_parameters_test/cpu-x2plus-scale1-half --suffix ""

# gpu-x2plus-scale1-half
# python inference_realesrgan.py --input ~/data/realesrgan_parameters_test/original/ --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --outscale 1 --half --output ~/data/realesrgan_parameters_test/gpu-x2plus-scale1-half --suffix "" --tile 1200

# gpu-x2plus-scale1
# python inference_realesrgan.py --input ~/data/realesrgan_parameters_test/original/ --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --outscale 1 --output ~/data/realesrgan_parameters_test/gpu-x2plus-scale1 --suffix "" --tile 1200

# gpu-x2plus-scale2
# python inference_realesrgan.py --input ~/data/realesrgan_parameters_test/original/ --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --outscale 2 --output ~/data/realesrgan_parameters_test/gpu-x2plus-scale2 --suffix "" --tile 1200

# gpu-x4plus-scale1-half
# python inference_realesrgan.py --input ~/data/realesrgan_parameters_test/original/ --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --outscale 1 --output ~/data/realesrgan_parameters_test/gpu-x4plus-scale1-half --suffix "" --tile 600 --half

# gpu-x4plus-scale1
# python inference_realesrgan.py --input ~/data/realesrgan_parameters_test/original/ --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --outscale 1 --output ~/data/realesrgan_parameters_test/gpu-x4plus-scale1 --suffix "" --tile 600

# gpu-x4plus-scale4
# python inference_realesrgan.py --input ~/data/realesrgan_parameters_test/original/ --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --outscale 4 --output ~/data/realesrgan_parameters_test/gpu-x4plus-scale4 --suffix "" --tile 600


## Running best result in fastest speed
python inference_realesrgan.py --input ~/data/andersonconstruction.7645/ --model_path experiments/pretrained_models/RealESRGAN_x2plus.pth --outscale 2 --output ~/data/andersonconstruction.7645_x2plus_scale2 --suffix "" --tile 1200


