#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/15 11:51:47
@Desc    :   Cleaned up batch inference script for RAVE
@Ref     :   
'''
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from pipelines.cn_rave import RAVE
import utils.video_grid_utils as vgu
from utils.save_video import save_video


data_root = '/data/trc/videdit-benchmark/DynEdit'

config = OmegaConf.create(dict(
    data_root=data_root,
    config_file=f'{data_root}/config.yaml',
    output_dir=f'{data_root}/outputs/rave',
    hf_path="/data/models/stable-diffusion-v1-5",
    hf_cn_path="/data/models/control_v11f1p_sd15_depth",
    preprocess_name='depth_zoe',  # controlnet preprocess
    sample_size=-1, # subsample input video
    grid_size=2,
    pad=1,  # same video as input
    n_frames=24,
    fps=12,
))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RAVE(device)
    model.init_models(config.hf_cn_path, config.hf_path, config.preprocess_name)

    data_config = OmegaConf.load(config.config_file)
    preprocess_elapsed_ls = []
    inference_elapsed_ls = []
    for row in tqdm(data_config['data']):
        output_dir = Path(f"{config.output_dir}/{row.video_id}")
        if output_dir.exists():
            print(f"Skip {row.video_id} ...")
            continue
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # load video
        print(f"Processing {row.video_id} ...")
        video_path = f'{config.data_root}/videos/{row.video_id}.mp4'
        image_pil_list = vgu.prepare_video_to_grid(video_path, config.sample_size, config.grid_size, config.pad)
        sample_size = len(image_pil_list)
        print(f'Frame count: {sample_size}')

        inverse_path = Path(f"{config.output_dir}/{row.video_id}/.cache")
        inverse_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        # preprocess video here
        latents_inverted, control_batch, indices = model.preprocess(
            image_pil_list,
            inversion_prompt=row['prompt'],
            inverse_path=inverse_path,
            grid_size=config.grid_size,
        )
        preprocess_elapsed = time.perf_counter() - start
        preprocess_elapsed_ls.append(preprocess_elapsed)

        print(f'Editting {row.video_id} ...')
        start = time.perf_counter()
        for i, edit in tqdm(enumerate(row.edit)):
            # inference video here
            samples = model(
                latents_inverted, 
                control_batch,
                indices,
                positive_prompts=edit['prompt'], 
                negative_prompts=edit['src_words'],
            )
            # samples[0].save(
            #     output_dir / f'{i}.gif', 
            #     save_all=True, append_images=samples[1:], optimize=False, 
            #     duration=83, loop=1,
            # )
            save_video(output_dir / f'{i}.mp4', samples, config.fps)
        inference_elapsed = time.perf_counter() - start
        inference_elapsed_ls.append(inference_elapsed)

    with open(f'{config.output_dir}/time.log', 'a') as f:
        f.write(f'Preprocess: {sum(preprocess_elapsed_ls)/len(preprocess_elapsed_ls):.2f} sec/video\n')
        n_prompts = len(row.edit)
        f.write(f'Edit:       {sum(inference_elapsed_ls)/len(inference_elapsed_ls)/n_prompts:.2f} sec/edit\n')
        f.write('Preprocess:\n')
        f.writelines([f'{e:.1f} ' for e in preprocess_elapsed_ls])
        f.write('\nEdit:\n')
        f.writelines([f'{e:.1f} ' for e in inference_elapsed_ls])
        f.write('\n')
    print('Everything done!')


if __name__ == '__main__':
    main()