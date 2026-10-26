#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import yaml

LOCAL_DOVER = Path(__file__).resolve().parents[1] / "Objective" / "Video" / "DOVER"
if LOCAL_DOVER.exists():
    sys.path.insert(0, str(LOCAL_DOVER))

from dover.datasets import ViewDecompositionDataset
from dover.models import DOVER


def fuse_results(results: list[float]) -> dict[str, float]:
    technical = (results[1] - 0.1107) / 0.07355
    aesthetic = (results[0] + 0.08285) / 0.03774
    overall = technical * 0.6104 + aesthetic * 0.3896
    return {
        'aesthetic': float(1 / (1 + np.exp(-aesthetic))),
        'technical': float(1 / (1 + np.exp(-technical))),
        'overall': float(1 / (1 + np.exp(-overall))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch DOVER evaluation')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open('r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    test_load_path = Path(opt['test_load_path'])
    if not test_load_path.is_absolute():
        opt['test_load_path'] = str((config_path.parent / test_load_path).resolve())

    evaluator = DOVER(**opt['model']['args']).to(args.device)
    checkpoint = torch.load(opt['test_load_path'], map_location=args.device)
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    evaluator.load_state_dict(state_dict, strict=True)
    evaluator.eval()

    dopt = opt['data']['val-l1080p']['args']
    dopt['anno_file'] = None
    dopt['data_prefix'] = str(input_dir)

    dataset = ViewDecompositionDataset(dopt)

    results = []
    sample_types = ['aesthetic', 'technical']
    for idx in range(len(dataset)):
        data = dataset[idx]
        if len(data.keys()) == 1:
            continue

        video = {}
        for key in sample_types:
            if key not in data:
                continue
            value = data[key].unsqueeze(0).to(args.device)
            b, c, t, h, w = value.shape
            clips = int(data['num_clips'][key])
            value = (
                value.reshape(b, c, clips, t // clips, h, w)
                .permute(0, 2, 1, 3, 4, 5)
                .reshape(b * clips, c, t // clips, h, w)
            )
            video[key] = value

        with torch.no_grad():
            raw = evaluator(video, reduce_scores=False)
            raw = [float(np.mean(item.detach().cpu().numpy())) for item in raw]
        fused = fuse_results(raw)
        file_path = str(data['name'])
        filename = Path(file_path).name
        results.append(
            {
                'file': file_path,
                'filename': filename,
                'aesthetic': fused['aesthetic'],
                'technical': fused['technical'],
                'overall': fused['overall'],
            }
        )

        sample, separator, model = Path(filename).stem.rpartition('__')
        if separator:
            print(f'{model} {sample} ok', flush=True)
        else:
            print(f'{filename} ok', flush=True)

    payload = {
        'metric': 'video_technical',
        'summary': {
            'mean_aesthetic': float(mean([item['aesthetic'] for item in results])) if results else 0.0,
            'mean_technical': float(mean([item['technical'] for item in results])) if results else 0.0,
            'mean_overall': float(mean([item['overall'] for item in results])) if results else 0.0,
            'total_samples': len(results),
        },
        'results': results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Saved results to {output_path}')


if __name__ == '__main__':
    main()
