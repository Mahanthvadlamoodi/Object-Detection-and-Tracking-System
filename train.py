import argparse
import os
from penet_original import PENetWrapper
import ultralytics.nn.tasks as _t
import ultralytics.nn.modules as _m
_t.__dict__['PENetWrapper'] = PENetWrapper
_m.__dict__['PENetWrapper'] = PENetWrapper

import torch, copy
from ultralytics import YOLO

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model = YOLO(args.yaml_config)
    model.model.to('cuda')
    pre = YOLO(args.baseline)
    pre_sd = pre.model.state_dict()
    mod_sd = model.model.state_dict()
    shifted = {}
    for k, v in pre_sd.items():
        if k.startswith('model.') and k.split('.')[1].isdigit():
            nk = 'model.' + str(int(k.split('.')[1])+1) + '.' + '.'.join(k.split('.')[2:])
            if nk in mod_sd and mod_sd[nk].shape == v.shape:
                shifted[nk] = v
    model.model.load_state_dict(shifted, strict=False)
    print(f'Loaded {len(shifted)} pretrained backbone weights')
    
    enh = sum(p.numel() for p in model.model.model[0].parameters())
    tot = sum(p.numel() for p in model.model.parameters())
    print(f'Enhanced Author PENet Parameters: {enh:,} ({100*enh/tot:.2f}%)  Total: {tot:,}')
    
    init_weights = f"init_{args.baseline.split('/')[-1]}"
    torch.save({
        'model': copy.deepcopy(model.model).half(),
        'train_args': {'task':'detect','data':args.data,'imgsz':args.imgsz,'model':args.yaml_config},
    }, init_weights)
    print('Ready. Proceed to Stage 1.')
    
    
    model = YOLO(init_weights)
    
    yolo_layers = [i for i in range(1, 23)]
    
    print(f'\n--- STARTING STAGE 1: Freezing YOLO ({args.stage1_epochs} Epochs) ---')
    model.train(
        data=args.data,
        epochs=args.stage1_epochs, imgsz=args.imgsz, batch=args.stage1_batch,
        optimizer='SGD', lr0=0.01, lrf=0.01, cos_lr=True, momentum=0.937,
        warmup_epochs=3, weight_decay=5e-4, workers=32, cache='ram', amp=True,
        patience=30, mosaic=1.0, close_mosaic=0, mixup=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.6,
        translate=0.1, scale=0.2, fliplr=0.5, erasing=0.0,
        project=args.project, name='stage1', exist_ok=True, plots=True, save=True, freeze=yolo_layers
    )
    
    # Safely fetch the best path from Stage 1 based on the project structure
    stage1_best = os.path.join(args.project, 'stage1', 'weights', 'best.pt')
    if not os.path.exists(stage1_best):
        stage1_best = os.path.join('runs/detect', args.project, 'stage1', 'weights', 'best.pt')

    model_stage2 = YOLO(stage1_best)
    print(f'\n--- STARTING STAGE 2: End to End Fine-Tuning ({args.stage2_epochs} Epochs) ---')
    model_stage2.train(
        data=args.data,
        epochs=args.stage2_epochs, imgsz=args.imgsz, batch=args.stage2_batch,
        optimizer='SGD', lr0=0.001,
        lrf=0.01, cos_lr=True, momentum=0.937,
        warmup_epochs=0, 
        weight_decay=5e-4, workers=16, cache='ram', amp=True,
        patience=30, mosaic=1.0, close_mosaic=0, mixup=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        translate=0.05, scale=0.3, fliplr=0.5, erasing=0.2, 
        project=args.project, name='stage2', exist_ok=True, plots=True, save=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PENet+YOLO pipeline")
    parser.add_argument('--yaml_config', type=str, default='pe_yolov8m.yaml', help='Model YAML config')
    parser.add_argument('--baseline', type=str, default='yolov8m.pt', help='Pretrained baseline weights')
    parser.add_argument('--data', type=str, default='/home/ai21ma3ai30/aaadi/Object-Detection-and-Tracking-System/exdark.yaml', help='Data YAML path')
    parser.add_argument('--project', type=str, default='runs/ver4.2', help='Project directory name')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    parser.add_argument('--stage1_epochs', type=int, default=20, help='Epochs for Stage 1')
    parser.add_argument('--stage1_batch', type=int, default=64, help='Batch size for Stage 1')
    parser.add_argument('--stage2_epochs', type=int, default=40, help='Epochs for Stage 2')
    parser.add_argument('--stage2_batch', type=int, default=32, help='Batch size for Stage 2')
    args = parser.parse_args()
    
    main(args)