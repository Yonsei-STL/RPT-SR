import logging, copy, time
import torch
from os import path as osp
import os
import numpy as np
import matplotlib.pyplot as plt  # 시각화 저장용


from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from tqdm import tqdm
import pynvml  # pip install nvidia-ml-py3
pynvml.nvmlInit()

from thop import profile, clever_format 
def normalize01(x):
    x = x - x.min()
    d = x.max() - x.min() + 1e-8
    return x / d

def save_overlay(ir_img, heat, out_png, alpha=0.5):
    """
    ir_img: (H,W) 또는 (H,W,3) numpy, 0..1
    heat  : (H,W) numpy, 0..1
    """
    heat = np.clip(heat, 0, 1)
    if ir_img.ndim == 2:
        base = np.stack([ir_img, ir_img, ir_img], axis=-1)
    else:
        base = ir_img
    cmap = plt.cm.jet(heat)[..., :3]   # (H,W,3)
    out = (1 - alpha) * base + alpha * cmap
    out = np.clip(out, 0, 1)
    plt.imsave(out_png, out)


def get_used_mem(handle):
    """현재 GPU 사용량(bytes) – nvidia‑smi와 동일."""
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used


def bytes2human(num_bytes, unit="MB"):
    div = 1024 ** (2 if unit.upper() == "MB" else 3)
    return f"{num_bytes / div:,.1f} {unit}"

def enable_rpa_recording(net_g, targets):
    """
    targets: [(stage_idx, rpal_idx), ...]
    ex) [(0,0), (3,-1)]  # Stage-1의 첫 RPAL과 Stage-4의 마지막 RPAL
    """
    for si, li in targets:
        rpal = net_g.layers[si].residual_group.blocks[li]
        # Window-attn + carrier self-attn 둘 다 기록
        rpal.attn.enable_recording(True)

def set_rpa_ablation(net_g, mode="full"):
    """
    mode: "full" | "local_only" | "prior_only"
    """
    for stage in net_g.layers:
        for rpal in stage.residual_group.blocks:
            if hasattr(rpal, "ablation_mode"):
                rpal.ablation_mode = mode

def run_validation_with_rpa(model, test_loader, out_root, targets, save_sr=False, ablation="full", max_images=None):
    """
    model     : basicsr 모델(wrapper)
    test_loader: dataloader
    out_root : 결과 저장 루트 (ex) .../experiments/xxx/rpa_vis
    targets  : 기록할 레이어 리스트 [(si, li), ...]
    save_sr  : SR 결과 이미지도 저장할지 여부(선택)
    ablation : "full" | "local_only" | "prior_only"
    max_images: 저장할 이미지 개수 제한(너무 많으면 오래 걸리므로 선택)
    """
    os.makedirs(out_root, exist_ok=True)
    net_g = model.net_g.eval()

    # 1) recording 켜기 + 아블레이션 모드
    enable_rpa_recording(net_g, targets)
    set_rpa_ablation(net_g, ablation)

    device = next(net_g.parameters()).device
    cnt = 0

    for data in test_loader:
        # 파일명/경로
        if 'lq_path' in data:
            img_name = os.path.splitext(os.path.basename(data['lq_path'][0]))[0]
        else:
            # 데이터셋 구현에 따라 다를 수 있음
            img_name = f"img_{cnt:05d}"

        # 2) 추론
        model.feed_data(data)
        with torch.no_grad():
            model.test()

        # 3) 시각화용 입력/출력 획득
        vis = model.get_current_visuals()  # dict: {'lq', 'result', 'gt', ...} 구현에 따라 다를 수 있음
        # 입력(IR) – 채널/스케일 맞춰 0..1로
        if 'lq' in vis:
            lq = vis['lq'].cpu().numpy()[0]  # (C,H,W)
        else:
            lq = data['lq'].cpu().numpy()[0]
        if lq.shape[0] == 1:
            ir = normalize01(lq[0])
        else:
            ir = np.transpose(normalize01(lq), (1,2,0))  # (H,W,3)

        # 4) 지정한 RPAL들의 carrier→local 영향맵 추출/저장
        for si, li in targets:
            rpal = net_g.layers[si].residual_group.blocks[li]
            heat_t = getattr(rpal, 'last_carrier2local_map', None)  # torch.Tensor (B,H,W)
            if heat_t is None:
                continue
            heat = heat_t[0].numpy()
            heat = normalize01(heat)
            out_png = os.path.join(out_root, f"{img_name}_S{si}_L{li}_{ablation}.png")
            save_overlay(ir, heat, out_png, alpha=0.5)

        # 5) (선택) SR 결과 저장
        if save_sr and 'result' in vis:
            sr = vis['result'].cpu().numpy()[0]  # (C, Hr, Wr)
            if sr.shape[0] == 1:
                sr_img = normalize01(sr[0])
            else:
                sr_img = np.transpose(normalize01(sr), (1,2,0))
            sr_png = os.path.join(out_root, f"{img_name}_SR.png")
            plt.imsave(sr_png, sr_img, cmap=None)

        cnt += 1
        if (max_images is not None) and (cnt >= max_images):
            break

    # 기록 해제(선택)
    for si, li in targets:
        rpal = net_g.layers[si].residual_group.blocks[li]
        rpal.attn.enable_recording(False)

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    
    model = build_model(opt)
    net_g = model.net_g.eval()
    
    in_c  = getattr(net_g, 'in_channels', 3)      # 없으면 기본값 3
    h, w  = 128, 128                                # 예: 64×64; opt에서 가져와도 OK
    dummy = torch.randn(1, in_c, h, w).to(next(net_g.parameters()).device)

    flops, params = profile(copy.deepcopy(net_g), inputs=(dummy,), verbose=False)
    flops, params = clever_format([flops, params], '%.3f')

    logger.info(f'*** Model complexity ***')
    logger.info(f'#Params: {params}')
    logger.info(f'FLOPs : {flops}')

    device = next(net_g.parameters()).device
    gpu_index = device.index if device.type == "cuda" else 0
    handle    = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    peak_used = get_used_mem(handle)  # 초기값(대개 0)
    start_time = time.time()

    dump_rpa = opt['val'].get('dump_rpa', False)
    rpa_targets = opt['val'].get('rpa_layers', [(0,0), (3,-1)])  # 예: Stage-1 첫 RPAL, Stage-4 마지막 RPAL
    rpa_mode    = opt['val'].get('rpa_mode', 'full')             # "full"|"local_only"|"prior_only"
    rpa_outdir  = osp.join(opt['path']['results_root'], f"{opt['name']}_rpa_vis_{rpa_mode}")

    for test_loader in tqdm(test_loaders):
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        if not dump_rpa:
            model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        else:
            # 시각화 모드: 직접 루프를 돌며 맵 저장
            logger.info(f'RPA visualization -> {rpa_outdir}')
            run_validation_with_rpa(model, test_loader, rpa_outdir, targets=rpa_targets,
                                    save_sr=False, ablation=rpa_mode, max_images=opt['val'].get('vis_num', None))

        peak_used = max(peak_used, get_used_mem(handle))
    elapsed = time.time() - start_time
    
    if device.type == "cuda":
        logger.info("═════════════════════════════════════════════")
        logger.info(f"Global Peak GPU memory (nvidia‑smi) : "
                    f"{bytes2human(peak_used, 'MB')} "
                    f"({bytes2human(peak_used, 'GB')})")
        logger.info(f"Total test time: {elapsed:,.1f}s")
        logger.info("═════════════════════════════════════════════")
    else:
        logger.info("CUDA 디바이스가 아니므로 GPU 메모리 사용량을 측정하지 않았습니다.")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
