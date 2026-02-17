import logging, copy, time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from tqdm import tqdm
import pynvml  # pip install nvidia-ml-py3
pynvml.nvmlInit()

from thop import profile, clever_format 

def get_used_mem(handle):
    """현재 GPU 사용량(bytes) – nvidia‑smi와 동일."""
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used


def bytes2human(num_bytes, unit="MB"):
    div = 1024 ** (2 if unit.upper() == "MB" else 3)
    return f"{num_bytes / div:,.1f} {unit}"

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

    for test_loader in tqdm(test_loaders):
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
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
