import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import torch.nn.functional as f

def _eval(model, args):
    state_dict = torch.load(args.test_model, map_location=args.device)
    model.load_state_dict(state_dict['model'])
    device = args.device
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()
    factor = 4
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            tm = time.time()

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()


            label_img = (label_img).to(device)
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))	

            '''
            MPS device throws error for F.adaptive_avg_pool2
            RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes.
            '''
            cpu=torch.device('cpu')
            params1 = f.adaptive_avg_pool2d(pred_clip.to(cpu), (int(H / down_ratio), int(W / down_ratio)))
            params2 = f.adaptive_avg_pool2d(label_img.to(cpu), (int(H / down_ratio), int(W / down_ratio)))

            ssim_val = ssim(params1, params2, data_range=1, size_average=False)	
            print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
            
            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr_val)

            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))

        print("Average time: %f" % adder.average())

