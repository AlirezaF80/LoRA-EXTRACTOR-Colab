# extract approximating LoRA by svd from two SD models
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo!

import argparse
import torch
import model_util
import lora


CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6

def svd(args):
  print(f"loading SD model : {args.model_org}")
  text_encoder_o, _, unet_o = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.model_org)
  print(f"loading SD model : {args.model_tuned}")
  text_encoder_t, _, unet_t = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.model_tuned, None, text_encoder_o)

  # create LoRA network to extract weights: Use dim (rank) as alpha
  lora_network_o = lora.create_network(1.0, 512, 512, None, text_encoder_o, unet_o)
  lora_network_t = lora.create_network(1.0, 512, 512, None, text_encoder_t, unet_t)
  assert len(lora_network_o.text_encoder_loras) == len(
      lora_network_t.text_encoder_loras), f"model version is different (SD1.x vs SD2.x) / それぞれのモデルのバージョンが違います（SD1.xベースとSD2.xベース） "

  # get diffs
  diffs = {}
  text_encoder_different = False
  for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras)):
    lora_name = lora_o.lora_name
    module_o = lora_o.org_module
    module_t = lora_t.org_module
    diff = module_t.weight - module_o.weight

    # Text Encoder might be same
    if torch.max(torch.abs(diff)) > MIN_DIFF:
      text_encoder_different = True

    diff = diff.float()
    diffs[lora_name] = diff

  if not text_encoder_different:
    print("Text encoder is same. Extract U-Net only.")
    lora_network_o.text_encoder_loras = []
    diffs = {}

  for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.unet_loras, lora_network_t.unet_loras)):
    lora_name = lora_o.lora_name
    module_o = lora_o.org_module
    module_t = lora_t.org_module
    diff = module_t.weight - module_o.weight
    diff = diff.float()

    if args.device:
      diff = diff.to(args.device)

    diffs[lora_name] = diff

  # make LoRA with svd
  print("calculating by svd")
  threshold = args.threshold
  with torch.no_grad():
    for lora_name, mat in list(diffs.items()):
      conv2d = (len(mat.size()) == 4)
      if conv2d:
        mat = mat.squeeze()

      U, S, Vh = torch.linalg.svd(mat)
      rank = 0
      S_sum = torch.sum(S)
      sum = 0
      for i in range(len(S)):
        sum += S[i]
        if sum / S_sum > threshold:
          rank = i + 1
          break
      break
  print(f"suggested dim: {rank}")



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--v2", action='store_true',
                      help='load Stable Diffusion v2.x model / Stable Diffusion 2.xのモデルを読み込む')
  parser.add_argument("--model_org", type=str, default=None,
                      help="Stable Diffusion original model: ckpt or safetensors file / 元モデル、ckptまたはsafetensors")
  parser.add_argument("--model_tuned", type=str, default=None,
                      help="Stable Diffusion tuned model, LoRA is difference of `original to tuned`: ckpt or safetensors file / 派生モデル（生成されるLoRAは元→派生の差分になります）、ckptまたはsafetensors")
  parser.add_argument("--threshold", type=float, default=0.9, help="threshold for SVD (default 0.9) / SVDの閾値（デフォルト0.95）")
  parser.add_argument("--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う")

  args = parser.parse_args()
  svd(args)
