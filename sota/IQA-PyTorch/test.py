import pyiqa, torch
import json, os
from rich.progress import track
import numpy as np
from scipy.stats import spearmanr, pearsonr
from PIL import Image

os.environ["http_proxy"] = 'http://192.168.195.225:7890'
os.environ["https_proxy"] = 'http://192.168.195.225:7890'
# os.environ["http_proxy"] = 'http://100.109.219.89:7890'
# os.environ["https_proxy"] = 'http://100.109.219.89:7890'

if __name__ == "__main__":
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  # create metric with default setting
  iqa_metric = pyiqa.create_metric('clipiqa+', device=device)
  os.makedirs("test/"+iqa_metric.metric_name, exist_ok=True)

  test_dir = "data_json/for_cross_set/test/"
  iqa_metric.eval()
  datasets = {}
  for json_file in os.listdir(test_dir):
    with open(test_dir + json_file, "r") as f:
      data = json.load(f)
      datasets[json_file[:-5]] = data["files"]

  for dataset_name, data in datasets.items():
    if dataset_name in ['spaq_test']:
      continue

    print('\n>>>>>>>', dataset_name)
    prs, gts = [], []
    for file in track(data):
      img_path = "data/" + file["image"]
      img_pil = Image.open(img_path).convert('RGB')

      # keep ratio and resize shorter edge to 1024
      w, h = img_pil.size
      if min(w, h) > 512:
          if w > h:
              ow = 512
              oh = int(512 * h / w)
          else:
              oh = 512
              ow = int(512 * w / h)
          img_pil = img_pil.resize((ow, oh), Image.BICUBIC)

      prs += [iqa_metric(img_pil)]
      gts += [file["score"]]

    prs_np = torch.cat(prs).squeeze(-1).cpu().numpy()
    gts_np = np.array(gts)
    srcc, plcc = spearmanr(prs_np, gts_np)[0], pearsonr(prs_np, gts_np)[0]

    prs_norm = prs_np
    gts_norm = gts_np / 100
    srcc_norm, plcc_norm = spearmanr(prs_norm, gts_norm)[0], pearsonr(prs_norm, gts_norm)[0]

    print(f"SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")
    print(f"Normalized SRCC: {srcc_norm:.4f}, PLCC: {plcc_norm:.4f}")


# ['ahiq', 'arniqa', 'arniqa-clive', 'arniqa-csiq', 'arniqa-flive', 'arniqa-kadid', 'arniqa-koniq', 'arniqa-live', 'arniqa-spaq', 'arniqa-tid', 'brisque', 'ckdn', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'clipscore', 'cnniqa', 'cw_ssim', 'dbcnn', 'dists', 'entropy', 'fid', 'fsim', 'gmsd', 'hyperiqa', 'ilniqe', 'inception_score', 'laion_aes', 'liqe', 'liqe_mix', 'lpips', 'lpips-vgg', 'mad', 'maniqa', 'maniqa-kadid', 'maniqa-koniq', 'maniqa-pipal', 'ms_ssim', 'musiq', 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-koniq', 'nima-spaq', 'nima-vgg16-ava', 'niqe', 'nlpd', 'nrqm', 'paq2piq', 'pi', 'pieapp', 'psnr', 'psnry', 'qalign', 'ssim', 'ssimc', 'stlpips', 'stlpips-vgg', 'topiq_fr', 'topiq_fr-pipal', 'topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 'topiq_nr-face', 'topiq_nr-flive', 'topiq_nr-spaq', 'tres', 'tres-flive', 'tres-koniq', 'unique', 'uranker', 'vif', 'vsi', 'wadiqam_fr', 'wadiqam_nr']