import os
import json
import pandas as pd
import numpy as np
import scipy


base_dir = '/home/dzc/workspace/G-IQA'
data_dir = f'{base_dir}/data'


def get_imgnames_from_json(json_file):
  with open(json_file, 'r') as f:
    imgnames = json.load(f)
  return set(item['image'] if 'image' in item else item['img_path'] for item in imgnames)



############################ PIQ23 ############################
def piq23_generator(domain_id_base = 0):
  all_df = pd.read_csv(os.path.join(data_dir, 'PIQ23', 'Scores_Overall.csv'))
  imgs_name = all_df['IMAGE PATH'].tolist()
  labels = all_df['JOD'].tolist()
  scenes = all_df['SCENE'].tolist()

  all_files = []
  for img_name, label, scene in zip(imgs_name, labels, scenes):
    all_files.append(dict(
      image = f"PIQ23/{img_name}".replace('\\', '/'),
      score = label,
      scene = scene,
      domain_id = domain_id_base + int(scene.split("_")[-1]),
    ))
    assert os.path.exists(os.path.join(data_dir, all_files[-1]['image']))

  with open(f'{base_dir}/data_json/for_leave_one_out/piq23_all.json', 'w') as f:
    json.dump({'files': all_files}, f)
  print(f'PIQ23 all: {len(all_files)}')



############################ SPAQ ############################

def extract_positive_indices(row):
  return [idx for idx, value in enumerate(row) if value > 0]

def spaq_generator(domain_id_base = 100):

  all_df = pd.read_excel(f'{data_dir}/SPAQ/annotations/MOS and Image attribute scores.xlsx')
  scene_df = pd.read_excel(f'{data_dir}/SPAQ/annotations/Scene category labels.xlsx')
  assert all_df['Image name'].tolist() == scene_df['Image name'].tolist()
  imgs_name = all_df['Image name'].tolist()
  labels = all_df['MOS'].tolist()

  # 由于SPAQ中一个图片可能对应多个场景，因此这里提取所有场景类别作为candidate，每轮epoch从对应的candidate中随机挑选一个作为其场景标签
  # scene 标签只用于训练时区分不同场景（场景采样），对于测试而言没有任何意义
  scene_candidate = scene_df.drop(columns='Image name').apply(lambda row: extract_positive_indices(row), axis=1).tolist()
  # Animal	Cityscape	Human	Indoor scene	Landscape	Night scene	Plant	Still-life	Others
  scene_dict = {
    0: 'Animal',
    1: 'Cityscape',
    2: 'Human',
    3: 'Indoor scene',
    4: 'Landscape',
    5: 'Night scene',
    6: 'Plant',
    7: 'Still-life',
    8: 'Others',
  }

  all_files = []
  for img_name, label, scene_list in zip(imgs_name, labels, scene_candidate):
    all_files.append(dict(
      image = f'SPAQ/TestImage/{img_name}',
      score = label,
      scene = [scene_dict[scene] for scene in scene_list],
      domain_id = [domain_id_base + scene for scene in scene_list],
    ))

    assert os.path.exists(os.path.join(data_dir, all_files[-1]['image']))

  with open(f'{base_dir}/data_json/for_leave_one_out/spaq_all.json', 'w') as f:
    json.dump({'files': all_files}, f)


  # split train/test follow by q-align
  train_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/training_sft/train_spaq.json')
  test_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/test_jsons/test_spaq.json')
  assert len(train_img & test_img) == 0
  train_files = [item for item in all_files if item['image'].replace('SPAQ/TestImage', 'spaq') in train_img]
  test_files = [item for item in all_files if item['image'].replace('SPAQ/TestImage', 'spaq') in test_img]
  assert len(train_files) + len(test_files) == len(train_img) + len(test_img)
  with open(f'{base_dir}/data_json/for_cross_set/train/spaq_train.json', 'w') as f:
    json.dump({'files': train_files}, f)
  with open(f'{base_dir}/data_json/for_cross_set/test/spaq_test.json', 'w') as f:
    json.dump({'files': test_files}, f)
  print(f'SPAQ all: {len(all_files)}, train: {len(train_files)}, test: {len(test_files)}')


############################ LIVE Challenge ############################
def livec_generator(domain_id_base = 200):
  imgpath = scipy.io.loadmat(os.path.join(data_dir, 'LIVEChallenge', 'Data', 'AllImages_release.mat'))
  imgpath = imgpath['AllImages_release']
  imgpath = imgpath
  mos = scipy.io.loadmat(os.path.join(data_dir, 'LIVEChallenge', 'Data', 'AllMOS_release.mat'))
  labels = mos['AllMOS_release']
  labels = labels[0]

  all_files = []
  for img, label in zip(imgpath, labels):
    all_files.append(dict(
      image = f'LIVEChallenge/Images/{img[0][0]}',
      score = label,
      domain_id = domain_id_base,
    ))
    assert os.path.exists(os.path.join(data_dir, all_files[-1]['image']))
  
  # split train/test follow by q-align
  # all for test
  test_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/test_jsons/livec.json')
  test_files = [item for item in all_files if item['image'].replace('LIVEChallenge/Images', 'livec') in test_img]
  assert len(test_files) == len(test_img)
  with open(f'{base_dir}/data_json/for_cross_set/test/livec.json', 'w') as f:
    json.dump({'files': test_files}, f)
  print(f'LIVE Challenge all: {len(all_files)}, test: {len(test_files)}')


############################ Koniq_10k ############################
def koniq_generator(domain_id_base = 300):
  all_df = pd.read_csv(os.path.join(data_dir, 'koniq10k/koniq10k_scores_and_distributions.csv'))
  imgs_name = all_df['image_name'].tolist()
  labels = all_df['MOS_zscore'].tolist()

  all_files = []
  for img_name, label in zip(imgs_name, labels):
    all_files.append(dict(
      image = f'koniq10k/1024x768/{img_name}',
      score = label,
      domain_id = domain_id_base,
    ))
    assert os.path.exists(os.path.join(data_dir, all_files[-1]['image']))

  # split train/test follow by q-align
  train_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/training_sft/train_koniq.json')
  test_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/test_jsons/test_koniq.json')
  assert len(train_img & test_img) == 0
  train_files = [item for item in all_files if item['image'].replace('koniq10k/1024x768', 'koniq') in train_img]
  test_files = [item for item in all_files if item['image'].replace('koniq10k/1024x768', 'koniq') in test_img]
  assert len(train_files) + len(test_files) == len(train_img) + len(test_img)
  with open(f'{base_dir}/data_json/for_cross_set/train/koniq10k_train.json', 'w') as f:
    json.dump({'files': train_files}, f)
  with open(f'{base_dir}/data_json/for_cross_set/test/koniq10k_test.json', 'w') as f:
    json.dump({'files': test_files}, f)
  print(f'Koniq all: {len(all_files)}, train: {len(train_files)}, test: {len(test_files)}')


############################ BID ############################
def bid_generator(domain_id_base = 400):
  pass


############################ AGIQA-3K ############################
def agiqa3k_generator(domain_id_base = 500):
  all_df = pd.read_csv(os.path.join(data_dir, 'AGIQA-3K/data.csv'))
  imgs_name = all_df['name'].tolist()
  labels = all_df['mos_quality'].tolist()
  style = all_df['style'].tolist()
  style = ['other' if s is np.nan else s for s in style]
  style_dict = {s: i for i, s in enumerate(sorted(set(style)))}

  all_files = []
  for img_name, label, style in zip(imgs_name, labels, style):
    all_files.append(dict(
      image = f'AGIQA-3K/images/{img_name}',
      score = label,
      style = style,
      domain_id = domain_id_base + style_dict[style],
    ))
    assert os.path.exists(os.path.join(data_dir, all_files[-1]['image']))

  # split test follow by q-align
  test_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/test_jsons/agi.json')
  test_files = [item for item in all_files if item['image'].replace('AGIQA-3K/images', 'agi-cgi') in test_img]
  assert len(test_files) == len(test_img)
  with open(f'{base_dir}/data_json/for_cross_set/test/agiqa3k.json', 'w') as f:
    json.dump({'files': test_files}, f)
  print(f'AGIQA-3K all: {len(all_files)}, test: {len(test_files)}')


############################ KADID-10k ############################
def kadid10k_generator(domain_id_base = 600):
  all_df = pd.read_csv(os.path.join(data_dir, 'kadid10k/dmos.csv'))
  imgs_name = all_df['dist_img'].tolist()
  labels = all_df['dmos'].tolist()
  dist_type = [img.split('_')[1] for img in imgs_name]

  all_files = []
  for img_name, label, dist_type in zip(imgs_name, labels, dist_type):
    all_files.append(dict(
      image = f'kadid10k/images/{img_name}',
      score = label,
      dist_type = dist_type,
      domain_id = domain_id_base + int(dist_type),
    ))
    assert os.path.exists(os.path.join(data_dir, all_files[-1]['image']))
  
  # split train/test follow by q-align
  train_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/training_sft/train_kadid.json')
  test_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/test_jsons/test_kadid.json')
  # assert len(train_img & test_img) == 0 # TODO: overlap
  train_files = [item for item in all_files if item['image'].replace('kadid10k/images', 'kadid10k') in train_img]
  test_files = [item for item in all_files if item['image'].replace('kadid10k/images', 'kadid10k') in test_img]
  assert len(train_files) + len(test_files) == len(train_img) + len(test_img)
  with open(f'{base_dir}/data_json/for_cross_set/train/kadid10k_train.json', 'w') as f:
    json.dump({'files': train_files}, f)
  with open(f'{base_dir}/data_json/for_cross_set/test/kadid10k_test.json', 'w') as f:
    json.dump({'files': test_files}, f)
  print(f'KADID-10k all: {len(all_files)}, train: {len(train_files)}, test: {len(test_files)}')
  
############################ LIVE ############################
def getFileName(path, suffix):
  filename = []
  f_list = os.listdir(path)
  for i in f_list:
      if os.path.splitext(i)[1] == suffix:
          filename.append(i)
  return filename
  
def live_generator(domain_id_base = 700):
  ''' Modified from https://github.com/SSL92/hyperIQA '''
  def getDistortionTypeFileName(path, num):
    filename = []
    index = 1
    for i in range(0, num):
        name = '%s%s%s' % ('img', str(index), '.bmp')
        filename.append(os.path.join(path, name))
        index = index + 1
    return filename
  
  root = os.path.join(data_dir, 'LIVE')
  refpath = os.path.join(root, 'refimgs')
  refname = getFileName(refpath, '.bmp')

  jp2kroot = os.path.join(root, 'jp2k')
  jp2kname = getDistortionTypeFileName(jp2kroot, 227)

  jpegroot = os.path.join(root, 'jpeg')
  jpegname = getDistortionTypeFileName(jpegroot, 233)

  wnroot = os.path.join(root, 'wn')
  wnname = getDistortionTypeFileName(wnroot, 174)

  gblurroot = os.path.join(root, 'gblur')
  gblurname = getDistortionTypeFileName(gblurroot, 174)

  fastfadingroot = os.path.join(root, 'fastfading')
  fastfadingname = getDistortionTypeFileName(fastfadingroot, 174)

  imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

  dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
  labels = dmos['dmos_new']

  orgs = dmos['orgs']
  refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
  refnames_all = refnames_all['refnames_all']

  dist_type_dict = {
    'jp2k': 0,
    'jpeg': 1,
    'wn': 2,
    'gblur': 3,
    'fastfading': 4,
  }

  all_files = []
  for i, rname in enumerate(refname):
    train_sel = (rname == refnames_all)
    train_sel = train_sel * ~orgs.astype(np.bool_)
    train_sel = np.where(train_sel == True)
    train_sel = train_sel[1].tolist()
    for j, item in enumerate(train_sel):
      dist_type = imgpath[item].split('/')[-2]
      all_files.append(dict(
        image = imgpath[item].replace(f'{data_dir}/', ''),
        score = labels[0][item],
        dist_type = dist_type,
        domain_id = domain_id_base + dist_type_dict[dist_type],
      ))
      assert os.path.exists(os.path.join(data_dir, all_files[-1]['image']))

  # split test follow by q-align
  test_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/test_jsons/live.json')
  test_files = [item for item in all_files if item['image'].replace('LIVE', 'live') in test_img]
  # assert len(test_files) == len(test_img) # TODO: not equal
  with open(f'{base_dir}/data_json/for_cross_set/test/live.json', 'w') as f:
    json.dump({'files': test_files}, f)
  print(f'LIVE all: {len(all_files)}, test: {len(test_files)}')
  

############################ CSIQ ############################
def csiq_generator(domain_id_base = 800):
  ''' Modified from https://github.com/SSL92/hyperIQA '''
  refpath = os.path.join(data_dir, 'CSIQ', 'src_imgs')
  refname = getFileName(refpath,'.png')
  txtpath = os.path.join(data_dir, 'CSIQ', 'csiq_label.txt') # see: https://github.com/SSL92/hyperIQA/blob/master/csiq_label.txt
  fh = open(txtpath, 'r')
  imgnames = []
  target = []
  refnames_all = []
  for line in fh:
      line = line.split('\n')
      words = line[0].split()
      imgnames.append((words[0]))
      target.append(words[1])
      ref_temp = words[0].split(".")
      refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

  labels = np.array(target)
  refnames_all = np.array(refnames_all)

  dist_type_dict = {
    'awgn': 0,
    'blur': 1,
    'contrast': 2,
    'fnoise': 3,
    'jpeg': 4,
    'jpeg2000': 5,
  }

  all_files = []
  for i, item in enumerate(refname):
    train_sel = (refname[i] == refnames_all)
    train_sel = np.where(train_sel == True)
    train_sel = train_sel[0].tolist()
    for j, item in enumerate(train_sel):
      dist_type = imgnames[item].split('.')[1].lower()
      all_files.append(dict(
        image = os.path.join('CSIQ', 'dst_imgs', dist_type, imgnames[item]),
        score = float(labels[item]),
        dist_type = dist_type,
        domain_id = domain_id_base + dist_type_dict[dist_type],
      ))

  # split test follow by q-align
  test_img = get_imgnames_from_json(f'{data_dir}/Q-Align/playground/data/test_jsons/csiq.json')
  test_files = [item for item in all_files if item['image'].replace('CSIQ', 'csiq') in test_img]
  # assert len(test_files) == len(test_img) # TODO: miss gt in csiq_label.txt
  with open(f'{base_dir}/data_json/for_cross_set/test/csiq.json', 'w') as f:
    json.dump({'files': test_files}, f)
  print(f'CSIQ all: {len(all_files)}, test: {len(test_files)}')



############################ PARA ############################



############################ EVA ############################
def eva_generator(domain_id_base = 1100):
  
  # animals, architectures and city scenes, human, natural and rural scenes, still life, other
  cat_dict = {
    "1": "animals",
    "2": "architectures and city scenes",
    "3": "human",
    "4": "natural and rural scenes",
    "5": "still life",
    "6": "other",
  }

  all_files = []
  with open(f'{data_dir}/EVA/eva-dataset-master/data/image_content_category.csv', 'r') as f:
    lines = f.readlines()

  info = pd.read_csv(f'{data_dir}/EVA/eva-dataset-master/data/votes_filtered.csv', sep='=', dtype={'image_id': str})
  for line in lines[1:]:
    id, cat = line.strip().split(',')
    id, cat = id.strip('"'), cat.strip('"')
    score = info['score'][info['image_id']==id].to_numpy()
    if len(score) == 0:
      continue
    all_files.append({
      'image': f'EVA/eva-dataset-master/images/EVA_category/EVA_category/{cat}/{id}.jpg',
      'score': np.mean(score),
      'scene': cat_dict[cat],
      'domain_id': int(cat) + domain_id_base,
    })
    assert os.path.exists(os.path.join(data_dir, all_files[-1]['image']))

  with open(f'{base_dir}/data_json/for_leave_one_out/eva_all.json', 'w') as f:
    json.dump({'files': all_files}, f)
  print(f'EVA all: {len(all_files)}')




if __name__ == '__main__':
  piq23_generator()
  spaq_generator()
  livec_generator()
  koniq_generator()
  agiqa3k_generator()
  kadid10k_generator()
  live_generator()
  csiq_generator()
  eva_generator()