import os
import json

work_dir = '.'

# load LIQE split
liqe_split_dir = dict(
  live=f'{work_dir}/sota/LIQE/IQA_Database/databaserelease2/splits2',
  csiq=f'{work_dir}/sota/LIQE/IQA_Database/CSIQ/splits2',
  # kadid10k=f'{work_dir}/sota/LIQE/IQA_Database/kadid10k/splits2',
  # bid=f'{work_dir}/sota/LIQE/IQA_Database/BID/splits2',
  # livec=f'{work_dir}/sota/LIQE/IQA_Database/ChallengeDB_release/splits2',
  # koniq10k=f'{work_dir}/sota/LIQE/IQA_Database/koniq-10k/splits2',

)

for d_name in liqe_split_dir.keys():
  # load all data json
  with open(f'{work_dir}/data_json/all/{d_name}_all.json', 'r') as f:
    all_data = json.load(f)['files']
  # we need to reversion the score of live and csiq, if we use them to mix-dataset training
  if d_name == 'live' or d_name == 'csiq':
    max_score = max([item['score'] for item in all_data])
    for item in all_data:
      item['score'] = max_score - item['score']

  for i in range(1, 11):
    for phase in ['train', 'val', 'test']:
      liqe_d_name = 'clive' if d_name == 'livec' else d_name
      split_file = f'{liqe_split_dir[d_name]}/{i}/{liqe_d_name}_{phase}_clip.txt'

      img_names = []
      with open(split_file, 'r') as f:
        for line in f:
          name = line.split()[0]
          if d_name == 'live':
            name = 'LIVE/' + name
          elif d_name == 'csiq':
            name = 'CSIQ/' + name
          elif d_name == 'kadid10k':
            name = 'kadid10k/' + name
          elif d_name == 'bid':
            name = name.replace('ImageDatabase', 'BID')
          elif d_name == 'livec':
            name = 'LIVEChallenge/' + name
          elif d_name == 'koniq10k':
            name = 'koniq10k/' + name
          img_names.append(name)
      
      split_data = [item for item in all_data if item['image'] in img_names]
      if len(split_data) != len(img_names):
        print(f'{d_name} {i} {phase} {len(split_data)}/LIQE:{len(img_names)}')

      os.makedirs(f'{work_dir}/data_json/liqe_split/{d_name}/{i}', exist_ok=True)
      with open(f'{work_dir}/data_json/liqe_split/{d_name}/{i}/{phase}.json', 'w') as f:
        json.dump(dict(files=split_data), f, indent=2)
