import json
import os

dataname = 'koniq10k'

# load all json
with open(f'data_json/all/{dataname}_all.json', 'r') as f:
  data = json.load(f)
  datajson = data['files']
  domain_name = data['domain_name']
  
# assert len(domain_name) == len(set(item['domain_id'] for item in datajson)), 'Domain number not match'

# split for leave one out
for test_domain, d_name in domain_name.items():
    test_domain = int(test_domain)

    save_dir = f'data_json/for_leave_one_out/{dataname}/test_for_{test_domain}_{d_name.replace(" ", "-")}'

    train_datajson = [item for item in datajson if item['domain_id'] != test_domain]
    test_datajson = [item for item in datajson if item['domain_id'] == test_domain]

    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/train.json', 'w') as f:
      json.dump({'files': train_datajson, 'domain_name': domain_name}, f, indent=2)
    with open(f'{save_dir}/test.json', 'w') as f:
      json.dump({'files': test_datajson, 'domain_name': domain_name}, f, indent=2)