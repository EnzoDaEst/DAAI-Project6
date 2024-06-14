target_domain=${1}

pythonProject/venv1/bin/python3 main.py \
--experiment=ASHbaseline \
--experiment_name=ASHbaseline/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1