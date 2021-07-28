cd ..
rm -rf mesh-transformer-jax || true

git clone https://github.com/jroakes/mesh-transformer-jax.git
cd mesh-transformer-jax
pip install -r requirements.txt
pip install jax==0.2.12
wget https://storage.googleapis.com/gpt-j-finetuning-europe/zoominfo_test.csv

python3 device_train.py --config=configs/company_data.json --tune-model-path=gs://gpt-j-finetuning-europe/step_383500/
python3 device_sample_df.py --config=configs/company_data.json --sample_file zoominfo_test.csv --prompt_column model_input --num_samples 25 --temp 0.3
