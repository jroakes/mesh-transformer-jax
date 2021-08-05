
echo "Removing Old Folder"
rm -rf mesh-transformer-jax || true
echo "Cloning new repo"
git clone https://github.com/jroakes/mesh-transformer-jax.git
echo "Changing directory"
cd mesh-transformer-jax
# pip install -r requirements.txt
# pip install jax==0.2.12

echo "Grabbing new test file"
wget https://storage.googleapis.com/gpt-j-finetuning-europe/zoominfo_test.csv

echo "Running sampling script"
python3 device_sample_df.py --config=configs/company_data.json --sample_file zoominfo_test.csv --prompt_column model_input --name_column name --num_samples 100 --temp 0.75 --top_p 0.92 --rep_penalty 1.5
