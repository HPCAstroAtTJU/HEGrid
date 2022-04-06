./HCGrid --fits_path /home/summit/Project/HPDC_2022/ --input_file multifreq --target_file target --output_file output --file_id 100 --beam_size 180 --order_arg 1 --block_num 352

python Visualize.py -p /home/summit/Project/HPDC_2022/ -o output -n 1

python Creat_target_file.py -p /home/summit/Project/HPDC_2022/ -t target -n 1
