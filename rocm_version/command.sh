./HCGrid --fits_path /home/summit/Project/HPDC_2022/HEGrider/ --input_file real --target_file real_target30 --output_file output --file_id 100 --beam_size 180 --order_arg 1 --block_num 128

./HCGrid --fits_path /home/summit/Project/HPDC_2022/HEGrider/ --input_file mock --target_file mock_target --output_file output --file_id 100 --beam_size 180 --order_arg 1 --block_num 128

python Visualize.py -p /home/summit/Project/HPDC_2022/HEGrider/ -o output -n 1

python target_map.py -p /home/summit/Project/HPDC_2022/HEGrider/ -t target -b 1
