CUDA_VISIBLE_DEVICES=4,5 python main.py \
--model=RERD --lonly --aonly --vonly \
--name='RERD-n-mosei-bert-lav-d64h4l3lbd1nl3-nparams-e80bs32adm-regsh-clp18-s1234-01' \
--dataset='mosei' --data_path='/home1/xiaobao/OrthTransFusion/bert_data/MMSA/MOSEI' \
--batch_size=16 --use_bert=True\
--seed=1234 --num_epochs=80 --when=40 \
--dis_d_mode=64 --dis_n_heads=4 --dis_e_layers=2 \
--optim='Adam' --reg_lambda=0.1 \
--schedule='c' --lr=0.001 --nlevels=2