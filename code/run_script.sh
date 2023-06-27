
"""
THEANO_FLAGS="device=gpu0,floatX=float32" python train.py \
--emb ../preprocessed_data/restaurant/w2v_embedding \
--domain restaurant \
-o output_dir \
"""



#tensorflow에서 hotel 도메인으로 train해보기 

python train.py --emb "../preprocessed_data/hotel/w2v_embedding" --domain "hotel" -o "output_dir"

python train.py --emb "../preprocessed_data/food/w2v_embedding" --domain "food" -o "output_dir"


#test
"""
python evaluation.py --domain "hotel" -o "output_dir" 
"""