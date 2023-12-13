lrset=("0.0005" "0.0007" "0.0009" "0.001" "0.003" "0.005" "0.007" "0.009" "0.01")

for lab in {0..15}
do
    for t in 0 1 2
    do
        for lr in ${lrset[@]}
        do
            PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python main.py --trial ${t} --label ${lab} --num_layers 2 --num_heads 8 --embedding_size 256 --lr ${lr}
        done
    done
done

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python main.py --trial 0 --label 10 --num_layers 2 --num_heads 8 --embedding_size 256 --lr 0.0005