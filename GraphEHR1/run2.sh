lrset=("0.0005" "0.001")
nlset=("1" "3")

for t in 0 1 2
do
    for lab in {0..15}
    do
        for lr in ${lrset[@]}
        do
            for nl in ${nlset[@]}
            do
                PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python main.py --trial ${t} --label ${lab} --num_layers ${nl} --num_heads 8 --embedding_size 256 --lr ${lr}
            done
        done
    done
done
