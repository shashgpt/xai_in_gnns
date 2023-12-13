for t in 0
do
    for lab in 15 14 13 12 11 10 9 8
    do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python train.py --trial ${t} --label ${lab}
    done
done
