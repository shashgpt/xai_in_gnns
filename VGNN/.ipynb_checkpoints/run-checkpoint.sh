for t in 0
do
    for lab in {0..7}
    do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python train.py --trial ${t} --label ${lab}
    done
done