for lab in {1..15}
do
    python preprocess_mimic.py --label ${lab}
done