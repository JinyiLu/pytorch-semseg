for dir in $1/*
do
    if [[ -d $dir ]]; then
        # echo $(basename $dir)
        python test.py --model_path fcn8s_dsbowl_best_model.pkl --img_path $dir/images/$(basename $dir).png --out_path $2$(basename $dir).png
    fi
done