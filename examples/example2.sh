export LD_LIBRARY_PATH=/home/anh/local/lib:/usr/local/cuda-7.5/targets/x86_64-linux/lib:/home/anh/anaconda/lib:${LD_LIBRARY_PATH}

python cnn_vis.py \
  --image_type=amplify_layer \
  --target_layer=conv5 \
  --gpu=0 \
  --num_steps=100 \
  --batch_size=25 \
  --output_iter=25 \
  --learning_rate=10 \
  --decay_rate=0.25 \
  --alpha=6.0 \
  --p_reg=1e-10 \
  --beta=2.5 \
  --tv_reg=5e-2 \
  --initial_size=270x480 \
  --final_size=270x480 \
  --num_sizes=2 \
  --iter_behavior=print \
  --output_file=alexnet_conv2.png \
  --caffe_model="/home/anh/workspace/upconvnet/encoder.caffemodel" \
  --deploy_txt="/home/anh/workspace/upconvnet/caffenet/encoder.prototxt" \
  --mean_image="/home/anh/src/caffe_latest/python/caffe/imagenet/ilsvrc_2012_mean.npy"
