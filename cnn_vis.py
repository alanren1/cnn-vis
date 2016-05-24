import sys
import argparse, os, tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize, imsave, imread
from scipy.ndimage.filters import gaussian_filter

import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
site.addsitedir(settings.caffe_root)
pycaffe_root = settings.caffe_root # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe
import scipy.ndimage as nd
import scipy.misc
import scipy.io

def get_code(data, layer="fc8"):
  '''
  Get a code from an image.
  '''

  # set up the inputs for the net: 
  # batch_size = 1
  # image_size = (3, 227, 227)
  # images = np.zeros((batch_size,) + image_size, dtype='float32')

  # in_image = scipy.misc.imread(path)

  # print "in_image", in_image.shape
  
  # in_image = in_image[:227,:227,:]
  # in_image = scipy.misc.imresize(in_image, (image_size[1], image_size[2]))

  # for ni in range(images.shape[0]):
  #   images[ni] = np.transpose(in_image, (2, 0, 1))

  # RGB to BGR, because this is what the net wants as input
  # data = images[:,::-1] 

  # subtract the ImageNet mean
  matfile = scipy.io.loadmat('ilsvrc_2012_mean.mat')
  image_mean = matfile['image_mean']
  # topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
  topleft = (14, 14)
  image_size = (3, 227, 227)
  image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
  del matfile
  data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

  #initialize the caffenet to extract the features
  # caffe.set_mode_cpu() # replace by caffe.set_mode_gpu() to run on a GPU
  caffenet = caffe.Net(settings.encoder_definition, settings.encoder_path, caffe.TEST)

  # run caffenet and extract the features
  caffenet.forward(data=data)
  feat = np.copy(caffenet.blobs[layer].data)
  del caffenet

  print "feat shape", feat.shape

  zero_feat = feat[0].copy()[np.newaxis]

  print "feat shape", zero_feat.shape


  return zero_feat, data

def make_step_encoder(net, image, end='fc8'): # xy=0, step_size=1.5, , unit=None):
  '''Basic gradient ascent step.'''

  src = net.blobs['data'] # input image is stored in Net's 'data' blob
  dst = net.blobs[end]

  print "make_step_encoder image", image.shape

  acts = net.forward(data=image, end=end)

  grad_clip = 15.0
  l1_weight = 1.0
  l2_weight = 1.0

  target_data = net.blobs[end].data.copy()
  target_diff = -l1_weight * np.abs(target_data)
  target_diff -= l2_weight * np.clip(target_data, -grad_clip, grad_clip)
  dst.diff[...] = target_diff

  # Get back the gradient at the optimization layer
  diffs = net.backward(start=end, diffs=['data'])

  return src.diff.copy()

  # g = diffs['data'][0]

  # grad_norm = norm(g)

  # return src.diff.copy()


def rmsprop(dx, cache=None, decay_rate=0.95):
  """
  Use RMSProp to compute a step from gradients.

  Inputs:
  - dx: numpy array of gradients.
  - cache: numpy array of same shape as dx giving RMSProp cache
  - decay_rate: How fast to decay cache

  Returns a tuple of:
  - step: numpy array of the same shape as dx giving the step. Note that this
    does not yet take the learning rate into account.
  - cache: Updated RMSProp cache.
  """
  if cache is None:
    cache = np.zeros_like(dx)
  cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
  step = -dx / np.sqrt(cache + 1e-8)
  return step, cache


def get_cnn_grads(encoder, decoder, topleft, cur_img, regions, net, target_layer, step_type='amplify_layer', **kwargs):
  """
  Inputs:
  - cur_img: 3 x H x W
  - regions: Array of (y0, y1, x0, x1); must all have same shape as input to CNN
  - target_layer: String
  
  Returns:
  - grads: N x 3 x h x w array where grads[i] is the image gradient for regions[i] of cur_img
  """
  cur_batch = np.zeros_like(net.blobs['data'].data)
  batch_size = cur_batch.shape[0]
  next_idx = 0
  
  def run_cnn(data):

    # data (1, 3, 227, 227)
    # print "data ", data.shape
    output_layer = "deconv0"
    image_size = (227, 227, 3)
    topleft = (14, 14)

    code, _ = get_code(data, layer="fc6")

    print ">>> ", code.shape

    # 1. pass the pool5 code to decoder to get an image x0
    generated = decoder.forward(feat=code)
    x0 = generated[output_layer]   # 256x256

    # Crop from 256x256 to 227x227
    cropped_x0 = x0.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

    # 2. pass the image x0 to AlexNet to maximize an unit k
    # 3. backprop the activation from AlexNet to the image to get an updated image x
    g = make_step_encoder(encoder, cropped_x0, end="fc8") # xy=0, step_size, , unit=unit)

    return g

    '''
    ##################################################################
    # Convert from BGR to RGB because TV works in RGB
    x = x[:,::-1, :, :]

    # 4. Place the changes in x (227x227) back to x0 (256x256)
    updated_x0 = x0.copy()
    # Crop and convert image from RGB back to BGR
    updated_x0[:,::-1,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = x.copy()

    # 5. backprop the image to encoder to get an updated pool5 code
    grad_norm_decoder, updated_code = make_step_decoder(decoder, updated_x0, x0, step_size, start=start_layer, end=output_layer)
    ##################################################################
    '''

    # if upper_bound != None:
    #   print "bounding ----"
    #   updated_code = np.maximum(updated_code, lower_bound) 
    #   updated_code = np.minimum(updated_code, upper_bound) 

    # Update code
    # src.data[:] = updated_code
    # net.forward(data=data)

    # if step_type == 'amplify_layer':
    #   l1_weight = kwargs.get('L1_weight', 1.0)
    #   l2_weight = kwargs.get('L2_weight', 1.0)
    #   grad_clip = kwargs.get('grad_clip', 5)
    #   target_data = net.blobs[target_layer].data.copy()
    #   target_diff = -l1_weight * np.abs(target_data)
    #   target_diff -= l2_weight * np.clip(target_data, -grad_clip, grad_clip)
    #   net.blobs[target_layer].diff[...] = target_diff
    
    # net.backward(start=target_layer)
    # return net.blobs['data'].diff.copy()
  
  grads = []
  for region in regions:
    y0, y1, x0, x1 = region
    cur_batch[next_idx] = cur_img[0, :, y0:y1, x0:x1]
    next_idx += 1
    if next_idx == batch_size:
      grads.append(run_cnn(cur_batch))
      next_idx = 0
  if next_idx > 0:
    grad = run_cnn(cur_batch)
    grads.append(grad[:next_idx])
  
  vgrads = np.vstack(grads)
  return vgrads


def img_to_uint(img, mean_img=None, rescale=False):
  """
  Do post-processing to convert images from caffe format to something more reasonable.

  Inputs:
  - img: numpy array of shape (1, C, H, W)
  - mean_img: numpy array giving a mean image to add in

  Returns:
  A version of img that can be saved to disk or shown with matplotlib
  """
  if mean_img is not None:
    # Be lazy and just add the mean color
    img = 1.2 * img + mean_img.mean()

  # Renormalize so everything is in the range [0, 255]
  if rescale:
    low, high = img.min(), img.max()
  else:
    low, high = 0, 255
  # low = max(img.mean() - 2.5 * img.std(axis=None), img.min())
  # high = min(img.mean() + 2.5 * img.std(axis=None), img.max())
  img = np.clip(img, low, high)
  img = 255.0 * (img - low) / (high - low)

  # Squeeze out extra dimensions and flip from (C, H, W) to (H, W, C)
  img = img.squeeze().transpose(1, 2, 0)

  # Caffe models are trained with BGR; flip to RGB
  img = img[:, :, [2, 1, 0]]

  # finally convert to uint8
  return img.astype('uint8')


def uint_to_img(uint_img, mean_img=None):
  """
  Do pre-processing to convert images from a normal format to caffe format.
  """
  img = uint_img.astype('float')
  img = img[:, :, [2, 1, 0]]
  img = img.transpose(2, 0, 1)
  img = img[np.newaxis, :, :, :]
  if mean_img is not None:
    img = img - mean_img.mean()
  return img


def resize_img(img, new_size, mean_img=None):
  img_uint = img_to_uint(img, mean_img)
  img_uint_r = imresize(img_uint, new_size, interp='bicubic')
  img_r = uint_to_img(img_uint_r, mean_img)
  return img_r
  high, low = img.max(), img.min()
  img_shifted = 255.0 * (img - low) / (high - low)
  img_uint = img_shifted.squeeze().transpose(1, 2, 0).astype('uint8')
  img_uint_r = imresize(img_uint, new_size)
  img_shifted_r = img_uint_r.astype(img.dtype).transpose(2, 0, 1)[None, :, :, :]
  img_r = (img_shifted_r / 255.0) * (high - low) + low
  return img_r


def write_temp_deploy(source_prototxt, batch_size):
  """
  Modifies an existing prototxt by adding force_backward=True and setting
  the batch size to a specific value. A modified prototxt file is written
  as a temporary file.
  
  Inputs:
  - source_prototxt: Path to a deploy.prototxt that will be modified
  - batch_size: Desired batch size for the network
  
  Returns:
  - path to the temporary file containing the modified prototxt
  """
  _, target = tempfile.mkstemp()
  with open(source_prototxt, 'r') as f:
    lines = f.readlines()
  force_back_str = 'force_backward: true\n'
  if force_back_str not in lines:
    lines.insert(1, force_back_str)

  found_batch_size_line = False
  with open(target, 'w') as f:
    for line in lines:
      if line.startswith('input_dim:') and not found_batch_size_line:
        found_batch_size_line = True
        line = 'input_dim: %d\n' % batch_size
      f.write(line)
  
  return target


def get_ranges(total_length, region_length, num):
  starts = np.linspace(0, total_length - region_length, num)
  starts = [int(round(s)) for s in starts]
  ranges = [(s, s + region_length) for s in starts]
  return ranges


def check_ranges(total_length, ranges):
  """
  Check to make sure the given ranges are valid.
  
  Inputs:
  - total_length: Integer giving total length
  - ranges: Sorted list of tuples giving (start, end) for each range.
  
  Returns: Boolean telling whether ranges are valid.
  """
  # The start of the first range must be 0
  if ranges[0][0] != 0:
    return False
  
  # The end of the last range must fill the length
  if ranges[-1][1] != total_length:
    return False
  
  for i, cur_range in enumerate(ranges):
    # The ranges must be distinct
    if i + 1 < len(ranges) and cur_range[0] == ranges[i + 1][0]:
      return False
    # The ranges must cover all the pixels
    if i + 1 < len(ranges) and cur_range[1] < ranges[i + 1][0]:
      return False
    # Each range should not overlap with its second neighbor
    if i + 2 < len(ranges) and cur_range[1] >= ranges[i + 2][0]:
      return False
  return True


def get_best_ranges(total_length, region_length):
  """
  Get the first packing that is valid.
  """
  max_num = 1000 # this should be enough for anyone ...
  num = 1
  while True:
    ranges = get_ranges(total_length, region_length, num)
    if check_ranges(total_length, ranges):
      return ranges
    else:
      if num > max_num:
        return None
    num = num + 1
  return None


def get_regions(total_size, region_size):
  print 'total_size: ', total_size
  print 'region_size: ', region_size
  H, W = total_size
  h, w = region_size
  
  y_ranges = get_best_ranges(H, h)
  x_ranges = get_best_ranges(W, w)

  regions_even = []
  regions_odd = []
  all_regions = []
  for i, x_range in enumerate(x_ranges):
    for j, y_range in enumerate(y_ranges):
      region = (y_range[0], y_range[1], x_range[0], x_range[1])
      if i % 2 == j % 2:
        regions_even.append(region)
      else:
        regions_odd.append(region)
  return regions_even, regions_odd


def count_regions_per_pixel(total_size, regions):
  counts = np.zeros(total_size)
  for region in regions:
    y0, y1, x0, x1 = region
    counts[y0:y1, x0:x1] += 1
  return counts


def get_base_size(net_size, initial_image):
  if initial_image is None:
    return net_size[2:]
  else:
    img = imread(initial_image)
    return img.shape[:2]


def get_size_sequence(base_size, initial_size, final_size, num_sizes, resize_type):
  base_h, base_w = base_size
  
  def parse_size_str(size_str):
    if size_str is None:
      return base_size
    elif size_str.startswith('x'):
      scale = float(size_str[1:])
      h = int(scale * base_h)
      w = int(scale * base_w)
      return h, w
    elif 'x' in size_str:
      h, w = size_str.split('x')
      return int(h), int(w)
  
  initial_h, initial_w = parse_size_str(initial_size)
  final_h, final_w = parse_size_str(final_size)
  
  if num_sizes == 1:
    return [(initial_h, initial_w)]
  else:
    if resize_type == 'geometric':
      h0, h1 = np.log10(initial_h), np.log10(final_h)
      w0, w1 = np.log10(initial_w), np.log10(final_w)
      heights = np.logspace(h0, h1, num_sizes)
      widths = np.logspace(w0, w1, num_sizes)
    elif resize_type == 'linear':
      heights = np.linspace(initial_h, final_h, num_sizes)
      widths = np.linspace(initial_w, final_w, num_sizes)
    else:
      raise ValueError('Invalid resize_type "%s"' % resize_type)
    heights = np.round(heights).astype('int')
    widths = np.round(widths).astype('int')
    return zip(heights, widths)

def get_shape(data_shape):
  if len(data_shape) == 4:
    # Return (227, 227) from (1, 3, 227, 227) tensor
    size = (data_shape[2], data_shape[3])
  else:
    raise Exception("Data shape invalid.")

  return size

def initialize_img(net_size, initial_image, initial_size, mean_img, scale, blur):
  _, C, H, W = net_size

  def init_size_fn(h, w):
    if initial_size is None:
      return h, w
    elif initial_size.startswith('x'):
      scale = float(initial_size[1:])
      return int(scale * h), int(scale * w)
    elif 'x' in initial_size:
      h, w = initial_size.split('x')
      return int(h), int(w)
  
  if initial_image is not None:
    init_img = imread(initial_image)
    init_h, init_w = init_img.shape[:2]
    init_h, init_w = init_size_fn(init_h, init_w)
    init_img = imresize(init_img, (init_h, init_w))
    init_img = uint_to_img(init_img, mean_img)
  else:
    init_h, init_w = init_size_fn(H, W)
    init_img = scale * np.random.randn(1, C, init_h, init_w)
    init_img_uint = img_to_uint(init_img, mean_img)
    init_img_uint_blur = gaussian_filter(init_img_uint, sigma=blur)
    init_img = uint_to_img(init_img_uint_blur, mean_img)

  return init_img


def build_parser():
  parser = argparse.ArgumentParser()
  
  # CNN options
  parser.add_argument('--deploy_txt', default=settings.encoder_definition)
  parser.add_argument('--caffe_model', default=settings.encoder_path)
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--mean_image', default="/home/anh/src/caffe-fr-chairs/python/caffe/imagenet/ilsvrc_2012_mean.npy")
  parser.add_argument('--gpu', type=int, default=1)
  
  # Image options
  parser.add_argument('--image_type', default='amplify_layer',
                      choices=['amplify_layer', 'amplify_neuron'])
  parser.add_argument('--target_layer', default='inception_4d/3x3_reduce')
  parser.add_argument('--target_neuron', default=0, type=int)

  # Initialization options
  parser.add_argument('--initial_image', default=None)
  parser.add_argument('--initialization_scale', type=float, default=1.0)
  parser.add_argument('--initialization_blur', type=float, default=0.0)

  # Resize options
  parser.add_argument('--initial_size', default=None)
  parser.add_argument('--final_size', default=None)
  parser.add_argument('--num_sizes', default=1, type=int)
  parser.add_argument('--resize_type', default='geometric',
                      choices=['geometric', 'linear'])
  
  # Optimization options
  parser.add_argument('--learning_rate', type=float, default=1.0)
  parser.add_argument('--decay_rate', type=float, default=0.95)
  parser.add_argument('--learning_rate_decay_iter', type=int, default=100)
  parser.add_argument('--learning_rate_decay_fraction', type=float, default=1.0)
  parser.add_argument('--num_steps', type=int, default=1000)
  parser.add_argument('--use_pixel_learning_rates', action='store_true')
  
  # Options for layer amplification
  parser.add_argument('--amplify_l1_weight', type=float, default=1.0)
  parser.add_argument('--amplify_l2_weight', type=float, default=1.0)
  parser.add_argument('--amplify_grad_clip', type=float, default=5.0)

  # P-norm regularization options
  parser.add_argument('--alpha', type=float, default=6.0)
  parser.add_argument('--p_scale', type=float, default=1.0)
  parser.add_argument('--p_reg', type=float, default=1e-4)
  
  # Auxiliary P-norm regularization options
  parser.add_argument('--alpha_aux', type=float, default=6.0)
  parser.add_argument('--p_scale_aux', type=float, default=1.0)
  parser.add_argument('--p_reg_aux', type=float, default=0.0)

  # TV regularization options
  parser.add_argument('--beta', type=float, default=2.0)
  parser.add_argument('--tv_reg', type=float, default=0.5)
  parser.add_argument('--tv_reg_scale', type=float, default=1.0)
  parser.add_argument('--tv_reg_step', type=float, default=0.0)
  parser.add_argument('--tv_reg_step_iter', type=int, default=50)
  parser.add_argument('--tv_grad_operator', default='naive',
                      choices=['naive', 'sobel', 'sobel_squish'])

  # Output options
  parser.add_argument('--output_file', default='out.png')
  parser.add_argument('--output_iter', default=50, type=int)
  parser.add_argument('--show_width', default=5, type=int)
  parser.add_argument('--show_height', default=5, type=int)
  parser.add_argument('--rescale_image', action='store_true')
  parser.add_argument('--iter_behavior', default='save+print')
  
  return parser


def main(args):
  if args.gpu < 0:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    #caffe.set_device(args.gpu)

  # Build the net; paths may have CAFFE_ROOT
  proto_file = os.path.expandvars(args.deploy_txt)
  proto_file = write_temp_deploy(proto_file, args.batch_size)
  caffe_model_file = os.path.expandvars(args.caffe_model)
  net = caffe.Net(proto_file, caffe_model_file, caffe.TEST)

  net_size = net.blobs['data'].data.shape
  C, H, W = net_size[1:]

  mean_img = np.load(os.path.expandvars(args.mean_image))
  init_img = initialize_img(net_size, args.initial_image, args.initial_size, mean_img,
                  args.initialization_scale,
                  args.initialization_blur)
  img = init_img.copy()
  if args.initial_image is None:
    init_img = None

  # Encoder and decoder
  # networks
  decoder = caffe.Net(settings.decoder_definition, settings.decoder_path, caffe.TEST)
  encoder = caffe.Classifier(settings.encoder_definition, settings.encoder_path,
                         mean = np.float32([104.0, 117.0, 123.0]), # ImageNet mean, training set dependent
                         channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

  # Compute top left
  # Get the input and output sizes
  output_layer = 'deconv0'
  data_shape = encoder.blobs['data'].data.shape
  output_shape = decoder.blobs[output_layer].data.shape

  image_size = get_shape(data_shape)
  output_size = get_shape(output_shape)

  # The top left offset that we start cropping the output image to get the 227x227 image
  topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)

  # Get size sequence
  base_size = get_base_size(net_size, args.initial_image)
  print 'base_size is %r' % (base_size,)
  size_sequence = get_size_sequence(base_size,
                                    args.initial_size,
                                    args.final_size,
                                    args.num_sizes,
                                    args.resize_type)
  msg = ('Initial size %r is too small; must be at least %r'
         % (size_sequence[0], (H, W)))
  assert size_sequence[0] >= (H, W), msg
  
  # Run optimization
  for size_idx, size in enumerate(size_sequence):
    size_flag = False
    if size_idx > 0:
      img = resize_img(img, size, mean_img)
      if init_img is not None:
        raw_init = imread(args.initial_image)
        init_img_uint = imresize(raw_init, size)
        init_img = uint_to_img(init_img_uint, mean_img)

    tv_reg = args.tv_reg
    learning_rate = args.learning_rate
    regions = get_regions((img.shape[2], img.shape[3]), (H, W))
    regions_even, regions_odd = regions
    regions_per_pixel = count_regions_per_pixel((img.shape[2], img.shape[3]), regions_even+regions_odd)
    pixel_learning_rates = 1.0 / regions_per_pixel
    caches = {}
    pix_history = defaultdict(list)
    pix = [(100, 100), (200, 200), (100, 200), (200, 100)]
    for t in xrange(args.num_steps):
      for c in [0, 1, 2]:
        for py, px in pix:
          pix_history[(c, py, px)].append(img[0, c, py, px])

      for cur_regions in [regions_even, regions_odd]:
        if len(cur_regions) == 0: continue
        cnn_grad = get_cnn_grads(encoder, decoder, topleft, img, cur_regions, net, args.target_layer,
                       step_type=args.image_type,
                       L1_weight=args.amplify_l1_weight,
                       L2_weight=args.amplify_l2_weight,
                       grad_clip=args.amplify_grad_clip,
                       target_neuron=args.target_neuron)
        for region_idx, region in enumerate(cur_regions):
          y0, y1, x0, x1 = region
          img_region = img[:, :, y0:y1, x0:x1]
          if init_img is not None:
            init_region = init_img[0, :, y0:y1, x0:x1]
          
          dimg = cnn_grad[region_idx]

          cache = caches.get(region, None)
          step, cache = rmsprop(dimg, cache=cache, decay_rate=args.decay_rate)
          caches[region] = cache
          step *= learning_rate
          if args.use_pixel_learning_rates:
            step *= pixel_learning_rates[y0:y1, x0:x1]
          img[:, :, y0:y1, x0:x1] += step

      if (t + 1) % args.tv_reg_step_iter == 0:
        tv_reg += args.tv_reg_step

      if (t + 1) % args.learning_rate_decay_iter == 0:
        learning_rate *= args.learning_rate_decay_fraction

      if (t + 1) % args.output_iter == 0:
        should_plot_pix = 'plot_pix' in args.iter_behavior
        should_show = 'show' in args.iter_behavior
        should_save = 'save' in args.iter_behavior
        should_print = args.iter_behavior

        if False:
          values = [img_region.flatten(),
                    cnn_grad.flatten(),
                    #(args.p_reg * p_grad).flatten(),
                    #(tv_reg * tv_grad).flatten()]
                    (args.p_reg * p_grad + tv_reg * tv_grad).flatten(),
                    step.flatten()]
          names = ['pixel', 'cnn grad', 'reg', 'step']
          subplot_idx = 1
          for i, (name_i, val_i) in enumerate(zip(names, values)):
            for j, (name_j, val_j) in enumerate(zip(names, values)):
              x_min = val_i.min() - 0.1 * np.abs(val_i.min())
              x_max = val_i.max() + 0.1 * np.abs(val_i.max())
              y_min = val_j.min() - 0.1 * np.abs(val_j.min())
              y_max = val_j.max() + 0.1 * np.abs(val_j.max())
              plt.subplot(len(values), len(values), subplot_idx)
              plt.scatter(val_i, val_j)
              plt.plot(np.linspace(x_min, x_max), np.linspace(x_min, x_max), '-k')
              plt.plot(np.linspace(x_min, x_max), -np.linspace(x_min, x_max), '-k')
              plt.xlim([x_min, x_max])
              plt.ylim([y_min, y_max])
              plt.xlabel(name_i)
              plt.ylabel(name_j)
              subplot_idx += 1
          plt.gcf().set_size_inches(15, 15)
          plt.show()

        if should_plot_pix:
          for p, h in pix_history.iteritems():
            plt.plot(h)
          plt.show()
        if should_print:
          print ('Finished iteration %d / %d for size %d / %d' % 
                (t + 1, args.num_steps, size_idx + 1, len(size_sequence)))
          if args.image_type == 'amplify_neuron':
            target_blob = net.blobs[args.target_layer]
            neuron_val = target_blob.data[:, args.target_neuron].mean()
            print 'mean neuron val: ', neuron_val
          print 'mean cnn_grad: ', np.abs(cnn_grad).mean()
          print 'step mean, median: ', np.abs(step).mean(), np.median(np.abs(step))
          print 'image mean, std: ', img.mean(), img.std()
          print 'mean step / val: ', np.mean(np.abs(step) / np.abs(img_region))
        img_uint = img_to_uint(img, mean_img, rescale=args.rescale_image)
        if should_show:
          plt.imshow(img_uint, interpolation='none')
          plt.axis('off')
          plt.gcf().set_size_inches(args.show_width, args.show_height)
          plt.show()
        if should_save:
          name, ext = os.path.splitext(args.output_file)
          filename = '%s_%d_%d%s' % (name, size_idx + 1, t + 1, ext)
          imsave(filename, img_uint)
          
          
  img_uint = img_to_uint(img, mean_img, rescale=args.rescale_image)
  imsave(args.output_file, img_uint)


if __name__ == '__main__':
  parser = build_parser()
  args = parser.parse_args()
  main(args)
