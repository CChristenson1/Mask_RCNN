"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
from PIL import Image, ImageDraw

from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/Maize/")


############################################################
#  Configurations
############################################################

class MaizeConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "Maize"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + Kernel + Ear

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 
    VALIDATION_STEPS = 

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 1000

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    # Class Weights dict for when you have an imbalance in class frequency
    # Can be used for R-CNNN training setup
    CLASS_WEIGHTS = None

class MaizeInferenceConfig(MaizeConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize images for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.5


############################################################
#  Dataset  
# 
# Possily change load dataset so you can load detection set as well, 
# should just have an "assertion" that will change type of dataset if an
# annotation json id provided
############################################################

class MaizeDataset(utils.Dataset):

    def load_Maize(self, annotation_json, images_dir):
        # Load json from file
        json_file = open(annotation_json)
        groundtruth_data = json.load(json_file)
        images = groundtruth_data['images']
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "Maize"
        for category in groundtruth_data['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations_index = {}
        if 'annotations' in groundtruth_data:
          print("Found groundtruth annotations. building annotations index.")
          for annotation in groundtruth_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_index:
              annotations_index[image_id] = []
            annotations_index[image_id].append(annotation)
        missing_annotation_count = 0
            
        
                
        seen_images = {}
        # Get all images and add them to the dataset
        for idx, image in enumerate(images):
            if idx % 10 == 0:
                print('On image {} of {}'.format(idx,len(images)))
            
            image_id = image['id']
            # Check for dulpicate Images and get image info from each image
            # Image_id, image_path, image_height, image_width
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
            image_path = os.path.abspath(os.path.join(images_dir, image_file_name))

            # Check for Images missing annotations and get annotation data for each image            
            if image_id not in annotations_index:
              missing_annotation_count += 1
              continue
            else:
              annotations = annotations_index[image_id]
                
        # Add the image using the base method from utils.Dataset
            self.add_image(
                source=source_name,
                image_id=image_id,
                path=image_path,
                height=image_height,
                width=image_width,
                annotations=annotations,
            )

        print('{} images are missing annotations.'.format(missing_annotation_count))

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            clear = 0
            for segmentation in annotation['segmentation']:
              if clear == 0:
                mask_draw.polygon(segmentation, outline=1, fill=1)
                clear += 1
              elif clear > 0:
                mask_draw.polygon(segmentation, outline=0, fill=0)
                clear += 1
            bool_array = np.array(mask) > 0
            instance_masks.append(bool_array)
            class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
       
        return mask, class_ids
       

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = MaizeDataset()
    dataset_train.load_Maize(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MaizeDataset()
    dataset_val.load_Maize(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=20,
                augmentation=augmentation,
                layers='all')


############################################################
#  Utils 
############################################################

def get_splits(image_width, split_number, overlap):
# A really complicated fuction to get the split sections, with overlap
    image_splits = []
    total_image_width = image_width
    overlap_width = overlap

    if split_number == 1:
        image_splits.append([0, total_image_width]) 

    # This will be the most used case, as of now, with a split of 3 sub-images.
    # In this case, since a lot of the ear images have significant space on the
    # left and right, I want the center sub-image to not be too big. To avoid
    # this, I'll do the overlaps from the left and right images and leave the
    # center image unchanged.` 
    elif split_number == 3:
        # Here's the split width if there's no overlap (note: probably will
        # need to do something about rounding errors here with certain image
        # widths).
        no_overlap_width = int(total_image_width / split_number)
        
        # Left split. The left side of the left split will always be zero.
        left_split = []
        left_split.append(0)

        # The other side of the left split will be the width (minus 1 to fix
        # the 0 index start) plus the overlap
        left_split.append(no_overlap_width + overlap_width)
        image_splits.append(left_split)

        # The middle has no overlap in this case
        middle_split = []
        middle_split.append(no_overlap_width-(overlap_width/2))
        middle_split.append((no_overlap_width * 2)+(overlap_width/2))
        image_splits.append(middle_split)

        # The right split is the opposite of the left split
        right_split = []
        right_split.append((2 * no_overlap_width) - overlap_width)
        right_split.append(total_image_width)
        image_splits.append(right_split)

    return image_splits


def spliting_image(image_np_array, split_list):
# The fuction that actually splits the images
    print(image_np_array.shape)
    array_list = []

    for split_nums in split_list:
        left_border = int(split_nums[0])
        right_border = int(split_nums[1])
        print("Borders:{}, {}".format(left_border,right_border))
        sub_array = image_np_array[:,left_border:right_border,:]
        array_list.append(sub_array)

    return(array_list)


def fix_relative_coord(output_dict, list_of_splits, image_position):
    output_dict_adj = output_dict

    # First we get a constant adjustment for the "image position". The
    # adjustment is where the left side of the current image starts, relative
    # to the entire image. We can get this from the list_of_splits.
    position_adjustment = list_of_splits[image_position][0] 

    # Now we adjust the x coordinates of the 'rois' ndarray, We
    # don't need to adjust the y coordinates because we only split on the x. If
    # later I add splitting on y, then the y coordinates need to be adjusted.
    # This adjustment "shrinks" the relative coordinates down.
    adjusted_boxes = output_dict['rois']

    #adjusted_boxes[:,[1,3]] = adjusted_boxes[:,[1,3]] *(split_width / image_width)

    # Adding the adjustment for which split image it is (the first image
    # doesn't need adjustment, hence the if statement).
    if image_position > 0:
        adjusted_boxes[:,[1,3]] = adjusted_boxes[:,[1,3]] + position_adjustment

    # Now adding back in the adjusted boxes to the original ndarray
    output_dict_adj['rois'] = adjusted_boxes

    return(output_dict_adj)


def pad_mask(results,list_of_splits, split_number):
  output_adj_dict = results
  # Getting the image width out of the list of splits (it's the right side of
  # the last split).
  image_width = list_of_splits[-1][1]
  
  padded_masks = []
  r = results
  if split_number == 0:
    added_width = int(image_width-list_of_splits[0][1])
    height = 450
    padding_array = np.zeros([height,added_width])
    for i in range(len(r['masks'][1,1,:])):
      combined = np.concatenate((r['masks'][:,:,i].astype(np.uint8),padding_array), axis=1)
      padded_masks.append(combined) 
                            
  elif split_number == 1:
    added_width_l = int(list_of_splits[1][0])
    added_width_r = int(image_width-list_of_splits[1][1])
    height = 450
    padding_array_l = np.zeros([height,added_width_l])
    padding_array_r = np.zeros([height,added_width_r])
    for i in range(len(r['masks'][1,1,:])):
      combined = np.concatenate((padding_array_r,r['masks'][:,:,i].astype(np.uint8),padding_array_l), axis=1)
      padded_masks.append(combined)  

  elif split_number == 2:
    added_width = int(list_of_splits[2][0])
    height = 450
    padding_array = np.zeros([height,added_width])
    for i in range(len(r['masks'][1,1,:])):
      combined = np.concatenate((padding_array,r['masks'][:,:,i].astype(np.uint8)), axis=1)
      padded_masks.append(combined) 

  # Check shape because splits with no instance dectected have the wrong shape

  output_adj_dict['masks'] = padded_masks

  return(output_adj_dict) 


def do_non_max_suppression(results):
    # The actual nms comes from Tensorflow
    nms_vec_ndarray = utils.non_max_suppression(
        results['rois'],
        results['scores'],
        threshold=0.5)
    
    print("the nms ndarray is:")
    print(nms_vec_ndarray)
    print(len(nms_vec_ndarray))
    print("the length of the input array is:")
    print(len(results['rois']))
    print("\n\n\n")

    # Indexing the input dictionary with the output of non_max_suppression,
    # which is the list of boxes (and score, class) to keep.
    out_dic = results.copy()
    out_dic['rois'] = results['rois'][nms_vec_ndarray].copy() 
    out_dic['scores'] = results['scores'][nms_vec_ndarray].copy() 
    out_dic['class_ids'] = results['class_ids'][nms_vec_ndarray].copy()
    results['masks'] = np.transpose(results['masks'])
    out_dic['masks'] = results['masks'][nms_vec_ndarray].copy()  
    out_dic['masks'] = np.transpose(out_dic['masks'])

    # Change to output dictionary
    return(out_dic)


def inference(model, image, class_names):
  ## Split the Image with an overlap
  # Determine the subdividsion of the splits, based on number of splits wanted.
  # The splits will be a list of set of two numbers, the lower and upper bounds of the splits.
  splits = get_splits(image.shape[1], 3, 10)

  # Actually split the image into the subdivision determined earlier
  split_images = spliting_image(image, splits)

  # Uncomment to Run detection and Visualize results of a single split
  #results = model.detect([image], verbose=1)
  #r = results[0]
  #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
  #                            val_dataset.class_names, r['scores'])

  image_split_number = 0
  ## Run Dectection on each Split
  for split_image in split_images:
    results = model.detect([split_image])
    r = results[0]
    visualize.display_instances(split_images[image_split_number], r['rois'], r['masks'], r['class_ids'], class_names)
    ## Fix relative coordinates
    adjusted_result = fix_relative_coord(results[0],splits, image_split_number)
    adjusted_result = pad_mask(adjusted_result,splits,image_split_number)
    ## Combine the predicted results
    if image_split_number == 0:
      output_result = adjusted_result
    else:
      output_result['rois'] = np.concatenate((output_result['rois'], adjusted_result['rois']))
      output_result['class_ids'] = np.concatenate((output_result['class_ids'], adjusted_result['class_ids']))
      output_result['scores'] = np.concatenate((output_result['scores'], adjusted_result['scores']))
      if len(adjusted_result['masks']) == 0:
        continue
      output_result['masks'] = np.concatenate((output_result['masks'], adjusted_result['masks']))
    image_split_number += 1

  #Visualize combined image witout NMS
  output_result['masks'] = np.transpose(output_result['masks'])
  output_result['masks'] = np.swapaxes(output_result['masks'],0,1)
  visualize.display_instances(image, output_result['rois'], output_result['masks'], output_result['class_ids'], class_names)

  ## Remove redundant rois/mask
  output_result = do_non_max_suppression(output_result)


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = MaizeDataset()
    dataset.load_Maize(dataset_dir)
    dataset.prepare()
    class_names = ['BG','Kernel','Ear']

    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        
        # Detect objects
        r = inference(model, [image],class_names)

        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for kernel counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MaizeConfig()
    else:
        config = MaizeInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
