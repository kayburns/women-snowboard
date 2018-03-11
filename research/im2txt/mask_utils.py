import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_image_masks(image, mask_dir, mask_size=32, pad=0, size=299):
    """
    Creates a directory of masked versions of
    the original image path.
    
    image -- pathname of image
    mask_dir -- directory to store masked images;
        must exist
    """
    im = Image.open(image)
    im = im.resize((size, size))
    image_array = np.array(im)
    if image_array.ndim == 2:
      image_array = np.tile(im[..., np.newaxis], (1, 1, 3))
    H, W, C = image_array.shape
     
    stride=mask_size
 
    masked_images = []
    mask = np.zeros((mask_size, mask_size, C))
    for i in range(H / stride):
        for j in range(W / stride):
            masked_image = image_array.copy()
            masked_image[i*mask_size:(i+1)*mask_size, j*mask_size:(j+1)*mask_size, :] = mask
            masked_images.append(masked_image)
            
        # add partial mask along vertical edge
        if (W % stride):
            start = (W // mask_size) * mask_size
            clipped_width = W % stride
            masked_image = image_array.copy()
            masked_image[i*mask_size:(i+1)*mask_size, start:, :] = mask[:, :clipped_width]
            masked_images.append(masked_image)
    
    # add partial mask along horizontal edge
    if (H % stride):
        start = (H // mask_size) * mask_size
        clipped_height = H % stride
        for j in range(W / stride):
            masked_image = image_array.copy()
            masked_image[start:, j*mask_size:(j+1)*mask_size, :] = mask[:clipped_height, :]
            masked_images.append(masked_image)
            
    # add partial mask to bottom right
    if (W % stride) and (H % stride):
        clipped_height = H % stride
        clipped_width = W % stride
        vertical_start = (H // mask_size) * mask_size
        horizontal_start = (W // mask_size) * mask_size
        masked_image[vertical_start:, horizontal_start:] = mask[:clipped_height, :clipped_width]
        masked_images.append(masked_image)
        
    # save images to file
    i = 0
    for masked_image in masked_images:
        im = Image.fromarray(masked_image)
        im_name = 'mask_%012d.jpg' % i
        im.save(mask_dir + im_name)
        i += 1
