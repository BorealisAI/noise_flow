import cv2
import numpy as np
import scipy.io as sio


def read_metadata(metadata_file_path):
    # metadata
    meta = sio.loadmat(metadata_file_path)
    meta = meta['metadata'][0, 0]
    # black_level = float(meta['black_level'][0, 0])
    # white_level = float(meta['white_level'][0, 0])
    bayer_pattern = get_bayer_pattern(meta)  # meta['bayer_pattern'].tolist()
    bayer_2by2 = (np.asarray(bayer_pattern) + 1).reshape((2, 2)).tolist()
    # nlf = meta['nlf']
    # shot_noise = nlf[0, 2]
    # read_noise = nlf[0, 3]
    wb = get_wb(meta)
    # cst1 = meta['cst1']
    cst1, cst2 = get_csts(meta)
    # cst2 = cst2.reshape([3, 3])  # use cst2 for rendering, TODO: interpolate between cst1 and cst2
    iso = get_iso(meta)
    cam = get_cam(meta)
    return meta, bayer_2by2, wb, cst2, iso, cam


def get_iso(metadata):
    try:
        iso = metadata['ISOSpeedRatings'][0][0]
    except:
        try:
            iso = metadata['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        except:
            raise Exception('ISO not found.')
    return iso


def get_cam(metadata):
    model = metadata['Make'][0]
    # cam_ids = [0, 1, 3, 3, 4]  # IP, GP, S6, N6, G4
    cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
    return cam_dict[model]


def get_bayer_pattern(metadata):
    bayer_id = 33422
    bayer_tag_idx = 1
    try:
        unknown_tags = metadata['UnknownTags']
        if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
            bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
        else:
            raise Exception
    except:
        try:
            unknown_tags = metadata['SubIFDs'][0, 0]['UnknownTags'][0, 0]
            if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
                bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
            else:
                raise Exception
        except:
            try:
                unknown_tags = metadata['SubIFDs'][0, 1]['UnknownTags']
                if unknown_tags[1]['ID'][0][0][0] == bayer_id:
                    bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
                else:
                    raise Exception
            except:
                print('Bayer pattern not found. Assuming RGGB.')
                bayer_pattern = [1, 2, 2, 3]
    return bayer_pattern


def get_wb(metadata):
    return metadata['AsShotNeutral']


def get_csts(metadata):
    return metadata['ColorMatrix1'].reshape((3, 3)), metadata['ColorMatrix2'].reshape((3, 3))


def RGGB2Bayer(im):
    # convert RGGB stacked image to one channel Bayer
    bayer = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    bayer[0::2, 0::2] = im[:, :, 0]
    bayer[0::2, 1::2] = im[:, :, 1]
    bayer[1::2, 0::2] = im[:, :, 2]
    bayer[1::2, 1::2] = im[:, :, 3]
    return bayer


def demosaic_CV2(rggb_channels_stack):
    # using opencv demosaic
    bayer = RGGB2Bayer(rggb_channels_stack)
    dem = cv2.cvtColor(np.clip(bayer * 16383, 0, 16383).astype(dtype=np.uint16), cv2.COLOR_BayerBG2RGB_EA)
    dem = dem.astype(dtype=np.float32) / 16383
    return dem


def flip_bayer(image, bayer_pattern):
    if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
    elif (bayer_pattern == [[2, 1], [3, 2]]):
        image = np.fliplr(image)
    elif (bayer_pattern == [[2, 3], [1, 2]]):
        image = np.flipud(image)
    elif (bayer_pattern == [[3, 2], [2, 1]]):
        image = np.fliplr(image)
        image = np.flipud(image)
    else:
        import pdb
        pdb.set_trace()
        print('Unknown Bayer pattern.')
    return image


def unflip_bayer(image, bayer_pattern):
    """Re-arrange a Bayer raw image ordered as RGGB to have the order specifiec in `bayer_pattern`"""
    if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
    elif (bayer_pattern == [[2, 1], [3, 2]]):
        image = np.fliplr(image)
    elif (bayer_pattern == [[2, 3], [1, 2]]):
        image = np.flipud(image)
    elif (bayer_pattern == [[3, 2], [2, 1]]):
        image = np.flipud(image)
        image = np.fliplr(image)
    return image


def stack_rggb_channels(raw_image):
    """Stack the four RGGB channels of a Bayer raw image along a third dimension"""
    height, width = raw_image.shape
    channels = []
    for yy in range(2):
        for xx in range(2):
            raw_image_c = raw_image[yy:height:2, xx:width:2].copy()
            channels.append(raw_image_c)
    channels = np.stack(channels, axis=-1)
    return channels


def simple_raw_to_rgb(rggb_channels_stack, black=0, wb=np.array([[1], [1], [1]])):
    """Simply, return an 3-channel RGB image by averaging the 2 green channels
    and stacking the resulting 3 channels along a third axis"""
    # channels = stack_rggb_channels(raw_image_rggb)
    channels_rgb = rggb_channels_stack[:, :, :3]
    channels_rgb[:, :, 0] = (channels_rgb[:, :, 0] - black) / wb[0][0]
    channels_rgb[:, :, 1] = (np.mean(rggb_channels_stack[:, :, 1:3], axis=2) - black) / wb[1][0]
    channels_rgb[:, :, 2] = (rggb_channels_stack[:, :, 3] - black) / wb[2][0]
    return channels_rgb


def simple_raw_rggb_stack_to_rgb(raw_rggb_stack):
    """Simply, return an 3-channel RGB image by averaging the 2 green channels in the stack of RGGB channels"""
    channels_rgb = raw_rggb_stack[:, :, :3]
    channels_rgb[:, :, 1] = np.mean(raw_rggb_stack[:, :, 1:3], axis=2)
    channels_rgb[:, :, 2] = raw_rggb_stack[:, :, 3]
    return channels_rgb


def space_to_depth(x, block_size):
    height, width = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    depth = block_size * block_size
    y = np.zeros((reduced_height, reduced_width, depth))
    k = 0
    for i in range(block_size):
        for j in range(block_size):
            y[:, :, k] = x[i::block_size, j::block_size]
            k += 1
    return y


def depth_to_space(x, block_size):
    reduced_height, reduced_width, depth = x.shape
    height = reduced_height * block_size
    width = reduced_width * block_size
    y = np.zeros((height, width))
    k = 0
    for i in range(block_size):
        for j in range(block_size):
            y[i::block_size, j::block_size] = x[:, :, k]
            k += 1
    return y
