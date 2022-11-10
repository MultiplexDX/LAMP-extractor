import numpy as np
from imutils.perspective import four_point_transform as imutils_four_point_transform, order_points
import cv2
import numpy as np
from skimage.measure import label, regionprops_table

from loguru import logger
#import matplotlib.pyplot as plt
from pkg_resources import resource_filename
from lamp_extractor.models.mobilenet_v2_02_aug_resume import loader 
from lamp_extractor.timer import measure
from lamp_extractor.utils import draw_keypoints
from skimage.draw import polygon2mask

HUE_CH_MAX = 180

model, model_config, model_transform = None, None, None
if model is None:
    model, model_config, model_transform = loader._load_pyfunc()

@measure
def segment_plate(img, findcorners_mode):
    logger.info(f"{findcorners_mode=}")
    if findcorners_mode == 'static':
        tl = [116, 36]
        tr = [1094, 31]
        br = [1100, 690]
        bl = [123, 690]
        pts = np.array([tl, tr, br, bl], dtype="float32")
        warped_img = four_point_transform(img, pts)
        warped_img = padd_img(warped_img, 0.08)
    elif findcorners_mode == 'net':
        pts = find_corners_by_kpsnet(img).numpy()
        warped_img = four_point_transform(img, pts)
        warped_img = padd_img(warped_img, 0.06)
    else:
        raise NotImplementedError(f"{findcorners_mode=}")

    return warped_img, pts

def padd_img(img, padding):
    h, w = img.shape[:2]
    wpad = int(w * padding)
    hpad = int(h * padding)
    return img[hpad : h - hpad, wpad : w - wpad, :]

def outer_mask(img, pts):
    yx = pts.copy()
    yx[:, [0, 1]] = yx[:, [1, 0]]
    mask = polygon2mask(img.shape[:2], yx)
    mask = np.logical_not(mask)
    return mask

def four_point_transform(image, pts):
    return imutils_four_point_transform(image,pts)

@measure
def find_corners_by_kpsnet(img):
    global model, model_config, model_transform
    output = loader.predict(model, model_transform, model_config, img)
    return output

def draw_grid(img, line_color=(0, 255, 0), thickness=1, pxxstep=50, pxystep=50):
    """(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    """
    x = pxxstep
    y = pxystep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, thickness=thickness)
        x += pxxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, thickness=thickness)
        y += pxystep
    return img


def dominant_color(regionmask, intensity):
    pixels_of_interest = intensity[regionmask > 0]
    values, counts = np.unique(pixels_of_interest, return_counts=True)
    if len(counts) > 0:
        index_of_max_count = counts.argmax()
        dominant_hue = values[index_of_max_count]
        return dominant_hue
    else:
        return None


@measure
def remove_shadows(img, normalize=False):
    """BGR image"""
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )
        # _, norm_img = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, diff_img = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    if normalize:
        result_norm = cv2.merge(result_norm_planes)
        return result_norm
    else:
        result = cv2.merge(result_planes)
        return result

@measure
def predict(
    img,
    config,
    visualize=False,
):
    ROWS = config['rows']
    COLUMNS = config['columns']
    FIND_CORNERS = config['findcorners_mode']

    warped_img, pts = segment_plate(img, findcorners_mode=FIND_CORNERS)
    num_columns = len(COLUMNS) 
    num_rows = len(ROWS)

    # Check cropped image ratio TODO
    expected_ratio = num_rows/num_columns 
    actual_ratio = warped_img.shape[0] / warped_img.shape[1] 
    assert abs(expected_ratio - actual_ratio) < config['pratioThreshold'], f"Cannot find plate in the image. {expected_ratio=}, {actual_ratio=}" 

    # cv2.imwrite("find_plate.png", draw_keypoints(img.copy(), pts, diameter=10))
    # cv2.imwrite("perspective_transform.png", warped_img)
    # Make image dividable by cells width and height
    matrix, classes = classify(warped_img=warped_img, num_rows=num_rows, num_columns=num_columns, params=config)
    return matrix, classes, ROWS, COLUMNS, warped_img

def class_mask(hue_ch, class_name, ranges):
    rmasks = []
    #logger.info(class_name)
    for range in ranges:     
        l = int(range[0] * HUE_CH_MAX) 
        u = int(range[1] * HUE_CH_MAX)
        #logger.info(f"{l=}, {u=}")
        m = ((l <= hue_ch) & (hue_ch <= u))
        rmasks.append(m)
    rmask = np.any(np.stack(rmasks, axis=0), axis=0).astype("float32")
    return rmask

def classify(warped_img, num_rows, num_columns, params):
    CLASSES = np.array(list(params['classes'].keys()))
    h, w = warped_img.shape[:2]
    h_div = (h // num_rows) * num_rows
    w_div = (w // num_columns) * num_columns
    resized_img = cv2.resize(warped_img, (w_div, h_div), interpolation = cv2.INTER_NEAREST)
    #resized_img = remove_shadows(img=resized_img)
    #hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    cells_mask = get_cells_mask(img=resized_img, params=params)
    #assert np.nonzero(cells_mask) > 0, f"Empty plate" 
    #plt.imshow(cells_mask), plt.show()

    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    hue_ch = hsv_img[:, :, 0]
    #plt.imshow(hue_ch * cells_mask), plt.show()
    
    masks = []
    for class_name, ranges in params['classes'].items(): 
        rmask = class_mask(hue_ch, class_name, ranges)
        masks.append(rmask)

    cmask = np.stack(masks, axis=0)
    cmask = cmask * cells_mask[None, :, :]
    cellh, cellw = resized_img.shape[0] // num_rows, resized_img.shape[1] // num_columns 
    cells_patches = get_cells_patches(cmask=cmask, cellh=cellh, cellw=cellw)
    #(8, 12, 4, 16, 16)

    preds = np.count_nonzero(cells_patches, axis=(3, 4)) #=> (8, 12, 4)

    # Set result matrix
    #
    # matrix = np.zeros((num_rows, num_columns))
    matrix = np.argmax(preds, axis=-1) # (8, 12)
    #plt.imshow(matrix), plt.show()
    vmatrix = np.max(preds, axis=-1).astype("float64") #(8, 12)
    matrix[vmatrix < 50] = np.argmax(CLASSES == "EMPTY")
    #plt.imshow(vmatrix), plt.show()

    # INCONCLUSIVE CLASS
    #cmatrix = np.count_nonzero(cells_patches, axis=(2, 3, 4)).astype("float64") #=>(8, 12)
    #prob_matrix = np.divide(vmatrix, cmatrix, out=np.ones_like(vmatrix), where=cmatrix!=0)
    #matrix[prob_matrix < 0.5] = np.argmax(CLASSES == "INCONCLUSIVE")

    # EMPTY CLASS
    #vis = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) * cells_mask[:, :, None]
    #vis = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)[:, :, 0]

    # from matplotlib import pyplot as plt
    # from lamp_extractor.utils import colorize_matrix
    # plt.imshow(colorize_matrix(matrix))
    # #plt.xticks(range(matrix.shape[1]))
    # #plt.yticks(range(matrix.shape[0]))
    # plt.xticks(range(matrix.shape[1]), params['columns'])
    # plt.yticks(range(matrix.shape[0]), params['rows'])
    # plt.savefig('matrix.png'), plt.show()
    
    # cv2.imwrite('cells_segmentation.png', cells_mask*255)
    
    # plt.xticks([])
    # plt.yticks([])
    # cv2.imwrite('grid.png', draw_grid(resized_img.copy(), pxxstep=cellw, pxystep=cellh, thickness=2))
    
    # for index, c in enumerate(cmask):
    #     m = np.repeat(np.expand_dims(c, axis=2), 3, axis=2).astype(np.uint8)
    #     tmp_c_bgr = (m * resized_img)
    #     tmp_c_bgr = tmp_c_bgr + ~m
    #     tmp_c_rgb = cv2.cvtColor(tmp_c_bgr, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(f'cmask_{CLASSES[index]}.png', tmp_c_bgr)
        
    # plt.imshow(matrix), plt.show()
    return matrix, CLASSES

@measure
def get_cells_mask(img, params):
    # Thresholding black spots in image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    vblur = cv2.GaussianBlur(hsv_img[:, :, 2], (3, 3), 0)
    _, vthresh = cv2.threshold(src=vblur,thresh=60, maxval=1,type=cv2.THRESH_BINARY)
    vthresh = cv2.dilate(vthresh, np.ones((3, 3), np.uint8), iterations=1)
    #plt.imshow(vthresh), plt.show()

    # Threshold spots with low color saturation
    sat_ch = hsv_img[:, :, 1] * vthresh
    #plt.imshow(sat_ch), plt.show()
    _, sthresh = cv2.threshold(src=sat_ch,thresh=70, maxval=1,type=cv2.THRESH_BINARY)
    sthresh = cv2.morphologyEx(sthresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    #plt.imshow(sthresh * sat_ch), plt.show()
    thresh = sthresh

    # Remove unknown color
    hue_ch = hsv_img[:, :, 0]

    emask = class_mask(hue_ch=hue_ch, class_name="EMPTY", ranges=params['classes']['EMPTY'])
    thresh *= np.logical_not(emask) 
    #plt.imshow(thresh * hue_ch), plt.show()

    return thresh

@measure
def get_matrix_for(cmask, cellh, cellw, cols, rows):
    matrix = np.zeros((8, 12))
    vmatrix = np.zeros((8, 12))
    cmatrix = np.zeros((8, 12))
    for col_i in range(cols):
        min_x = cellw * col_i
        max_x = min_x + cellw
        for row_i in range(rows):
            min_y = cellh * row_i
            max_y = min_y + cellh
            ccell = cmask[:, min_y:max_y, min_x:max_x]
            # cell.shape = (4, 50, 50)
            pred = np.count_nonzero(ccell, axis=(1, 2))
            total = np.count_nonzero(ccell, axis=0)
            matrix[row_i, col_i] = np.argmax(pred)
            cmatrix[row_i, col_i] = np.count_nonzero(total)
            vmatrix[row_i, col_i] = np.max(pred)

    # Color distribution is too small to predict result set empty 
    prob_matrix = np.divide(vmatrix, cmatrix, out=np.ones_like(vmatrix), where=cmatrix!=0)

    # Lower confidence set to INCONCLUSIVE
    matrix[prob_matrix < 0.5] = 3
    return matrix

from skimage.util import view_as_blocks
@measure
def get_cells_patches(cmask, cellw, cellh):
    # x = np.ones((4, 24, 36))
    # rows = 8
    # cols = 12
    # cell_size = 16
    # h = 24
    # w = 32
    # matrix.shape => (channels, rows, cols, cell_size, cell_size)
    c, _, _ = cmask.shape
    #cells = cmask.reshape(c, h//cell_size, w//cell_size, -1), 
    # (8*16, 12*16, 4)
    cells = view_as_blocks(cmask, (c, cellh, cellw)).squeeze(axis=0) #=> (8, 12, 4, 16, 16)
    return cells

if __name__ == "__main__":
    import cv2
    #import matplotlib.pyplot as plt
    from lamp_extractor import utils

    #img = cv2.imread(r"F:\aston\machine-learning\lamp\assets\12_marked.jpg")
    #img = cv2.imread(r"F:\aston\Aston ITM spol. s r.o\MultiplexDX - LAMP TESTS - Dokumenty\General\multiplex_202109-20211111\images\CAP89863420806912011.jpg")
    #img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\test\samples\CAP805904071572992837.jpg")
    #img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\test\samples\CAP1346660511792376914.jpg")
    #img = cv2.imread(r"F:\aston\datasets\lamp_karton+white_paper\50\1.jpg")
    #img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\test\samples\_20211206_FE00357953_rna.jpg")
    #img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\data\aston_test_20211203\_20211202_FE00357953_rna.jpg")
    #img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\test\samples\asttest_20211206_FE00357953_rna.jpg")
    #img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\test\samples\astest\asttest_20211206_FE00357953_rna.jpg")
    #img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\test\samples\1.jpg")
    #img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\test\samples\astest\asttest_20211206_FE00357953_sars.jpg")
    #img = cv2.imread(r"test/samples/astest/CAP458323291941563926.jpg")
    #img = cv2.imread(r"test/samples/astest/asttest_20211206_FE00357953_rna.jpg")
    img = cv2.imread(r"F:\aston\machine-learning\lamp-extractor\data\multiplex_202109-20211111\asttest_20211207_u56_6000373311_rna.jpg")
    config = utils.load_yaml(resource_filename("lamp_extractor", f"apis/rest/config.yaml"))
    matrix, CLASSES, ROWS, COLUMNS, warped_img = predict(
        img=img,
        config=config,
        visualize=False,
    )
    #plt.imshow(warped_img), plt.show()

    colors = [
        [224, 224, 224],
        [255, 0, 255],
        [255, 128, 0],
        [255, 255, 0],
    ]

    colormask = np.zeros([*matrix.shape, 3])
    colormask[matrix == 0] = colors[0]
    colormask[matrix == 1] = colors[1]
    colormask[matrix == 2] = colors[2]
    colormask[matrix == 3] = colors[3]
    plt.imshow(colormask.astype(np.uint8)), plt.show()
    print("END")