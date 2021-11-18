import numpy as np
import pandas as pd

root = '/home/labuser/LogoDet/LogoDetection_DSBAProject/inference/'

# Upload test images annotations
real_box_by_image = pd.read_csv(root + 'data_iou/_annotations.csv')

real_box_by_image = real_box_by_image[real_box_by_image['filename'].notna()]

real_box_by_image = real_box_by_image[(real_box_by_image["area"] > 1000)]

print(f'The shape of the test set is {real_box_by_image.shape}')

def compute_area_box(row):
    # Compute area box given coordinates
    return (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])

# Let's create a new column with the area of each box in the image annotation
# This value will be usefull when matching real boxes with predicted ones
real_box_by_image['area'] = real_box_by_image.apply(lambda x: compute_area_box(x), axis=1)

# Upload predictions of test images
filename_prediction = 'prediction_bounding_boxes_60.csv'

predicted_box_by_image = pd.read_csv(root + 'data_iou/' + filename_prediction)
print(f'The shape of the prediction set is {predicted_box_by_image.shape}')


### Run this part to change the name of the files if still not changed from Roboflow
"""
def clean_filename(filename):
    # Clean the filename from Roboflow hash
    return filename.split('.')[0]

real_box_by_image['filename'] = real_box_by_image['filename'].apply(lambda x: clean_filename(x))
real_box_by_image.to_csv(root + 'data_iou/_annotations.csv', index=False)

predicted_box_by_image['filename'] = predicted_box_by_image['filename'].apply(lambda x: clean_filename(x))

predicted_box_by_image.to_csv(root + 'data_iou/' + filename_prediction, index=False)"""


def extract_coordinates(row, width=1, height=1):
    return [row['ymin']*height, row['xmin']*width, row['ymax']*height, row['xmax']*width]

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# Create final dataframe from the test annotation
df_iou = real_box_by_image[['filename', 'class', 'area']].copy()

# Let's keep in the dataframe only the images belonging both to test set and the predictions
intersection_imgs = list(set(predicted_box_by_image['filename']) & set(real_box_by_image['filename']))
df_iou = df_iou[df_iou['filename'].isin(intersection_imgs)]

# For the complete IoU the default value is 0 because if there is no prediction the IoU is 0
df_iou['iou_complete'] = 0 

# For the IoU only on the predicted boxes, the default value is NaN so it does not contribute to the average for the non-predicted boxes
df_iou['iou_only_predicted'] = np.nan 

print(f'The shape of the IoU df is {df_iou.shape}')

for image in set(df_iou['filename']):
    
    # What are the different types of logos present in the image?
    logo_type_list = list(set(real_box_by_image[real_box_by_image['filename'] == image]['class']))

    for logo_type in logo_type_list:

        # Slice the dataframes to keep only rows related to
        # > current image
        # > current logo type
        bb_real = real_box_by_image[(real_box_by_image['filename'] == image) & (real_box_by_image['class'] == logo_type)].copy().reset_index(drop=True)
        bb_predicted = predicted_box_by_image[(predicted_box_by_image['filename'] == image) & (predicted_box_by_image['class'] == logo_type)].copy().reset_index(drop=True)

        # Let's sort the real images by area of the box
        # Big images are matched first because IoU is more indicative
        bb_real.sort_values(by='area', ascending=False, inplace=True)

        # Let's iterate over the rows of the real bounding boxes
        for _, row_real in bb_real.iterrows():

            current_real_box_area = row_real['area']

            # For every real bounding box, start with the ones with biggest area
            # Compute the IoU over all the predicted bounding boxes and pick the one with the highest value
            # Remove the predicted box just matched and move to the next logo
            # If there are no more predicted boxes, exit the loop
            iou_list = []

            coords_real = extract_coordinates(row_real)

            for _, row_predicted in bb_predicted.iterrows():
                
                coords_predicted = extract_coordinates(row_predicted, width=512, height=512)
                iou_list.append(intersection_over_union(coords_real, coords_predicted))

            if len(iou_list) > 0:
                # Find best value in the IoU list
                max_iou = max(iou_list)

                # Insert the value in the IoU df
                # IMPORTANT: the assumption here for the correct slices is that there are not 2 boxes with same area for the same image
                # It should be very unlikely to happen, area good proxy as primary key within image to slice the IoU df
                df_iou.loc[(df_iou['filename'] == image) & (df_iou['area'] == current_real_box_area), 'iou_complete'] = max_iou
                df_iou.loc[(df_iou['filename'] == image) & (df_iou['area'] == current_real_box_area), 'iou_only_predicted'] = max_iou

                # Find index of the best value (= index of the predicted box to remove in bb_predicted)
                max_index = np.argmax(iou_list)

                # Remove row with matched box from predicted boxes
                bb_predicted.drop(max_index, inplace=True)
                bb_predicted.reset_index(drop=True, inplace=True)


df_iou.to_csv(root + 'data_iou/iou_results_' + filename_prediction, index=False)

print("Result IOU complete")
print(df_iou.groupby('class')['iou_complete'].mean().reset_index())
print()
print("Result IOU only predicted")
print(df_iou.groupby('class')['iou_only_predicted'].mean().reset_index())


