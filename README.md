# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/hog_car.png
[image2]: ./writeup_images/hog_non_car.png
[image3]: ./writeup_images/detection_windows.png
[image4]: ./writeup_images/advanced_windows.png
[image5]: ./writeup_images/positive_boxes.png
[image6]: ./writeup_images/heat_map.png
[image7]: ./writeup_images/thresh_img.png
[image8]: ./writeup_images/final_img.png

## Introduction

In this project, we are tasked with detecting vehicles in a video. The video consists of a car mounted with a camera driving through a highway, we have to detect vehicles from this camera feed. This report gives an overview of our approach to solving this problem. We have included code snippets or function names where deemed necessary and helpful. This report will explain how we achieved each of the project goals (list above) and, in the process, show that all the rubric requirements have been satisfied.

## Files, System Configuration, and Dependencies

### Files

- *README.md* (this file) is the project report
- *vechicle_detection_code_clean.ipynb* contains the image processing pipline and related functions
- *vechicle_detection_code_exploration.ipynb* contains the inital code created to iterate through the parameters etc.  
- *results/project_video_out.mp4* is a video of the project video with the vehicles marked

### System Configuration

- Windows 10
- Nvidia GeForce GTX 1070
- Intel i7 4.20GHz
- 32GB RAM

### Python Package Dependencies

- Python 3.6
- moviepy
- opencv
- matplotlib
- numpy
- scipy
- skimage
- sklearn
- pickle
- pandas
- os
- itertools
- traceback


## The Machine Learning

In order to detect cars we need a function that can take-in an image and output whether or not it is an image of a car. This requires 3 main components:

- Data
- Features
- Machine Learning Algorithm

The data was provided to us. The full dataset contains around 9000 car and another 9000 non-car images. Data was read from the corresponding folders using the `get_image_locs()` and `get_images()` functions. The `get_images()` function allows us to decide what *color map* we want to use. As a pre-processing step, we applied a *high-pass* filter to each of the images to enhance the edges using `sharpen_img()` function.

Based on the lecture, it was evident that using *HOG* would be important to extract the right features to train the classifier (see `get_hog_feats()`). In addition to HOG, we also decided that it would make sense to add the *color histogram* information as the color channels themselves contain a lot of important information (see `add_color_hist()`). All three color channels were used (separately) in generating the HOG features of an image. After the two sets of features (HOG and color histogram) are generated they are combined to form a single feature vector for each image. The entire data is then *scaled* and split into testing and training sets using the `combine_scale_split()` function. A *scaler* object is also returned for use later. Once, the features are generated for an image, this feature vector is run through a Support Vector Machine classifier to generate a classification to decide whether it is car or not a car.

|![][image1] |![][image2]|

However, there are many different parameters that need to be picked for the classifier. Hence, in order to chose the best set parameters we built a `grid_search()` function and tested out different parameter combinations:

```
'color_map_lst':['HLS','YCrCb','LUV'],
'bins':[32,64],
'sharpen':[True],
'orient':[9,11],
'pix_per_cell':[8],
'cell_per_block':[2],
'svm_kernel':['rbf','linear']
```
This resulted in a total of 24 tests. Here are the results obtained:

<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Color_map</th>
      <th>Bins</th>
      <th>HOG_sharpen</th>
      <th>HOG_orientation</th>
      <th>HOG_pixels_cell</th>
      <th>HOG_cells_block</th>
      <th>SVM_kernel</th>
      <th>SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>HLS</td>
      <td>32</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.992826</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HLS</td>
      <td>32</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.995696</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HLS</td>
      <td>32</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.991392</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HLS</td>
      <td>32</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.995696</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HLS</td>
      <td>64</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.994261</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HLS</td>
      <td>64</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.997131</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HLS</td>
      <td>64</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.994261</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HLS</td>
      <td>64</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.997131</td>
    </tr>
    <tr>
      <th>9</th>
      <td>YCrCb</td>
      <td>32</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.987088</td>
    </tr>
    <tr>
      <th>10</th>
      <td>YCrCb</td>
      <td>32</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.998565</td>
    </tr>
    <tr>
      <th>11</th>
      <td>YCrCb</td>
      <td>32</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.978479</td>
    </tr>
    <tr>
      <th>12</th>
      <td>YCrCb</td>
      <td>32</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.998565</td>
    </tr>
    <tr>
      <th>13</th>
      <td>YCrCb</td>
      <td>64</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.977044</td>
    </tr>
    <tr>
      <th>14</th>
      <td>YCrCb</td>
      <td>64</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.994261</td>
    </tr>
    <tr>
      <th>15</th>
      <td>YCrCb</td>
      <td>64</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.972740</td>
    </tr>
    <tr>
      <th>16</th>
      <td>YCrCb</td>
      <td>64</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.995696</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LUV</td>
      <td>32</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.979914</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LUV</td>
      <td>32</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.997131</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LUV</td>
      <td>32</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.975610</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LUV</td>
      <td>32</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.997131</td>
    </tr>
    <tr>
      <th>21</th>
      <td>LUV</td>
      <td>64</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.954089</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LUV</td>
      <td>64</td>
      <td>True</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.995696</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LUV</td>
      <td>64</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>rbf</td>
      <td>0.945481</td>
    </tr>
    <tr>
      <th>24</th>
      <td>LUV</td>
      <td>64</td>
      <td>True</td>
      <td>11</td>
      <td>8</td>
      <td>2</td>
      <td>linear</td>
      <td>0.994261</td>
    </tr>
  </tbody>
</table>
</div>

Based on these results it is evident that using a YCrCb color map, with 32 bins in the color histogram for each channel, 11 orientations, 8 pixels/cell, and 2 cells/block for HOG features and *linear* SVM classifier would provide the highest accuracy of 99.8% (in this run). We used the same model pararmeters to construct the final classifier (`classify_img()` function).

Please note that the above results were obtained by using a smaller dataset, this was done in order to save time. Once the final parameters are chosen, the final model that we used is constructed using the full image dataset to get a more robust model.

## Image Frame Processing

Now that we have a classifier, we need a way to traverse through each of the image frames of the video to find the vehicles them. This is accomplished through the following process (entire process elaborated further below):

- slide windows of different sizes through the image and extract the regions encircled by each window
- resize each region to 64x64 since that is the image size the classifier was trained on
- classify the resized region
- if the classifier states that the region is a vehicle save the bounds of the window, if not, discard

This process has been implemented in the `find_good_windows()` function. Since, vehicles can be of varying sizes in the frame -- if the vehicle is close to the camera it is larger and if it farther away it is smaller -- we cannot stick to a fixed window size. Hence, in this function we cover different window sizes. Traversing through an image using multiple window sizes with overlap between each window is a highly time consuming process. Hence, we need to reduce the search region to only to an area where we know could be vehicles.

At first, we restricted the search area to only include the bottom half of the image pushed slightly to the right. We chose 3 window sizes (32x32, 64x64, 128x128) and used an overlap of 0.5. This resulted in the following image (with the windows):

![][image3]

As we can see it is overcrowded with windows and makes the whole process very slow and tedious. Hence, the function was modified to come with a smarter search method. Keeping in mind that vehicles appear larger as we go down the image, we re-designed the function to apply small windows to the top part, slightly bigger windows to the middle part, and the biggest windows to the lower part. The biggest window is the least expensive one to run, hence, it is also applied to the top parts. In addition, this function pushes the starting points of the windows further to the right and ends the windows before the end of the image (*see image*).

![][image4]

This approach is much faster and is able to avoid many of the *false positive* detections that previous approach was making. We also modified the classifier output to give us the probability that an inputted image is a car rather than a boolean stating whether it is a car or not. By increasing the probability threshold to over 0.5, we were again able to reduce some more of the false positives.

As seen below, there are no false positives (`draw_boxes() function`):
![][image5]

Once we have the bounds of the windows with the positive vehicle detections a *heat map* is generated (see below). A hotter area indicates more windows found a vehicle in that area (`gen_heat_map() function`).

![][image6]

We apply a minimum threshold to the heatmap to again filter out any false positive detections (`apply_thresh()` function). This also binarizes the image.

![][image7]

Using the scipy's `label` function, we find the distinct regions in this image and generate the bounding boxes of the vehicles in the original video frame using the `get_bounding_boxes()` function. The bounding boxes are generated as the min and max values of a region. This function also draws the boxes on the original image frame to give us the final output image.

![][image8]

## Pipeline

All these functions are called using a pipeline function which takes in the input frame from the video and outputs an image with rectangles drawn over the vehicles:

```
save_errors = [] # save errors to look later

def vehicle_detection_pipeline(img):

    try:
        img_copy = np.copy(img)
        # sliding windows on image
        good_windows = find_good_windows(img_copy, viz=False)
        # generate a heatmap
        heat = gen_heat_map(img_copy, good_windows, viz=False)
        # apply a threshold to remove false positives
        thresh_img = apply_thresh(heat, thresh=1, viz=False)
        # generate final image with bouding boxes around vehicles
        final_img = get_bounding_boxes(img, thresh_img, viz=False)

    except Exception as e:
        print(e)
        save_errors.append(e)

    return final_img
```

We used the moviepy package to apply the `vehicle_detection_pipeline()` function to the video to generate an output video with the vehicles detected.

## Discussion

This was a challenging project for a few reasons:

- The success rate of the machine learning algorithm in finding vehicles in the video was not directly related with the accuracy it was getting on the training images, which meant a lot of trail and error as there was no quantitative way of predicting its performance.

- Deciding the window sizes, overlap, and bounds took was a time taking processing as there were so many different possibilities. And to further complicate things, small changes to any of the pipeline functions resulted in very different results.

- Often times, the classifier would detect many non-car images as car images resulting in too many false positives. Dealing with false positives with out causing too many false negatives was tricky.

- Each run took a very long time since we needed to preprocess, extract features, scale, and then run classifier for each window in each image and there are about 1200 images in the project video.

- One things to note is that the windows chosen are not very robust, they will likely not work when applied to video shot in a different angle on a different road. More robust windows, would require traversing a larger part of the image, which would again slow down the image processing significantly and result in more false positives.

Based on my reading online, it seems that using a Convolutional Neural Network (the *YOLO* network in particular) instead of an SVM classifier would give a higher accuracy and is much faster. Further we can use a GPU with CNNs to speed up the process. In addition, the CNN would automatically decide on the best image features to use, which would save a lot of training time (and uncertainty). I plan to try this approach out as a future project.
