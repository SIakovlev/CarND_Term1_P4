**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./report_images/calibration.jpg "Undistorted"
[image2]: ./report_images/undistortion.jpg "Road Transformed"
[image3]: ./report_images/thresh_bin_img.jpg "Binary Example"
[image4]: ./report_images/region1.jpg "Perspective transform straight"
[image5]: ./report_images/region2.jpg "Perspective transform curved"
[image6]: ./report_images/pipeline.jpg "Pipeline"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first section called "Camera calibration" of the IPython notebook `lane_finding.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function from OpenCV library.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of image distortion correction for one of the test images:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The main routines are in the "Image processing pipeline" sectin of `lane_finding.ipynb` file.

Image thresholding is the logical disjunction (pythonic `|`) of the following:

* Yellow color thresholding (`yellow_select(...)` function in the code):
    * Convert image to HLS color space.
    * Determine the range of H, L, and S components for yellow color: **H: 15 - 55, L: 120 - 200, S: 120 - 255.** 
    * Threshold image based on these values 
* White color thresholding (`white_select(...)` function in the code):
    * Convert image to HLS color space.
    * Determine the range of H, L, and S components for white color: **H: 0 - 360, L: 210 - 255, S: 0 - 255.** 
    * Threshold image based on these values
* Sobel gradient thresholding for x coordinate (`sobel_abs(...)` function in the code).

Here's an example of the sequential application of methods above:

![alt text][image3]

The original undistorted picture is on the left, next we see the result of application of `yellow_select(...)` only, then we add white color thresholding and picture on the right is the result of all thresholding methods combined together (via logical "or").

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the function called `perspective(img, mode='f', scr=src, dst=dst):`. This function takes `img` as an input and, based on `mode`, returns inverse or forward perspective transform of this image. The source(`src`) and destination(`dst`) points were chosen in the following manner:

`cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)`

```python
src = np.float32([[0, img_size[1]], 
                  [575, 450],
                  [705, 450],
                  [img_size[0], img_size[1]]])
dst = np.float32([[100, img_size[1]],
                  [100, 0],
                  [img_size[0]-100, 0], 
                  [img_size[0]-100, img_size[1]]])
```

This resulted in the following source and destination points:

| Source (x, y)       | Destination (x, y)  | 
|:-------------:|:-------------:| 
| (0, 720)      | (100, 720)    | 
| (575, 450)    | (100, 0)      |
| (705, 450)    | (1180, 0)     |
| (1280, 720)   | (1180, 720)   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here is an example for straight lines:

![alt text][image4]

This example shows the perspective transform for curved lines:

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?



![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
