import numpy as np
import cv2
import glob

class LaneDetector():
    def __init__(self):
        src = np.float32([[207, 719], [597, 450], [686, 450], [1107, 719]])
        dst = np.float32([[300, 719], [300, 0], [1279 - 300, 0], [1279 - 300, 719]])

        # Transformation matrix
        self.M = cv2.getPerspectiveTransform(src, dst)

        # Inverse transformation
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def calibrate_camera(self):
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('./camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)

            shape = img.shape[1::-1]

            if shape[0] != 1280 or shape[1] != 720:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)


        cv2.destroyAllWindows()

        # Calculate claibraion parameters
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

    def undistort(self, image):
        return cv2.undistort(np.copy(image), self.mtx, self.dist, None, self.mtx)

    def thresholded_binary(self, gray_image, threshold):
        binary = np.zeros_like(gray_image)
        binary[(gray_image >= threshold[0]) & (gray_image <= threshold[1])] = 1
        return binary

    def gradient_xy(self, gray_image, sobel_kernel=5, dir_threshold=(0, 1.2)):
        ## Apply Sobel operator
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in y

        # Absolute values of gradients in X and Y directions
        absx = np.absolute(sobelx)
        absy = np.absolute(sobely)

        # Project them on diagonal line
        proj = np.sqrt(2) / 2.0;
        gradxy = absx * proj + absy * proj

        # Calculate gradient direction
        dir = np.arctan2(absy, absx)

        # Filter direction threshold
        gradxy[(dir < dir_threshold[0]) | (dir > dir_threshold[1])] = 0

        # Scale it to range 0...255
        scaled = np.uint8(255 * gradxy / np.max(gradxy))

        return scaled

    def thresholded_binary(self, gray_image, threshold):
        binary = np.zeros_like(gray_image)
        binary[(gray_image >= threshold[0]) & (gray_image <= threshold[1])] = 1
        return binary

    def gradient_xy(self, gray_image, sobel_kernel=5, dir_threshold=(0, 1.2)):
        ## Apply Sobel operator
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in y

        # Absolute values of gradients in X and Y directions
        absx = np.absolute(sobelx)
        absy = np.absolute(sobely)

        # Project them on diagonal line
        proj = np.sqrt(2) / 2.0;
        gradxy = absx * proj + absy * proj

        # Calculate gradient direction
        dir = np.arctan2(absy, absx)

        # Filter direction threshold
        gradxy[(dir < dir_threshold[0]) | (dir > dir_threshold[1])] = 0

        # Scale it to range 0...255
        scaled = np.uint8(255 * gradxy / np.max(gradxy))

        return scaled

    def combine_and(self, mask1, mask2):
        result = np.zeros_like(mask1, dtype=np.uint8)
        result[(mask1 > 0) & (mask2 > 0)] = 1
        return result

    def combine_or(self, mask1, mask2):
        result = np.zeros_like(mask1, dtype=np.uint8)
        result[(mask1 > 0) | (mask2 > 0)] = 1
        return result

    def process_image(self, image):

        # Distortion correction
        undistorted = self.undistort(image)

        # Split channels
        red = undistorted[:, :, 2]

        # HLS
        hls = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Step 1
        red_thresholded = self.thresholded_binary(red, threshold=(128, 255))

        # Step 2
        s_channel = cv2.equalizeHist(s_channel)

        # Step 3
        s_thresholded = self.thresholded_binary(s_channel, threshold=(192, 255))

        # Step 4
        red_and_s = self.combine_and(red_thresholded, s_thresholded)

        # Steps 5,6
        l_gradient = self.gradient_xy(l_channel, sobel_kernel=9, dir_threshold=(0, 1.1))

        # Step 7
        l_gradient_thresholded = self.thresholded_binary(l_gradient, threshold=(60, 255))

        # Step 8
        result = self.combine_or(red_and_s, l_gradient_thresholded)

        return result

    def warped(self, image):
        size = image.shape[1::-1]
        result = cv2.warpPerspective(image, self.M, size)
        return result

    def warped_binary(self, image):
        size = image.shape[1::-1]
        binary = self.process_image(image)

        result = cv2.convertScaleAbs(cv2.warpPerspective(binary, self.M, size))
        return result

    def find_lines_with_histogram(self, image):

        binary_warped = self.warped_binary(image)

        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        val_l = histogram[leftx_base]
        val_r = histogram[rightx_base]

        quality_left = min(val_l / 50, 1.0)
        quality_right = min(val_r / 50, 1.0)

        # print(val_l, val_r)

        # Choose the number of sliding windows
        nwindows = 9

        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return (left_fit, right_fit, quality_left, quality_right)

    def find_lines_incremental(self, image, base_poly_left, base_poly_right):

        binary_warped = self.warped_binary(image)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        left_lane_points = base_poly_left[0] * (nonzeroy ** 2) + base_poly_left[1] * nonzeroy + base_poly_left[2]
        left_lane_inds = np.absolute(nonzerox - left_lane_points) < margin

        right_lane_points = base_poly_right[0] * (nonzeroy ** 2) + base_poly_right[1] * nonzeroy + base_poly_right[2]
        right_lane_inds = np.absolute(nonzerox - right_lane_points) < margin

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # If our polynomial were calculated from 1000 or more points, consider the quality be 100%
        quality_left = min(1, len(lefty) / 1000)
        quality_right = min(1, len(righty) / 1000)

        return left_fit, right_fit, quality_left, quality_right

    class FrameInfo():
        def __init__(self, frame_height, frame_width, left_poly, right_poly, quality_left, quality_right):
            self.frame_height = frame_height
            self.frame_width = frame_width
            self.left_poly = left_poly
            self.right_poly = right_poly
            self.quality_left = quality_left
            self.quality_right = quality_right
            self.radius = 0
            self.position = 0

        def curvature(self):
            ym_per_pix = 50 / 720  # meters per pixel in y dimension
            xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

            left_fit = self.left_poly
            right_fit = self.right_poly

            y_eval = self.frame_height
            ploty = np.linspace(0, y_eval - 1, num=y_eval)
            leftx = left_fit[0] * ploty ** 2 + ploty * left_fit[1] + left_fit[2]
            rightx = right_fit[0] * ploty ** 2 + ploty * right_fit[1] + right_fit[2]

            # Calculate polynomial coefficents in real world coordinates
            left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

            count = 0

            if self.quality_left > 0.5:
                # left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
                left_curverad = ((1 + (
                2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                    2 * left_fit_cr[0])
                count += 1
            else:
                left_curverad = 0

            if self.quality_right > 0.5:
                # right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
                right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                    2 * right_fit_cr[0])
                count += 1
            else:
                right_curverad = 0

            self.radius = (left_curverad + right_curverad) / count
            return self.radius

        def pos(self):
            # Here we assume the lane be 3.7 meter wide
            px = self.frame_height - 1

            lx = px * px * self.left_poly[0] + px * self.left_poly[1] + self.left_poly[2]
            rx = px * px * self.right_poly[0] + px * self.right_poly[1] + self.right_poly[2]
            scale = (rx - lx) / 3.7
            lane_center = (lx + rx) / 2.0
            # print(lx, rx, lane_center)

            self.position = (self.frame_width / 2 - lane_center) / scale
            return self.position

    def analyze_frame(self, image):
        lp, rp, ql, qr = self.find_lines_with_histogram(image)
        frame_info = self.FrameInfo(image.shape[0], image.shape[1], lp, rp, ql, qr)
        return frame_info

    def analyze_frame_inc(self, image, base_frame_info):
        lp, rp, ql, qr = self.find_lines_incremental(image, base_frame_info.left_poly, base_frame_info.right_poly)
        frame_info = self.FrameInfo(image.shape[0], image.shape[1], lp, rp, ql, qr)
        return frame_info

    def draw_measurement(self, image, frame_info):
        color_warp = np.zeros_like(image).astype(np.uint8)
        height = image.shape[0]

        # Prepare values
        ploty = np.linspace(0, height - 1, height, dtype=np.int)
        left_fitx = frame_info.left_poly[0] * ploty ** 2 + frame_info.left_poly[1] * ploty + frame_info.left_poly[2]
        right_fitx = frame_info.right_poly[0] * ploty ** 2 + frame_info.right_poly[1] * ploty + frame_info.right_poly[2]

        # Calculate points
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, self.Minv, (image.shape[1], image.shape[0]))

        # Combine the result with the original image
        undist = self.undistort(image)

        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result

    def map_lane(self, image, frame_info):
        result = np.copy(self.draw_measurement(image, frame_info))

        crad = "{:.0f} (m)".format(frame_info.curvature())
        if frame_info.curvature() > 10000:
            # Consider curves with radium > 10 km be straight roads
            crad = "straight road"

        ctext = "Radius of curvature: " + crad

        if frame_info.pos() < 0:
            ptext = "Position is {:.2f}m left of center".format(-frame_info.pos())
        else:
            ptext = "Position is {:.2f}m right of center".format(frame_info.pos())

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, ctext, (10, 80), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(result, ptext, (10, 160), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
        return result

    def init_video(self):
        self.frame_info = None
        self.smooth_frame_info = None
        self.quality = 1
        self.idx = 0

    def process_video_frame(self, image):

        alpha = 0.1

        current_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if (self.smooth_frame_info is not None) and (self.quality > 0.9):
            # Use incremental search
            frame_info = self.analyze_frame_inc(current_image, self.smooth_frame_info)
        else:
            # Run full frame analyzis
            frame_info = self.analyze_frame(current_image)

        lx = frame_info.left_poly[2]
        rx = frame_info.right_poly[2]
        middle = image.shape[1] / 2

        self.quality = (frame_info.quality_left + frame_info.quality_right) / 2

        weight = 1

        # Detect outlier
        if abs(lx - middle) > 1000 or abs(rx - middle) > 1000:
            weight = 0

        if self.smooth_frame_info == None:
            smooth_frame_info = frame_info

        # Add some inertia with low pass filter
        smooth_frame_info.left_poly = smooth_frame_info.left_poly * (
        1 - alpha * weight) + frame_info.left_poly * alpha * weight
        smooth_frame_info.right_poly = smooth_frame_info.right_poly * (
        1 - alpha * weight) + frame_info.right_poly * alpha * weight

        lane = self.map_lane(current_image, smooth_frame_info)
        self.idx += 1

        result = cv2.cvtColor(lane, cv2.COLOR_BGR2RGB)
        return result