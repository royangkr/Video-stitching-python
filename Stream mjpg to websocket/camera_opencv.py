import cv2
import numpy as np
import imutils
from imutils.video import FileVideoStream
import tqdm
import os
import datetime
from base_camera import BaseCamera
import threading


class Camera(BaseCamera):
    
    display = False
    video_out_width=2560
    saved_homo_matrix = None
    bwidth=None
    bheight=None
    by=None
    bx=None
    video_source = "Library_output.mp4"
    left_video_in_path='Library_L_cut.mp4'
    right_video_in_path='Library_R_cut.mp4'

    @staticmethod
    def stitch(images, ratio=0.75, reproj_thresh=4.0):
        # Unpack the images
        (image_b, image_a) = images

        # If the saved homography matrix is None, then we need to apply keypoint matching to construct it
        if Camera.saved_homo_matrix is None:
            # Detect keypoints and extract
            (keypoints_a, features_a) = Camera.detect_and_extract(image_a)
            (keypoints_b, features_b) = Camera.detect_and_extract(image_b)

            # Match features between the two images
            matched_keypoints = Camera.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

            # If the match is None, then there aren't enough matched keypoints to create a panorama
            if matched_keypoints is None:
                return None

            # get bounding rectangle of homography of matched_keypoints[1]
            height, width = image_a.shape[:2]
            corners = np.array([
              [0, 0],
              [0, height - 1],
              [width - 1, height - 1],
              [width - 1, 0]
            ])
            corners = cv2.perspectiveTransform(np.float32([corners]), matched_keypoints[1])[0]

            # Find the bounding rectangle
            bx, by, bwidth, bheight = cv2.boundingRect(corners)

            # Compute the translation homography that will move (bx, by) to (0, 0)
            th = np.array([
              [ 1, 0, -bx ],
              [ 0, 1, -by ],
              [ 0, 0,   1 ]
            ])

            # Combine the homographies
            Camera.saved_homo_matrix = th.dot(matched_keypoints[1])
            Camera.bwidth=bwidth+abs(bx)
            Camera.bheight=bheight+abs(by)
            Camera.by=by
            
        # Apply a perspective transform to stitch the images together using the saved homography matrix
        result=cv2.warpPerspective(image_a, Camera.saved_homo_matrix, (Camera.bwidth, Camera.bheight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        result[-Camera.by:image_b.shape[0]-Camera.by, 0:image_b.shape[1]] = image_b
        return result

    @staticmethod
    def warpOne(image_a):
        result=cv2.warpPerspective(image_a, Camera.saved_homo_matrix, (Camera.bwidth, Camera.bheight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return result
    
    @staticmethod
    def detect_and_extract(image):
        # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
        descriptor = cv2.xfeatures2d.SIFT_create()
        (keypoints, features) = descriptor.detectAndCompute(image, None)

        # Convert the keypoints from KeyPoint objects to numpy arrays
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        # Return a tuple of keypoints and features
        return (keypoints, features)

    @staticmethod
    def match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh):
        # Compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, k=2)
        matches = []

        for raw_match in raw_matches:
            # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(raw_match) == 2 and raw_match[0].distance < raw_match[1].distance * ratio:
                matches.append((raw_match[0].trainIdx, raw_match[0].queryIdx))

        # Computing a homography requires at least 4 matches
        if len(matches) > 4:
            # Construct the two sets of points
            points_a = np.float32([keypoints_a[i] for (_, i) in matches])
            points_b = np.float32([keypoints_b[i] for (i, _) in matches])

            # Compute the homography between the two sets of points
            (homography_matrix, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_thresh)

            # Return the matches, homography matrix and status of each matched point
            return (matches, homography_matrix, status)

        # No homography could be computed
        return None

    @staticmethod
    def draw_matches(image_a, image_b, keypoints_a, keypoints_b, matches, status):
        # Initialize the output visualization image
        (height_a, width_a) = image_a.shape[:2]
        (height_b, width_b) = image_b.shape[:2]
        visualisation = np.zeros((max(height_a, height_b), width_a + width_b, 3), dtype="uint8")
        visualisation[0:height_a, 0:width_a] = image_a
        visualisation[0:height_b, width_a:] = image_b

        for ((train_index, query_index), s) in zip(matches, status):
            # Only process the match if the keypoint was successfully matched
            if s == 1:
                # Draw the match
                point_a = (int(keypoints_a[query_index][0]), int(keypoints_a[query_index][1]))
                point_b = (int(keypoints_b[train_index][0]) + width_a, int(keypoints_b[train_index][1]))
                cv2.line(visualisation, point_a, point_b, (0, 255, 0), 1)

        # return the visualization
        return visualisation

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        '''camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()'''
        if Camera.saved_homo_matrix is None:
            left = cv2.VideoCapture(Camera.left_video_in_path)
            right = cv2.VideoCapture(Camera.right_video_in_path)
            matched_keypoints=None
            ratio=0.75
            reproj_thresh=4.0
            while matched_keypoints is None:
                ok,image_a=left.read()
                _,image_b=right.read()
                # Detect keypoints and extract
                (keypoints_a, features_a) = Camera.detect_and_extract(image_a)
                (keypoints_b, features_b) = Camera.detect_and_extract(image_b)

                # Match features between the two images
                matched_keypoints = Camera.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)
            # get bounding rectangle of homography of matched_keypoints[1]
            height, width = image_a.shape[:2]
            corners = np.array([
              [0, 0],
              [0, height - 1],
              [width - 1, height - 1],
              [width - 1, 0]
            ])
            corners = cv2.perspectiveTransform(np.float32([corners]), matched_keypoints[1])[0]

            # Find the bounding rectangle
            bx, by, bwidth, bheight = cv2.boundingRect(corners)

            # Compute the translation homography that will move (bx, by) to (0, 0)
            # no need to translate when bx and by are positive, aka already can be seen
            if by>0:
                translate_y=0
            else:
                translate_y=by
            if bx>0:
                translate_x=0
            else:
                translate_x=bx
            # translate if either bx and by are negative
            th = np.array([
              [ 1, 0, -translate_x ],
              [ 0, 1, -translate_y ],
              [ 0, 0,   1 ]
            ])

            # Combine the homographies
            Camera.saved_homo_matrix = th.dot(matched_keypoints[1])
            if bx>=0:
                Camera.bwidth=max(bwidth+bx,width)
            else:
                Camera.bwidth=max(width+abs(bx),bwidth)
            if by>=0:
                Camera.bheight=max(bheight+by,height)
            else:
                Camera.bheight=max(height+abs(by),bheight)
            Camera.by=-translate_y
            Camera.bx=-translate_x
            print("dimensions"+str(Camera.bwidth)+" "+str(Camera.bheight))
            left.release()
            right.release()
        left_video=FileVideoStream(Camera.left_video_in_path,transform=Camera.warpOne).start()
        right_video=FileVideoStream(Camera.right_video_in_path).start()
        print('[INFO]: {} and {} loaded'.format(Camera.left_video_in_path.split('/')[-1],
                                                Camera.right_video_in_path.split('/')[-1]))
        startStitch=datetime.datetime.now()
        print('[INFO]: Video stitching starting.... ')
        print('[INFO]: Started at '+str(startStitch))

        # Get information about the videos
        n_frames = min(int(left_video.stream.get(cv2.CAP_PROP_FRAME_COUNT)),
                       int(right_video.stream.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = int(left_video.stream.get(cv2.CAP_PROP_FPS))
        first=1
        x=None
        for _ in tqdm.tqdm(np.arange(n_frames)):
            # left video frame is transformed in stream thread
            stitched_frame = left_video.read()
            image_b = right_video.read()

            # right video frame added on top of left
            stitched_frame[Camera.by:image_b.shape[0]+Camera.by, Camera.bx:image_b.shape[1]+Camera.bx] = image_b
            yield stitched_frame
        endStitch=datetime.datetime.now()
        print('[INFO]: Video stitching finished at '+str(endStitch))
        print('[INFO]: Stitch time:\t'+str(((endStitch-startStitch).total_seconds()-(endStitch-startStitch).total_seconds()%60)/60)+' mins '+str((endStitch-startStitch).total_seconds()%60)+' seconds')
