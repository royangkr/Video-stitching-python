'''
Written by Roy Ang
Referenced from https://github.com/Toemazz/VideoStitcher with
Optimisation in homography tranformation, output file codec, and computational time
'''
import cv2
import numpy as np
import imutils
from imutils.video import FileVideoStream
import tqdm
import os
import datetime

class VideoStitcher:
    saved_homo_matrix = None
    bwidth=None
    bheight=None
    left_video_in_path = None
    right_video_in_path = None
    video_out_path = None
    video_out_width=2560
    display=False

    saved_homo_matrix = None
    bwidth=None
    bheight=None
    by=None
    bx=None

    def stitch(images, ratio=0.75, reproj_thresh=4.0):
        # Unpack the images
        (image_b, image_a) = images

        # If the saved homography matrix is None, then we need to apply keypoint matching to construct it
        if VideoStitcher.saved_homo_matrix is None:
            # Detect keypoints and extract
            (keypoints_a, features_a) = VideoStitcher.detect_and_extract(image_a)
            (keypoints_b, features_b) = VideoStitcher.detect_and_extract(image_b)

            # Match features between the two images
            matched_keypoints = VideoStitcher.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

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
              [ 1, 0, 0 ],
              [ 0, 1, -by ],
              [ 0, 0,   1 ]
            ])

            # Combine the homographies
            VideoStitcher.saved_homo_matrix = th.dot(matched_keypoints[1])
            VideoStitcher.bwidth=bwidth+bx
            VideoStitcher.bheight=bheight
            VideoStitcher.by=by
            
        # Apply a perspective transform to stitch the images together using the saved homography matrix
        result=cv2.warpPerspective(image_a, VideoStitcher.saved_homo_matrix, (VideoStitcher.bwidth, VideoStitcher.bheight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        result[-VideoStitcher.by:image_b.shape[0]-VideoStitcher.by, 0:image_b.shape[1]] = image_b
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
    def warpOne(image_a):
        result=cv2.warpPerspective(image_a, VideoStitcher.saved_homo_matrix, (VideoStitcher.bwidth, VideoStitcher.bheight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return result
    @staticmethod
    def run():
        # Set up video capture
        if VideoStitcher.saved_homo_matrix is None:
            left = cv2.VideoCapture(VideoStitcher.left_video_in_path)
            right = cv2.VideoCapture(VideoStitcher.right_video_in_path)
            matched_keypoints=None
            ratio=0.75
            reproj_thresh=4.0
            while matched_keypoints is None:
                ok,image_a=left.read()
                _,image_b=right.read()
                # Detect keypoints and extract
                (keypoints_a, features_a) = VideoStitcher.detect_and_extract(image_a)
                (keypoints_b, features_b) = VideoStitcher.detect_and_extract(image_b)

                # Match features between the two images
                matched_keypoints = VideoStitcher.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)
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
            VideoStitcher.saved_homo_matrix = th.dot(matched_keypoints[1])
            if bx>=0:
                VideoStitcher.bwidth=max(bwidth+bx,width)
            else:
                VideoStitcher.bwidth=max(width+abs(bx),bwidth)
            if by>=0:
                VideoStitcher.bheight=max(bheight+by,height)
            else:
                VideoStitcher.bheight=max(height+abs(by),bheight)
            VideoStitcher.by=-translate_y
            VideoStitcher.bx=-translate_x
            left.release()
            right.release()
        left_video=FileVideoStream(VideoStitcher.left_video_in_path,transform=VideoStitcher.warpOne).start()
        right_video=FileVideoStream(VideoStitcher.right_video_in_path).start()
        print('[INFO]: {} and {} loaded'.format(VideoStitcher.left_video_in_path.split('/')[-1],
                                                VideoStitcher.right_video_in_path.split('/')[-1]))
        startStitch=datetime.datetime.now()
        print('[INFO]: Video stitching starting.... ')
        print('[INFO]: Started at '+str(startStitch))

        # Get information about the videos
        n_frames = min(int(left_video.stream.get(cv2.CAP_PROP_FRAME_COUNT)),
                       int(right_video.stream.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = int(left_video.stream.get(cv2.CAP_PROP_FPS))
        first=1
        for _ in tqdm.tqdm(np.arange(n_frames)):
            # Grab the frames from their respective video streams
            stitched_frame = left_video.read()
            image_b = right_video.read()

            # right video frame added on top of left
            
            stitched_frame[VideoStitcher.by:image_b.shape[0]+VideoStitcher.by, VideoStitcher.bx:image_b.shape[1]+VideoStitcher.bx] = image_b
        
            #stitched_frame = Camera.stitch([left, right])
            
            # No homography could not be computed
            if stitched_frame is None:
                print("[INFO]: Homography could not be computed!")
                break

            # if first iteration, create videowriter object with frame size
            if first:
                height , width , layers =  stitched_frame.shape
                video = cv2.VideoWriter(VideoStitcher.video_out_path,0x00000020, fps=fps,frameSize=(width,height))
                first=0
                cv2.namedWindow("First frame", cv2.WINDOW_NORMAL)
                cv2.imshow("First frame", stitched_frame)
                cv2.waitKey(1)
            video.write(stitched_frame)

            if VideoStitcher.display:
                # Show the output images
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.imshow("Result", stitched_frame)
                cv2.waitKey(1)

            # If the 'q' key was pressed, break from the loop
            

        cv2.destroyAllWindows()
        video.release()
        endStitch=datetime.datetime.now()
        print('[INFO]: Video stitching finished at '+str(endStitch))
        print('[INFO]: Stitch time:\t'+str(((endStitch-startStitch).total_seconds()-(endStitch-startStitch).total_seconds()%60)/60)+' mins '+str((endStitch-startStitch).total_seconds()%60)+' seconds')
        


# replace with any video. tested file formats include mp4, avi
VideoStitcher.left_video_in_path='Library_L_cut.mp4'
VideoStitcher.right_video_in_path='Library_R_cut.mp4'
VideoStitcher.video_out_path='Library_output_cut.mp4'
VideoStitcher.run()
