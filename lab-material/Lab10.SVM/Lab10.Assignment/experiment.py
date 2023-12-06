import cv2
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "samples/vehicles"
neg_dir = "samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/test_video.mp4"

# orientations = [3,6,9,12,15]
# colors = ["bgr", "hsv", "luv", "hls", "yuv", "ycrcb",'gray']

def experiment(color,orientation):
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """
    # TODO: You need to adjust hyperparameters
    # Extract HOG features from images in the sample directories and 
    # return results and parameters in a dict.
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,hog_features=True,color_space=color,hog_bins=orientation,spatial_features=True,hist_features=True)


    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)


    # TODO: You need to adjust hyperparameters of loadClassifier() and detectVideo()
    #       to obtain a better performance

    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector(init_size=(80,80), y_step=0.006, x_range=(0.09, 0.91), y_range=(0.55, 0.85), scale=1.2).loadClassifier(classifier_data=classifier_data)
    # detector = Detector().loadClassifier(classifier_data=classifier_data)
  
    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    # detector.detectVideo(video_capture=cap)
    detector.detectVideo(video_capture=cap, num_frames=9, threshold=120)

    # return test_acc

if __name__ == "__main__":
    # best_accuracy = 0
    # for orientation in orientations:
    #     for color in colors:

    #         # Train the model and calculate test accuracy
    #         # This is a placeholder, replace it with your actual function
    #         accuracy = experiment(color,orientation)

    #         if accuracy > best_accuracy:
    #             best_accuracy = accuracy
    #             best_params = (orientation, color)

    # print(f"Best parameters: orientations={best_params[0]}, color_space={best_params[1]}")
    # print(f"Best accuracy: {best_accuracy}")
    best_orientation = 12
    best_color = 'bgr'
    experiment(best_color,best_orientation)


