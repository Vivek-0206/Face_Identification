import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd


def find_cosine_similarity(source_representation, test_representation):
    """Find CosinesSimilarity"""
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


class FaceRecognition:
    def __init__(self, dataset_path=r'dataset'):
        """
        Info : Face_Recognition class

        :param dataset_path : str (example: 'Path_to_Data')

        :return: final person name
        """
        self.img_height = 224
        self.img_width = 224
        self.dataset_path = dataset_path

    def create_model(self):
        """
        Info : Load Model with tf.keras load_model function.
        :return : Tensorflow model
        """
        try:
            model = tf.keras.models.load_model(
                'face_identification\model\model.h5')
            print("\nModel Loaded...")

            return model
        except Exception as e:
            print('\nERROR: Tensorflow model not found.\n', e)

    def preprocess_image(self, image_path):
        """
        Info : Loads image from path and resizes it
        :return : resize image
        """
        try:
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(self.img_height, self.img_width))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.imagenet_utils.preprocess_input(img)

            return img
        except Exception as e:
            print('\nERROR: ', e)

    def load_detector(self):
        """
        Info :  Load caffe model for face detection.
        :return : detector
        """
        try:
            prototxt = r"face_identification\model\deploy.prototxt"
            caffmodel = r"face_identification\model\res10_300x300_ssd_iter_140000.caffemodel"
            detector = cv2.dnn.readNetFromCaffe(prototxt, caffmodel)

            return detector
        except Exception as e:
            print('\nERROR: ', e)

    def detect_face(self, img):
        """
        Info : Detect face from image.
        :return : detections, aspect_ratio
        """
        try:
            original_size = img.shape
            target_size = (300, 300)
            img = cv2.resize(img, target_size)  # Resize to target_size
            aspect_ratio_x = original_size[1] / target_size[1]
            aspect_ratio_y = original_size[0] / target_size[0]
            imageBlob = cv2.dnn.blobFromImage(image=img)
            detector = self.load_detector()
            detector.setInput(imageBlob)
            detections = detector.forward()

            return detections, aspect_ratio_x, aspect_ratio_y
        except Exception as e:
            print('\nERROR: ', e)

    def predict_person(self):
        """
        Info : Predict on webcam and return name of detected person if it's already known.
        :return : user_name
        """
        found = 0
        user_name = ""
        try:
            model = self.create_model()

            mypath = self.dataset_path
            all_people_faces = dict()
            if os.path.isdir(mypath) == True:
                # r=root, d=directories, f = files
                for r, d, f in os.walk(mypath):
                    for file in f:
                        if ".jpg" in file:
                            exact_path = r + "\\" + file
                            person_face = file.split("/")[-1].split(".")[0]
                            all_people_faces[person_face] = model.predict(
                                self.preprocess_image(exact_path)
                            )[0, :]

            print("\nFace representations retrieved successfully")

            cap = cv2.VideoCapture(
                0, cv2.CAP_DSHOW
            )
            print("\nStart Recognition .....")
            while True:
                ret, img = cap.read()
                base_img = img.copy()
                detections, aspect_ratio_x, aspect_ratio_y = self.detect_face(
                    img)
                detections_df = pd.DataFrame(
                    detections[0][0],
                    columns=[
                        "img_id",
                        "is_face",
                        "confidence",
                        "left",
                        "top",
                        "right",
                        "bottom",
                    ],
                )
                detections_df = detections_df[detections_df["is_face"] == 1]
                detections_df = detections_df[detections_df["confidence"] >= 0.95]
                if len(detections_df) != 0:
                    for i, instance in detections_df.iterrows():
                        left = int(instance["left"] * 300)
                        bottom = int(instance["bottom"] * 300)
                        right = int(instance["right"] * 300)
                        top = int(instance["top"] * 300)
                        # draw rectangle to main image
                        cv2.rectangle(
                            img,
                            (int(left * aspect_ratio_x),
                             int(top * aspect_ratio_y)),
                            (int(right * aspect_ratio_x),
                             int(bottom * aspect_ratio_y)),
                            (255, 0, 0),
                            2,
                        )
                        cv2.imshow("img", img)
                        detected_face = base_img[
                            int(top * aspect_ratio_y)
                            - 100: int(bottom * aspect_ratio_y) + 100,
                            int(left * aspect_ratio_x)
                            - 100: int(right * aspect_ratio_x) + 100,
                        ]
                        if len(detected_face) != 0:
                            try:
                                detected_face = cv2.resize(
                                    detected_face, (self.img_height,
                                                    self.img_width)
                                )
                                img_pixels = tf.keras.preprocessing.image.img_to_array(
                                    detected_face
                                )
                                img_pixels = np.expand_dims(img_pixels, axis=0)
                                img_pixels /= 255
                                captured_representation = model.predict(img_pixels)[
                                    0, :]
                                for person in all_people_faces:
                                    person_name = person
                                    representation = all_people_faces[person]
                                    similarity = find_cosine_similarity(
                                        representation, captured_representation
                                    )
                                    if similarity < 0.30:
                                        user_name = person_name.split("_")[1]
                                        found = 1
                                        break
                            except Exception as e:
                                print(e)
                cv2.imshow("img", img)
                if found == 1:
                    return user_name
                if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                    break

        except Exception as e:
            print("\nERROR: ", e)
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    obj = FaceRecognition()
