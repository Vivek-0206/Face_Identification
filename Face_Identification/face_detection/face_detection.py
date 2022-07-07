import cv2
import os
import pandas as pd


class FaceDetection:
    def __init__(self, dataset_path=r'',
                 user_name='testuser',
                 no_of_samples=5):
        """
        Info : Face_detection class

        :param dataset_path: str (example: 'path_of_dataset')
        :param user_name: str (example: 'name_of_user')
        :param no_of_samples: int (example: 5)


        :return: None
        """
        self.dataset_path = dataset_path
        self.user_name = user_name
        self.no_of_samples = no_of_samples
        self.width = 300
        self.height = 300

    def load_model(self):
        """
        Info : Load OpenCV CAFFE Model.

        :return: model
        """
        try:
            prototxt = r"face_identification\model\deploy.prototxt"
            model = r"face_identification\model\res10_300x300_ssd_iter_140000.caffemodel"
            detector = cv2.dnn.readNetFromCaffe(prototxt, model)
            return detector
        except Exception as e:
            print('\nERROR: CaffeModel not found.\n', e)

    def detect_face(self, image):
        """
        Info : Detect face from image.
        """
        try:
            detector = self.load_model()
            original_size = image.shape
            target_size = (self.height, self.width)
            image = cv2.resize(image, target_size)
            aspect_ratio_x = original_size[1] / target_size[1]
            aspect_ratio_y = original_size[0] / target_size[0]
            imageBlob = cv2.dnn.blobFromImage(image=image)
            detector.setInput(imageBlob)
            detections = detector.forward()

            return detections, aspect_ratio_x, aspect_ratio_y
        except Exception as e:
            print('\nERROR: ', e)

    def save_image(self, count, user_name, image):
        """
        Info : Save detected person images.
        """
        try:
            image = cv2.resize(image, (self.width, self.height))
            path = f"{self.dataset_path}\\{user_name}\\"
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(
                f"{path}\\face_{user_name}_{str(count)}.jpg", image)
            print(f"{path}\\face_{user_name}_{str(count)}.jpg")
            count += 1

            return count
        except Exception as e:
            print('\nERROR: Image size is too small.\n', e)

    def generate_face_data(self, count, user_name, image):
        base_img = image.copy()
        detections, aspect_ratio_x, aspect_ratio_y = self.detect_face(
            image=image)
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
        try:
            detections_df = detections_df[detections_df["is_face"] == 1]
            detections_df = detections_df[detections_df["confidence"] >= 0.93]
            for i, instance in detections_df.iterrows():
                left = int(instance["left"] * 300)
                bottom = int(instance["bottom"] * 300)
                right = int(instance["right"] * 300)
                top = int(instance["top"] * 300)

                if len(detections_df) != 0:
                    detected_face = base_img[
                        int(top * aspect_ratio_y) - 100: int(bottom * aspect_ratio_y) + 100,
                        int(left * aspect_ratio_x) - 100: int(right * aspect_ratio_x) + 100,
                    ]
                    count = self.save_image(count, user_name, detected_face)

            return count
        except Exception as e:
            print("\nERROR :", e)

    def generate_face_data_from_webcam(self):
        """
        Info : Create dataset of person form webcam and store face-data into dataset_path folder.
        """
        try:

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            print("\nFace Detection start Look at camera and smile :)...\n")
            count = 0

            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)

            while True:
                _, image = cap.read()
                count = self.generate_face_data(count, self.user_name, image)
                if count == self.no_of_samples:
                    break
                # Stop if 'q' key is pressed
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print("\nERROR :", e)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            exit

    def generate_face_data_from_images(self, src_images_path):
        """
        Detect image from src_images_path folder and store face-data into dataset_path folder

        INFO : Make sure every image have only one person in it.
        """
        try:
            image_file_paths = []
            count = 0

            # check passed db folder exists
            if os.path.isdir(src_images_path) == True:
                # r=root, d=directories, f = files
                for r, d, f in os.walk(src_images_path):
                    for file in f:
                        if ".jpg" in file:
                            # exact_path = os.path.join(r, file)
                            exact_path = r + "\\" + file
                            image_file_paths.append(exact_path)
                print("\nImages Collected\n")

            num_of_images = len(image_file_paths)

            if num_of_images == 0:
                print(
                    f"WARNING: There is no image in this path {src_images_path}. Face data will not be generated."
                )
                exit
            for i, image_path in enumerate(image_file_paths):
                image = cv2.imread(image_path)
                print("I", i)
                count = self.generate_face_data(
                    i, image_path.split("\\")[-2], image)

        except Exception as e:
            print("\nERROR :", e)


if __name__ == '__main__':
    obj = FaceDetection()
