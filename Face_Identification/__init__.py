from face_identification.face_detection import face_detection
from face_identification.face_recognition import face_recognition


class FaceIdentificaion():
    """Face Identification class."""

    def __init__(self):
        pass

    def face_detetion(self,
                      dataset_path=r'',
                      user_name='testuser',
                      no_of_samples=5,
                      flag=False,
                      src_images_path=''):
        """
        Info : Dataset Create by face detection with OpenCV SSD.

        :param dataset_path: str (example: 'path_of_dataset')
        :param user_name: str (example: 'name_of_user')
        :param no_of_samples: int (example: 5)
        :param flag: bool (example: False for webcam, True for images)

        :return: None
        """
        if(obj.dataset_path == ''):
            print("Enter dataset path")
            return
        obj = face_detection.FaceDetection(
            dataset_path, user_name, no_of_samples)
        if(flag == False):
            obj.generate_face_data_from_webcam()
        else:
            if(src_images_path == ''):
                return
            obj.generate_face_data_from_images(src_images_path)

    def face_recognition(self, dataset_path=r''):
        """
        Info : Face_Recognition class

        :param dataset_path : str (example: 'Path_to_Data')

        :return: final person name
        """

        obj = face_recognition.FaceRecognition(dataset_path)

        person_name = obj.predict_person()

        return person_name


if __name__ == '__main__':
    obj = FaceIdentificaion()
