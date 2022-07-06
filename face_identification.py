from face_identification import FaceIdentificaion

dataset_path = input("Enter full path of the dataset: ")

obj = FaceIdentificaion()
user_name = obj.face_recognition(dataset_path)