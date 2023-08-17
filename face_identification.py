from Face_Identification import FaceIdentificaion

dataset_path = input("\nEnter full path of the dataset: ")

obj = FaceIdentificaion()
user_name = obj.face_recognition(dataset_path)
print(f"Welcome {user_name}")