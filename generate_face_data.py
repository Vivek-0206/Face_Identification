from face_identification import FaceIdentificaion

dataset_path = input("Enter full path of the dataset: ")
user_name = input("Enter name of the new user: ")
no_of_samples = int(input("Enter number of samples: "))


obj = FaceIdentificaion()
obj.face_detetion(dataset_path=dataset_path,
                  user_name=user_name, no_of_samples=no_of_samples,flag=False)
