from face_identification import FaceIdentificaion

dataset_path = input("\nEnter full path of the dataset: ")
user_name = input("\nEnter name of the new user: ")
no_of_samples = int(input("\nEnter number of samples: "))

print("""
Please Select one
0 --> From Webcam
1 --> From Images --> INFO : Make sure every image have only one person in it.
""")
flag = int(input())

if(dataset_path == ''):
    print("\nEnter Dataset path: ")
    exit

if(flag == 0):
    obj = FaceIdentificaion()
    obj.face_detetion(dataset_path=dataset_path,
                      user_name=user_name, no_of_samples=no_of_samples, flag=False)
elif(flag == 1):
    obj = FaceIdentificaion()
    obj.face_detetion(dataset_path=dataset_path,
                      user_name=user_name, no_of_samples=no_of_samples, flag=True)
else:
    print("\nSelect one option")
    exit
