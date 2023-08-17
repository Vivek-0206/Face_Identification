from Face_Identification import FaceIdentificaion

while True:
    dataset_path = input("\nEnter full path of the dataset: ")
    if(dataset_path == ''):
        print("\n[INFO] -- Enter Dataset path.")
    else:
        break
    
while True:
    user_name = input("\nEnter name of the new user: ")
    if(user_name == ''):
        print("\n[INFO] -- Enter name of the new user")
    else:
        break
        
while True:
    no_of_samples = input("\nEnter number of samples: ")
    if(no_of_samples == ''):
        print("\n[INFO] -- Enter number of samples: ")
    elif int(no_of_samples) <= 0:
        print("\n[INFO] -- Enter number of samples: ")
    elif int(no_of_samples) > 100:
        no_of_samples = 100
    else:
        no_of_samples = int(no_of_samples)
        break

while True:

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
        break
    elif(flag == 1):
        obj = FaceIdentificaion()
        obj.face_detetion(dataset_path=dataset_path,
                        user_name=user_name, no_of_samples=no_of_samples, flag=True)
        break
    else:
        print("\nSelect one option")
