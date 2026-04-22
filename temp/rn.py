import os

list_dir = os.listdir("../dataset")
list_dir.sort()
print(len(list_dir))
list_dir = list_dir[1:]

for food_name in list_dir :
    try:
        image_list = os.listdir(f"{food_name}/images")        
    except:
        pass
    print(len(image_list))
    if len(image_list) < 100 :
        print(food_name)