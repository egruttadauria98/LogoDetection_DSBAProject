import os
import pandas as pd
import re

#csv is the previous file saved

#possibilità: 
#   1) creazione di folder in base al threshold ex. thrs = 0.5/ 0.8 ...
#   2) tutto nella stessa directory
# counter pre fixed is needed to understand if we need to create a new or taking and existed one

def build_predicion_CSV (threshold , name_image , counter, detections, category):

    #if "csv prediction" not in os.listdir():
    #    os.mkdir("csv prediction")
    
    sub = re.compile(r"/home/labuser/LogoDet/LogoDetection_DSBAProject/training_process/training/INFERENCE_DIR/")
    name_image = re.sub(sub,"",name_image)
    
    info = ["filename","class","probability","ymin","xmin","ymax","xmax"]

    accuracy = detections['detection_scores'][0].numpy()
    boxes = detections['detection_boxes'][0].numpy()
    classes = (detections['detection_classes'][0].numpy() + 1).astype(int)
    mask_accuracy = accuracy >= threshold

    true_accuracy = accuracy[mask_accuracy]
    true_boxes = boxes[mask_accuracy, :]
    true_classes = classes[mask_accuracy]
    

    if counter == 0:
        df_csv = pd.DataFrame(columns=info)
        
    else:
        df_csv = pd.read_csv(f"/home/labuser/LogoDet/LogoDetection_DSBAProject/training_process/training/csv prediction/prediction_bounding_boxes_{threshold}.csv")
        df_csv = df_csv
    lenght = len (true_accuracy)
    for num in range(lenght):




        collection_image={}
        collection_image[name_image]={}
        collection_image[name_image][num]={}
        collection_image[name_image][num]["label"] = true_classes[num]  
        collection_image[name_image][num]["accuracy"] = true_accuracy[num] 
        collection_image[name_image][num]["prediction"] = [true_boxes[num,:][0],true_boxes[num,:][1],true_boxes[num,:][2],true_boxes[num,:][3]]
        print(collection_image)
        
        
        #dict_for_df = {
        #    info[0] : name_image,
        #    info[1] : collection_image[name_image][num]["label"],
        #    info[2] : collection_image[name_image][num]["accuracy"],
        #    info[3] : collection_image[name_image][num]["prediction"][0],
        #    info[4] : collection_image[name_image][num]["prediction"][1],
        #    info[5] : collection_image[name_image][num]["prediction"][2],
        #    info[6] : collection_image[name_image][num]["prediction"][3]
        #}
        
        dict_for_df = [
            name_image,
            category[collection_image[name_image][num]["label"]]["name"],
            collection_image[name_image][num]["accuracy"],
            collection_image[name_image][num]["prediction"][0],
            collection_image[name_image][num]["prediction"][1],
            collection_image[name_image][num]["prediction"][2],
            collection_image[name_image][num]["prediction"][3]
        ]
        
        element = pd.DataFrame([dict_for_df], columns=info)
       
        df_csv = pd.concat([df_csv,element],ignore_index=True)
    df_csv.to_csv(f"/home/labuser/LogoDet/LogoDetection_DSBAProject/training_process/training/csv prediction/prediction_bounding_boxes_{threshold}.csv",index=False)