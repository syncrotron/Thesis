import pandas as pd
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import os

DATASTORE_PATH = "/datastore/AGILE/BELToutput/LDP_Sebastian"

# This script expects the following headers in the below csv: img,k,group_pred
INPUT_CSV_PATH = "/birl2/users/sch923/Thesis/Lentil/pattern_10000.csv"

SUB_IMG_X_Y = 200
IMGS_PER_ROW = 10
COLLAGE_X_Y = SUB_IMG_X_Y * IMGS_PER_ROW

OUTDIR = "/birl2/users/sch923/Thesis/Lentil/Collages"

FONT_PATH = "/birl2/users/sch923/times-new-roman.ttf"



# Cell used to grab all image locals, kgroups, and prediction groups from csv

predictions_df = pd.read_csv(INPUT_CSV_PATH)            


# Creates collage

max_k_list = predictions_df['k'].unique().tolist()
font = ImageFont.truetype(FONT_PATH, 64)
for max_k in max_k_list:
    # Filters predictions to only include the current max_k
    max_k_filtered_df = predictions_df[predictions_df['k'] == max_k]
    for pred in range(0, max_k):
        pred_sample_df = max_k_filtered_df[max_k_filtered_df['group_pred'] == pred].sample(n=IMGS_PER_ROW*IMGS_PER_ROW, random_state=1) # Random state was selected at random (manually)
        collage = Image.new("RGBA", (COLLAGE_X_Y,COLLAGE_X_Y))
        img_path_list = pred_sample_df['img'].tolist()
        for x in range(0, IMGS_PER_ROW):
            for y in range(0, IMGS_PER_ROW):
                img = Image.open(img_path_list.pop())
                img = img.resize((SUB_IMG_X_Y,SUB_IMG_X_Y))
                collage.paste(img, (x*SUB_IMG_X_Y,y*SUB_IMG_X_Y))
        
        
        # Label the image with interesting metadata and save
        draw = ImageDraw.Draw(collage)
        draw.text((0, 0),f" Total Groups = {max_k}, Group Prediction = {pred+1}",(255,255,255), font=font)
        collage.show()
        collage_name = f"totalk{max_k}_predk{pred+1}_lentil_collage.png"
        collage.save(os.path.join(OUTDIR,collage_name))
