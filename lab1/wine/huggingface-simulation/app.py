import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

columns = ['type_white','fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','alcohol']


def get_image_with_text(text):
    img = Image.new(size=(200,200),mode="RGBA", color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=150)
    draw.text((50,0), text, font=font, fill=(0, 0, 0, 0))
    return img

def wine(type_white,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[type_white,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,alcohol]], 
                      columns=columns)
    print("Predicting")
    print(df)
    res = model.predict(df)
    res_text = f'the predicted quality of your wine was {res[0]}'
    print(res)

    img = get_image_with_text(f'{res[0]}')
    return img,res_text
        
demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with the features to predict which quality the wine is.",
    allow_flagging="never",
    inputs=[
        gr.Number(value=1.0, label='is_white_wine (1/0)'),
        gr.Number(value=7.0, label=columns[1]),
        gr.Number(value=0.20, label=columns[2]),
        gr.Number(value=0.25, label=columns[3]),
        gr.Number(value=9.0, label=columns[4]),
        gr.Number(value=0.05, label=columns[5]),
        gr.Number(value=40.0, label=columns[6]),
        gr.Number(value=150.0, label=columns[7]),
        gr.Number(value=0.9950, label=columns[8]),
        gr.Number(value=10.0, label=columns[9]),
        ],
    outputs=[
        gr.Image(type="pil"),
        gr.Text(),
        ]
    )

demo.launch(debug=True)