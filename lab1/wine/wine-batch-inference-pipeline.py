import os
# import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def get_image_with_text(text):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new(size=(200,200),mode="RGBA", color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=150)
    draw.text((50,0), text, font=font, fill=(0, 0, 0, 0))
    return img

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image, ImageDraw, ImageFont
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    offset = 1
    predicted = y_pred[y_pred.size-offset]
    print(f'Quality predicted: {predicted}')

    img = get_image_with_text(f'{predicted}')
    img.save('images/latest_wine.png')
    dataset_api = project.get_dataset_api()
    dataset_api.upload("images/latest_wine.png", "Resources/images", overwrite=True)

    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 
    label = int(df.iloc[-offset]['quality'])
    
    img = get_image_with_text(f'{label}')
    img.save("images/actual_wine.png")
    dataset_api.upload("images/actual_wine.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                            version=1,
                                            primary_key=["datetime"],
                                            description="Wine quality Prediction/Outcome Monitoring"
                                            )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
    'prediction': [predicted],
    'label': [label],
    'datetime': [now],
    }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, 'images/df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("images/df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    print("Number of different quality predictions to date: " + str(predictions.value_counts().count()))
    results = confusion_matrix(labels, predictions)

    rows = [f'True_{classNumber}' for classNumber in range(3,10)]
    cols = [f'Pred_{classNumber}' for classNumber in range(3,10)]
    
    confusion_matrix_df = pd.DataFrame(results,rows,cols)
    g = sns.heatmap(confusion_matrix_df,annot=True,lw=.5)
    fig = g.get_figure()
    fig.savefig("images/confusion_matrix.png")
    dataset_api.upload("images/confusion_matrix.png", "Resources/images", overwrite=True)


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

