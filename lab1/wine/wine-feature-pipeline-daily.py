import os
# import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("iris_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("jim-hopsworks-ai"))
   def f():
       g()

def generate_wine(quality, df):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    # Create bounds based on current class values
    mask = df['quality'] == quality
    quality_df = df[mask]
    quality_df = quality_df.drop('quality', axis=1)

    wine_df = pd.DataFrame()

    for col in quality_df:
        max_col_value = quality_df[col].max()
        min_col_value = quality_df[col].min()
        wine_df[col] = [random.uniform(max_col_value, min_col_value)]

    # We need to redo the type_white since it is one-hot encoded
    wine_df['type_white'] = random.randint(0,1)
    wine_df['quality'] = quality

    return wine_df


def get_random_wine(df):
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    random_quality = random.randint(3,9)
    random_wine_df = generate_wine(random_quality, df)
    
    print(f'A wine with quality {random_quality} added')

    return random_wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_df = wine_fg.read()
    wine_df.to_csv('winedf.csv')

    random_wine_df = get_random_wine(wine_df)

    wine_fg.insert(random_wine_df)


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("iris_daily")
        with stub.run():
            f()
