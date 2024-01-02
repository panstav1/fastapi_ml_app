from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ml.data import process_data, load_encoders
from ml.model import inference, load_model
import constants
import os
import pandas as pd

app = FastAPI()

@app.on_event("startup")
async def load_components():

    # Load the model, encoder, and label binarizer
    app.state.model = load_model(os.path.join(os.path.abspath(constants.model_folder),
                                              constants.model_file
                                              ))

    app.state.encoder, app.state.lb = load_encoders (os.path.join (os.path.abspath(constants.model_folder),
                                                                   constants.model_encoder),
                                                        os.path.join (os.path.abspath(constants.model_folder),
                                                                      constants.label_bin))

# Pydantic model for the POST request
class InferenceRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    edunation_int: str = Field(None, alias='education-int')
    marital: str = Field(None, alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(0, alias='capital-gain')
    capital_loss: int = Field (0, alias='capital-loss')
    hours_per_week: int = Field (0, alias='hours-per-week')
    native_country: str = Field (0, alias='native-country')

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Husband',
                'race':'White',
                'sex':'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }


# GET route on the root domain
@app.get("/")
async def read_root():
    return {"message": "Hello, welcome to our API!"}


# POST route for model inference
@app.post("/infer/")
async def do_inference(request: InferenceRequest):

    # Proces the test data with the process_data function.
    request_dict = request.model_dump (by_alias=True)

    # Create a DataFrame from the dictionary
    request_df = pd.DataFrame ([request_dict])

    X_test, _, _,_ = process_data (
        request_df, categorical_features=constants.cat_features, training=False, encoder=app.state.encoder, lb=app.state.lb
    )

    y_pred = inference (app.state.model, X_test)
    return {"Inference": f"{app.state.lb.inverse_transform(y_pred)}"}
