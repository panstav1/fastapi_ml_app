from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

# Pydantic model for the POST request
class InferenceRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    edunation_int: str = Field(None, alias='my-field')
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

    return {"message": f"Data received: {request.workclass}", "param": request.age}
