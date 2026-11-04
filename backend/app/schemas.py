from pydantic import BaseModel, Field, model_validator


class SensorDataInput(BaseModel):
    Air_temperature_K: float = Field(..., gt=0, le=500)
    Process_temperature_K: float = Field(..., gt=0, le=500)
    Rotational_speed_rpm: float = Field(..., gt=0, le=10000)
    Torque_Nm: float = Field(..., ge=0, le=200)
    Tool_wear_min: float = Field(..., ge=0, le=1000)

    @model_validator(mode="after")
    def validate_temperatures(self):
        if self.Process_temperature_K < self.Air_temperature_K:
            raise ValueError(
                "Process temperature should not be lower than air temperature."
            )
        return self

    def to_model_input(self):
        return {
            "Air temperature [K]": self.Air_temperature_K,
            "Process temperature [K]": self.Process_temperature_K,
            "Rotational speed [rpm]": self.Rotational_speed_rpm,
            "Torque [Nm]": self.Torque_Nm,
            "Tool wear [min]": self.Tool_wear_min,
        }