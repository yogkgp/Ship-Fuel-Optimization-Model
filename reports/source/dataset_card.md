---
license: cc-by-nc-nd-4.0
configs:
- config_name: cps_poseidon
  data_files: CPS_Poseidon.parquet
- config_name: cps_triton
  data_files: CPS_Triton.parquet
- config_name: oss_ceto
  data_files: OSS_Ceto.parquet
tags:
- timeseries
- tabular
- Time Series Regression
- Marine
- Ship
- Fuel Consumption
pretty_name: FuelCast
task_categories:
- tabular-regression
- time-series-forecasting
---

# Dataset for the **FuelCast Benchmark - Benchmarking Tabular and Temporal Models for Ship Fuel Consumption**

The related paper is available on Springer Nature Link: https://link.springer.com/chapter/10.1007/978-3-032-15535-1_4

The preprint is available on ArXiv: https://arxiv.org/abs/2510.08217

## Dataset

The dataset was created using three main sources:
- High frequency auto logger data of important parameters from the ship, recorded by the [KROHNE EcoMATE™ system](https://www.krohne.com/en-us/solutions/monitoring-solutions/ecomate-fuel-emission-monitoring-seagoing-vessels).
- Environment data from [Copernicus Marine Environment Monitoring Service](https://marine.copernicus.eu/)
- Weather hintcast data from [OpenMeteo](https://open-meteo.com/)

<img src="https://cdn-uploads.huggingface.co/production/uploads/6883322683873f346bfaf8e5/SjFnr6B6Fqxkk9d-GBww_.png" width="600"/>

## Tasks

### Fuel Consumption Prediction

The main task of the FuelCast Benchmark is to predict (or better simulate) the total fuel consumption of a ship
under arbitrary operational and environmental conditions.

<img src="https://cdn-uploads.huggingface.co/production/uploads/6883322683873f346bfaf8e5/m7qzji8webtFwV6kzTsY3.png" width="400"/>

The chosen **input variables** `X` are:
- Ship_SpeedOverGround
- Environment_SeaFloorDepth
- Weather_Temperature2m
- Weather_WindSpeed10m
- Weather_WindDirection10m
- Weather_WaveHeight
- Weather_WaveDirection
- Weather_WavePeriod
- Weather_OceanCurrentVelocity
- Weather_OceanCurrentDirection

The **target variable** `y` to predict is
- Consumer_Total_MomentaryFuel

## Column Description

### Index

The data was sampled using 5-minute intervals. Each interval has a unique index which acts as a timestamp equivalent.

| Column | Data Type | Unit | Description |
| --- | --- | --- | --- |
| index | int64 | - | The 5-minutes timestep index of the data sample starting with 0. |

### Ship Data

| Column | Data Type | Unit | Description |
| --- | --- | --- | --- |
| Ship_SpeedOverGround | f64 | m / s | The speed over ground (SOG) of the vessel. |
| Ship_SpeedThroughWater | f64 | m / s | The speed through water (STW) of the vessel. |
| Ship_Heading | f64 | ° | The heading of the vessel, typically measured by a magnetic compass. |
| Ship_Bearing | f64 | ° | The bearing / track (direction of movement) of the vessel, calculated using GPS. |
| Ship_AnemometerWindDirection | f64 | ° | The measured wind direction (relative to the vessel speed/heading). |
| Ship_AnemometerWindSpeed | f64 | m / s | The measured wind speed (relative to the vessel). |
| Ship_DraftAft | f64 | m | The vessel draft aft |
| Ship_DraftFore | f64 | m | The vessel draft fore |

### Consumer Data

Data from all fuel consumers like engines or boilers.

| Column | Data Type | Unit | Description |
| --- | --- | --- | --- |
| Consumer_\<consumer-name>_FuelType | String | - | The type of the used fuel per consumer. E.g. DM, RM 380, ... |
| Consumer_\<consumer-name>_RotationSpeed | f64 | rpm | Rotation speed of the consumer shaft. Only available for engines. |
| Consumer_\<consumer-name>_ShaftPower | f64 | W | The output power of the consumer. Only available for engines. |
| Consumer_\<consumer-name>_MomentaryFuel | f64 | kg / s | The measured fuel consumption of the consumer. |
| Consumer_Total_MomentaryFuel | f64 | kg / s | The measured fuel consumption of all consumers together. |
| Consumer_Total_ShaftPower | f64 | W | The output power of all engines together. |

### Propeller Data

| Column | Data Type | Unit | Description |
| --- | --- | --- | --- |
| Propeller_\<consumer-name>_ShaftPower | f64 | W | The power of the propeller shaft. |
| Propeller_\<consumer-name>_RotationSpeed | f64 | rpm | The RPM of the propeller. |
| Propeller_\<consumer-name>_ShaftTorque | f64 | N m | The torque of the propeller shaft. |
| Propeller_Total_ShaftPower | f64 | W | The power of all propeller shafts together. |

### Environment Data

The environment data is based on reanalysis data ([Global Ocean Waves Analysis and Forecast](https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-027.pdf)) from [Copernicus Marine Environment Monitoring Service](https://marine.copernicus.eu/).

| Column | Data Type | Unit | Description |
| --- | --- | --- | --- |
| Environment_SeaFloorDepth | f64 | m | The sea floor depth below sea level (bathymetry) from Copernicus Marine. |

### Weather Data

The weather data is based on reanalysis data ([Historical Forecast](https://open-meteo.com/en/docs/historical-forecast-api) and [Marine Forecast](https://open-meteo.com/en/docs/marine-weather-api)) from satellite images and other sources. The data was taken from [OpenMeteo](https://open-meteo.com/).

| Column | Data Type | Unit | Description |
| --- | --- | --- | --- |
| Weather_Temperature2m            | f64       | °C                     | OpenMeteo: The air temperature at 2 meters above ground.                                         |
| Weather_RelativeHumidity2m       | f64       | \%                     | OpenMeteo: The relative humidity at 2 meters above ground.                                       |
| Weather_Precipitation            | f64       | mm                     | OpenMeteo: Total precipitation (rain, showers, snow) sum of the preceding hour.                  |
| Weather_WeatherCode              | f64       | WMO code               | OpenMeteo: Weather condition as a numeric code following WMO weather interpretation codes.       |
| Weather_SurfacePressure          | f64       | hPa                    | OpenMeteo: Atmospheric air pressure at surface level; decreases with elevation.                  |
| Weather_WindSpeed10m             | f64       | m/2                    | OpenMeteo: Wind speed at 10 meters above ground.                                                 |
| Weather_WindDirection10m         | f64       | °                      | OpenMeteo: Wind direction at 10 meters above ground.                                             |
| Weather_WindGusts10m             | f64       | m/s                    | OpenMeteo: Maximum wind gusts at 10 meters above ground over the preceding hour.                 |
| Weather_ShortwaveRadiation       | f64       | W / m^2                | OpenMeteo: Average shortwave solar radiation over the preceding hour.                            |
| Weather_DirectRadiation          | f64       | W / m^2                | OpenMeteo: Average direct solar radiation on the horizontal plane over the preceding hour.       |
| Weather_DiffuseRadiation         | f64       | W / m^2                | OpenMeteo: Average diffuse solar radiation over the preceding hour.                              |
| Weather_DirectNormalIrradiance   | f64       | W / m^2                | OpenMeteo: Average direct solar radiation on the normal plane over the preceding hour.           |
| Weather_SunshineDuration         | f64       | s                      | OpenMeteo: Sum of sunshine duration within the preceding hour.                                   |
| Weather_WaveHeight               | f64       | m                      | OpenMeteo: Significant mean wave height.                                                         |
| Weather_WaveDirection            | f64       | °                      | OpenMeteo: Mean direction of significant waves.                                                  |
| Weather_WavePeriod               | f64       | s                      | OpenMeteo: Period between significant waves.                                                     |
| Weather_WindWaveHeight           | f64       | m                      | OpenMeteo: Significant wind wave height.                                                         |
| Weather_WindWaveDirection        | f64       | °                      | OpenMeteo: Mean direction of wind waves.                                                         |
| Weather_WindWavePeriod           | f64       | s                      | OpenMeteo: Period between wind waves.                                                            |
| Weather_WindWavePeakPeriod       | f64       | s                      | OpenMeteo: Peak period between wind waves.                                                       |
| Weather_SwellWaveHeight          | f64       | m                      | OpenMeteo: Significant swell wave height.                                                        |
| Weather_SwellWaveDirection       | f64       | °                      | OpenMeteo: Mean direction of swell waves.                                                        |
| Weather_SwellWavePeriod          | f64       | s                      | OpenMeteo: Period between swell waves.                                                           |
| Weather_SwellWavePeakPeriod      | f64       | s                      | OpenMeteo: Peak period between swell waves.                                                      |
| Weather_OceanCurrentVelocity     | f64       | m / s                  | OpenMeteo: Ocean current velocity considering all components.                                    |
| Weather_OceanCurrentDirection    | f64       | °                      | OpenMeteo: Direction of the ocean current flow.                                                  |

## Citation

```
@InProceedings{10.1007/978-3-032-15535-1_4,
    author="Viga, Justus
    and Mueck, Penelope
    and L{\"o}ser, Alexander
    and Weis, Torben",
    editor="Lemaire, Vincent
    and Ifrim, Georgiana
    and Bagnall, Anthony
    and Malinowski, Simon
    and Sch{\"a}fer, Patrick
    and Tavenard, Romain",
    title="FuelCast: Benchmarking Tabular and Temporal Models for Ship Fuel Consumption",
    booktitle="Advanced Analytics and Learning on Temporal Data",
    year="2026",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="54--69",
    abstract="In the shipping industry, fuel consumption and emissions are critical factors due to their significant impact on economic efficiency and environmental sustainability. Accurate prediction of ship fuel consumption is essential for further optimization of maritime operations. However, heterogeneous methodologies and limited high-quality datasets hinder direct comparison of modeling approaches. This paper makes three key contributions: (1) we introduce and release a new dataset (https://huggingface.co/datasets/krohnedigital/FuelCast) comprising operational and environmental data from three ships; (2) we define a standardized benchmark covering tabular regression and time-series regression (3) we investigate the application of in-context learning for ship consumption modeling using the TabPFN foundation model - a first in this domain to our knowledge. Our results demonstrate strong performance across all evaluated models, supporting the feasibility of onboard, data-driven fuel prediction. Models incorporating environmental conditions consistently outperform simple polynomial baselines relying solely on vessel speed. TabPFN slightly outperforms other techniques, highlighting the potential of foundation models with in-context learning capabilities for tabular prediction. Furthermore, including temporal context improves accuracy.",
    isbn="978-3-032-15535-1"
}
```


