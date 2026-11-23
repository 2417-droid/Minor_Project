# Dataset Report: UCI Individual Household Electric Power Consumption

## Source and Provenance

- **Repository:** UCI Machine Learning Repository
- **Dataset URL:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- **Direct download:** https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
- **DOI:** 10.24432/C58K54
- **Creators:** Georges Hebrail, Alice Berard
- **Donation date:** August 29, 2012
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## Dataset Description

Measurements of electric power consumption in one household located in Sceaux
(7 km south of Paris, France) collected over almost 4 years (December 2006 to
November 2010, 47 months).

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| Type | Multivariate, Time-Series |
| Subject area | Physics and Chemistry |
| Associated tasks | Regression, Clustering |
| Feature type | Real |
| Instances | 2,075,259 |
| Features | 9 |
| Sampling rate | 1 minute |
| Has missing values | Yes (~1.25%) |

## Data Dictionary

| # | Column | Role | Type | Unit | Description |
|---|--------|------|------|------|-------------|
| 1 | Date | Feature | Date | - | Format: dd/mm/yyyy |
| 2 | Time | Feature | Categorical | - | Format: hh:mm:ss |
| 3 | Global_active_power | Feature | Continuous | kW | Household global minute-averaged active power |
| 4 | Global_reactive_power | Feature | Continuous | kW | Household global minute-averaged reactive power |
| 5 | Voltage | Feature | Continuous | V | Minute-averaged voltage |
| 6 | Global_intensity | Feature | Continuous | A | Household global minute-averaged current intensity |
| 7 | Sub_metering_1 | Feature | Continuous | Wh | Energy sub-metering No. 1 (kitchen: dishwasher, oven, microwave) |
| 8 | Sub_metering_2 | Feature | Continuous | Wh | Energy sub-metering No. 2 (laundry: washing-machine, tumble-drier, refrigerator, light) |
| 9 | Sub_metering_3 | Feature | Continuous | Wh | Energy sub-metering No. 3 (electric water-heater and air-conditioner) |

### Derived Variable

The UCI documentation notes that:
```
(global_active_power * 1000 / 60 - sub_metering_1 - sub_metering_2 - sub_metering_3)
```
represents the active energy consumed every minute (in watt-hour) by electrical
equipment not measured in sub-meterings 1, 2, and 3.

## Data Quality Assessment

### Missing Values

- ~1.25% of rows have missing values (25,979 out of 2,075,259).
- Missing values are encoded as `?` in the raw text file.
- Missing values occur across all numeric columns simultaneously (i.e., when
  a timestamp is present but measurements are absent).
- Specific example: April 28, 2007 has missing measurements.
- **Imputation strategy:** Forward-fill then backward-fill. This is a
  conservative approach for time-series that does not use future information
  for the current row.

### Duplicates

- Duplicate timestamps exist in the raw data.
- **Resolution:** Rows with the same timestamp are averaged using
  `groupby(level=0).mean()`.

### Outliers

- High consumption spikes (up to ~6.5 kW hourly average) are genuine
  consumption events (e.g., multiple appliances running simultaneously).
- We do **not** remove outliers because they represent real household
  behavior that the model should learn to forecast.

## Preprocessing Applied

### Hourly Resampling

The 1-minute data is resampled to hourly frequency with the following
aggregation rules:

| Column | Aggregation | Rationale |
|--------|-------------|-----------|
| Global_active_power | Mean | Hourly mean of kW = hourly kWh |
| Global_reactive_power | Mean | Average reactive power |
| Voltage | Mean | Average voltage |
| Global_intensity | Mean | Average current |
| Sub_metering_1/2/3 | Sum | Total Wh consumed in the hour |

**Why mean for Global_active_power?** The raw value is minute-averaged active
power in kW. The hourly mean of minute-averaged kW equals the hourly energy
consumption in kWh, because each minute contributes
`power_kW * (1/60) h` of energy and there are 60 minutes per hour:

```
hourly_kWh = sum(power_minute * 1/60) / 1 = mean(power_minute)
```

### After Preprocessing

| Metric | Value |
|--------|-------|
| Hourly rows | 34,589 |
| Date range | 2006-12-16 17:00 to 2010-11-26 21:00 |
| Missing values | 0 (after imputation) |
| Columns | 7 |

## Suitability for the Project

### Strengths

1. **Household-level**: Data is from a single household, matching the problem
   statement for individual household forecasting.
2. **Rich features**: Includes active power, reactive power, voltage, current,
   and sub-metering for different appliance groups.
3. **Long time span**: ~4 years of data captures daily, weekly, and seasonal
   patterns.
4. **High frequency**: 1-minute sampling allows flexible resampling.
5. **Publicly available**: CC BY 4.0 license allows free use with attribution.
6. **Well-cited**: Used in 29+ academic papers, validating its quality.

### Limitations

1. **Single household**: Only one home is measured. The model cannot
   generalize to other households without retraining.
2. **No weather data**: Temperature, humidity, and solar radiation are not
   included, though they significantly affect consumption.
3. **No occupancy information**: We do not know when the house is occupied or
   empty, which strongly influences consumption.
4. **Old data (2006-2010)**: Consumption patterns may have changed with modern
   appliances, EVs, and smart home systems.
5. **No appliance-level data**: Sub-metering groups are coarse (kitchen,
   laundry, water-heater/AC). Individual appliance signatures are not
   available.
6. **Location-specific**: A house near Paris may not represent households in
   other climates or cultural contexts.

## License and Usage

This dataset is licensed under the Creative Commons Attribution 4.0
International (CC BY 4.0) license. This allows sharing and adaptation for any
purpose, provided appropriate credit is given.

### Citation

```
Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54
```
