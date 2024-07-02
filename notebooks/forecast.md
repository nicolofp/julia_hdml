Electricity forecast
================

## Dataset

As of 2020, 6.7 million people reside in Victoria, Australia’s second
most populated state. Most of them, 5 million, live or work in
Melbourne, state’s capital. During 2020 Australia was among the first to
close international borders, followed by a closure of interstate
borders. Victoria introduced some of the strictest pandemic-related
restrictions on business activity that resulted in a significant portion
of its population working from home.

The dataset covers 2016 days between 1 January 2015 and 6 October 2020.
During some intraday intervals RRP was negative, so that energy
producers were paying consumers rather than wise versa. Below is a brief
description fo the data:

- **date**: datetime, the date of the recording
- **demand**: float, a total daily electricity demand in MWh
- **RRP**: float, a recommended retail price in AUD\$/MWh
- **demand_pos_RRP**: float, a total daily demand at positive RRP in MWh
- **RRP_positive**: float, an averaged positive RRP, weighted by the
  corresponding intraday demand in AUD\$/MWh
- **demand_neg_RRP**: float, an total daily demand at negative RRP in
  MWh
- **RRP_negative**: float, an average negative RRP, weighted by the
  corresponding intraday demand in AUD\$/MWh
- **frac_at_neg_RRP**: float, a fraction of the day when the demand was
  traded at negative RRP
- **min_temperature**: float, minimum temperature during the day in
  Celsius
- **max_temperature**: float, maximum temperature during the day in
  Celsius
- **solar_exposure**: float, total daily sunlight energy in
  MJ/m<sup>2</sup>
- **rainfall**: float, daily rainfall in mm
- **school_day**: boolean, if students were at school on that day
- **holiday**: boolean, if the day was a state or national holiday
