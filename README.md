# forecast_rnn
## Description
A simple rnn to predict weather from multiple features. Current latest version 0.5 uses RNN and produces 1500-2300 MSE. Use ipynb or python script to run. ipynb directly runs on github.



## Scores
All scores are **TRUE MSE** values. This means that they have been upscaled or denormalised to true values of y rather than kept between 0-1. The current method to do this is to multiply all predictions by a constant found originally as np.max(y_input).

|                        | RNN  | CNN   | Lin Reg | Support Vector |
|------------------------|------|-------|---------|----------------|
| Round 1 (Last 55 rows) | 1600 :smirk: | ~4000 :worried: | ~3200  :relieved: | 53000!! :dizzy_face:  |
|                        |      |       |         |                |
|                        |      |       |         |                |
