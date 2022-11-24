import pandas as pd
import mesa
import time
from schelling import SchellingAgent, SchellingModel



params = {"side": [30], "density":[.3,.5],
			"n_towns":3, 
			"mobility": [
                                                 {"model" :"radiation", "metric":"relevance"}
                                     ]}

iterations = 10

results = mesa.batch_run(
    SchellingModel,
    parameters=params,
    iterations=iterations,
    max_steps=100,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)

results = pd.DataFrame(results)

t = f'{time.localtime().tm_year}_{time.localtime().tm_mon}_{time.localtime().tm_mday}_{time.localtime().tm_hour}_{time.localtime().tm_min}'

results.to_csv(f'{t}.csv')
